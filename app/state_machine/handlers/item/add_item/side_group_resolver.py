# app/state_machine/handlers/item/add_item/side_group_resolver.py
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.utils.candidate_texts import build_candidate_texts_normalized
from app.utils.token_matcher import is_controlled_partial_match, is_strong_token_match, tokenize

AUTO_ACCEPT_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.60
MIN_CONFIRM_GAP = 0.08


@dataclass(frozen=True, slots=True)
class SideGroupMatch:
    matched_item_ids: list[str]
    matched_names: list[str]
    unmatched_values: list[str]


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = (value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def remove_leading_side_filler(text: str) -> str:
    filler_words = {
        "the",
        "a",
        "an",
        "with",
        "add",
        "please",
        "thanks",
        "thank",
        "you",
        "and",
        "just",
        "um",
        "uh",
        "okay",
        "ok",
        "ill",
        "i",
        "want",
        "take",
        "have",
        "get",
        "like",
        "would",
        "id",
        "said",
        "mean",
        "my",
        "will",
        "side",
        "sides",
    }
    tokens = [token for token in (text or "").split() if token not in filler_words]
    return " ".join(tokens).strip()


def extract_side_slot_values_normalized(context) -> list[str]:
    slots = context.last_slots or ()
    values: list[str] = []
    seen: set[str] = set()

    for slot in slots:
        name = str(slot.name).upper()
        if name not in {"SIDE", "ITEM", "MENU_ITEM"}:
            continue

        value = slot.value
        if not isinstance(value, str):
            continue

        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        values.append(normalized)

    return values


class SideGroupResolver:
    """
    Resolve one or more side choices strictly from the active side group.

    Design goals:
    - no full-menu search
    - only current group choices may match
    - allow multiple side matches from one utterance
    - preserve spoken order
    """

    def resolve(
        self,
        *,
        group,
        normalized_user_text: str,
        normalized_slot_values: list[str],
        already_selected_ids: list[str] | None = None,
    ) -> SideGroupMatch:
        already_selected_ids = already_selected_ids or []

        candidate_values = self._build_candidate_values(
            normalized_user_text=normalized_user_text,
            normalized_slot_values=normalized_slot_values,
        )
        if not candidate_values:
            return SideGroupMatch(
                matched_item_ids=[],
                matched_names=[],
                unmatched_values=[],
            )

        matched_item_ids: list[str] = []
        matched_names: list[str] = []
        unmatched_values: list[str] = []

        for candidate in candidate_values:
            scored = self._resolve_best_choice_for_candidate(group=group, candidate=candidate)
            if scored is None:
                unmatched_values.append(candidate)
                continue

            item_id, choice_name, confidence = scored
            if confidence < AUTO_ACCEPT_THRESHOLD:
                unmatched_values.append(candidate)
                continue

            if item_id in already_selected_ids or item_id in matched_item_ids:
                continue

            matched_item_ids.append(item_id)
            matched_names.append(choice_name)

        return SideGroupMatch(
            matched_item_ids=matched_item_ids,
            matched_names=matched_names,
            unmatched_values=dedupe_keep_order(unmatched_values),
        )

    def _build_candidate_values(
        self,
        *,
        normalized_user_text: str,
        normalized_slot_values: list[str],
    ) -> list[str]:
        full_candidates = dedupe_keep_order(
            [remove_leading_side_filler(value) for value in normalized_slot_values if value]
        )

        if normalized_user_text:
            cleaned_text = remove_leading_side_filler(normalized_user_text)
            if cleaned_text and cleaned_text not in full_candidates:
                full_candidates.append(cleaned_text)

        split_candidates = build_candidate_texts_normalized(
            normalized_user_text=normalized_user_text,
            normalized_slot_values=normalized_slot_values,
            allow_split=True,
        )
        split_candidates = dedupe_keep_order(
            [remove_leading_side_filler(value) for value in split_candidates if value]
        )

        return dedupe_keep_order(full_candidates + split_candidates)

    def _resolve_best_choice_for_candidate(
        self,
        *,
        group,
        candidate: str,
    ) -> tuple[str, str, float] | None:
        best_choice = None
        best_confidence = 0.0
        second_confidence = 0.0

        for choice in group.choices:
            confidence = self._choice_confidence(candidate, choice.normalized_name)
            if confidence > best_confidence:
                second_confidence = best_confidence
                best_confidence = confidence
                best_choice = choice
            elif confidence > second_confidence:
                second_confidence = confidence

        if best_choice is None:
            return None

        if best_confidence < CONFIRM_THRESHOLD:
            return None

        if (
            best_confidence < AUTO_ACCEPT_THRESHOLD
            and (best_confidence - second_confidence) < MIN_CONFIRM_GAP
        ):
            return None

        return (best_choice.item_id, best_choice.name, best_confidence)

    def _choice_confidence(self, candidate: str, choice_name: str) -> float:
        if not candidate or not choice_name:
            return 0.0

        if candidate == choice_name:
            return 1.0

        best = 0.0
        candidate_tokens = set(tokenize(candidate))
        choice_tokens = set(tokenize(choice_name))

        if candidate_tokens and choice_tokens:
            overlap = len(candidate_tokens & choice_tokens)
            coverage = overlap / len(choice_tokens)
            candidate_coverage = overlap / len(candidate_tokens)
            best = max(best, max(coverage, candidate_coverage))

        if is_strong_token_match(candidate, choice_name):
            best = max(best, 0.92)

        if is_controlled_partial_match(candidate, choice_name):
            best = max(best, 0.82)

        best = max(best, SequenceMatcher(None, candidate, choice_name).ratio())
        return best