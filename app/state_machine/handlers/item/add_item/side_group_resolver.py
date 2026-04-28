# app/state_machine/handlers/item/add_item/side_group_resolver.py
from __future__ import annotations

from dataclasses import dataclass
import re

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.option_matching import (
    OptionCandidate,
    build_match_debug_payload,
    build_scoped_phrase_candidates,
    build_slot_first_option_candidates,
    extract_slot_candidate_texts,
    score_scoped_choice,
    strip_common_option_fillers,
)
from app.utils.candidate_texts import build_candidate_texts_normalized
from app.utils.token_matcher import tokenize

AUTO_ACCEPT_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.60
MIN_CONFIRM_GAP = 0.08
SIZE_CONTEXT_WORDS = {
    "small",
    "medium",
    "large",
    "regular",
    "mini",
    "xl",
    "extra",
}
NEGATION_WORDS = {"no", "without"}


@dataclass(frozen=True, slots=True)
class SideGroupMatch:
    matched_item_ids: list[str]
    matched_names: list[str]
    unmatched_values: list[str]
    match_debug: dict[str, object] | None = None


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
        "oka",
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
    return extract_slot_candidate_texts(
        slots=context.last_slots or (),
        allowed_slot_labels={"SIDE", "VARIANT"},
        cleaner=_normalize_side_candidate,
    )


def build_side_option_candidates(context, normalized_user_text: str) -> list[OptionCandidate]:
    return build_slot_first_option_candidates(
        raw_utterance=normalized_user_text,
        slots=context.last_slots or (),
        allowed_slot_labels={"SIDE", "VARIANT"},
        cleaner=_normalize_side_candidate,
        allow_split=True,
    )


def _normalize_side_candidate(text: str) -> str:
    return remove_leading_side_filler(strip_common_option_fillers(text))


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
        normalized_slot_values: list[str] | None = None,
        option_candidates: list[OptionCandidate] | None = None,
        already_selected_ids: list[str] | None = None,
    ) -> SideGroupMatch:
        already_selected_ids = already_selected_ids or []
        candidates = option_candidates or self._build_candidate_values(
            normalized_user_text=normalized_user_text,
            normalized_slot_values=normalized_slot_values or [],
        )
        candidates = self._augment_candidates_with_group_phrases(
            candidates=candidates,
            group=group,
            normalized_user_text=normalized_user_text,
        )
        if not candidates:
            return SideGroupMatch(
                matched_item_ids=[],
                matched_names=[],
                unmatched_values=[],
                match_debug=build_match_debug_payload(
                    raw_utterance=normalized_user_text,
                    candidates=[],
                    selected_candidate=None,
                    matched_option=None,
                    match_source=None,
                    match_score=None,
                ),
            )

        matched_item_ids: list[str] = []
        matched_names: list[str] = []
        unmatched_values: list[str] = []
        slot_candidates_present = any(
            candidate.source in {"slot_value", "slot_raw"}
            for candidate in candidates
        )
        debug_candidate_text: str | None = None
        debug_match_source: str | None = None
        debug_matched_option: str | None = None
        debug_match_score: float | None = None

        for candidate in candidates:
            scored = self._resolve_best_choice_for_candidate(
                group=group,
                candidate=candidate.text,
            )
            if scored is None:
                if debug_candidate_text is None:
                    debug_candidate_text = candidate.text
                    debug_match_source = candidate.source
                if candidate.source != "raw_utterance" or not slot_candidates_present:
                    unmatched_values.append(candidate.text)
                continue

            item_id, choice_name, confidence = scored
            if debug_candidate_text is None:
                debug_candidate_text = candidate.text
                debug_match_source = candidate.source
                debug_matched_option = choice_name
                debug_match_score = confidence
            if confidence < AUTO_ACCEPT_THRESHOLD:
                if candidate.source != "raw_utterance" or not slot_candidates_present:
                    unmatched_values.append(candidate.text)
                continue

            if item_id in already_selected_ids or item_id in matched_item_ids:
                continue

            matched_item_ids.append(item_id)
            matched_names.append(choice_name)
            if debug_match_score is None or confidence >= debug_match_score:
                debug_candidate_text = candidate.text
                debug_match_source = candidate.source
                debug_matched_option = choice_name
                debug_match_score = confidence

        if not matched_item_ids and normalized_user_text:
            greedy_ids, greedy_names = self._greedy_scan_for_embedded_sides(
                group=group,
                text=normalized_user_text,
                already_selected_ids=already_selected_ids,
            )
            for item_id, choice_name in zip(greedy_ids, greedy_names):
                if item_id in matched_item_ids:
                    continue
                matched_item_ids.append(item_id)
                matched_names.append(choice_name)

        # ── Clean up unmatched: remove composite strings whose tokens
        #    are fully covered by matched + other unmatched tokens ──
        matched_tokens: set[str] = set()
        for name in matched_names:
            matched_tokens.update(tokenize(normalize_text(name)))

        # First pass: keep only values with at least one non-matched token
        first_pass: list[str] = []
        for val in unmatched_values:
            val_tokens = set(tokenize(val))
            if val_tokens and not val_tokens.issubset(matched_tokens):
                first_pass.append(val)

        # Second pass: remove composites redundant with shorter values
        cleaned_unmatched: list[str] = []
        for val in first_pass:
            val_tokens = set(tokenize(val))
            novel_tokens = val_tokens - matched_tokens
            other_unmatched_tokens = set()
            for other in first_pass:
                if other != val and len(other) < len(val):
                    other_unmatched_tokens.update(tokenize(other))
            if novel_tokens and novel_tokens.issubset(other_unmatched_tokens):
                continue
            cleaned_unmatched.append(val)

        return SideGroupMatch(
            matched_item_ids=matched_item_ids,
            matched_names=matched_names,
            unmatched_values=dedupe_keep_order(cleaned_unmatched),
            match_debug=build_match_debug_payload(
                raw_utterance=normalized_user_text,
                candidates=candidates,
                selected_candidate=debug_candidate_text,
                matched_option=debug_matched_option,
                match_source=debug_match_source,
                match_score=debug_match_score,
            ),
        )

    def _augment_candidates_with_group_phrases(
        self,
        *,
        candidates: list[OptionCandidate],
        group,
        normalized_user_text: str,
    ) -> list[OptionCandidate]:
        slot_first = [candidate for candidate in candidates if candidate.source != "raw_utterance"]
        raw_fallback = [candidate for candidate in candidates if candidate.source == "raw_utterance"]
        scoped_candidates = build_scoped_phrase_candidates(
            raw_utterance=normalized_user_text,
            phrases=self._group_choice_phrases(group),
            cleaner=_normalize_side_candidate,
            source="raw_utterance",
        )
        return dedupe_option_candidates(slot_first + scoped_candidates + raw_fallback)

    def _greedy_scan_for_embedded_sides(
        self,
        *,
        group,
        text: str,
        already_selected_ids: list[str],
    ) -> tuple[list[str], list[str]]:
        matched_item_ids: list[str] = []
        matched_names: list[str] = []

        for choice in group.choices:
            if choice.item_id in already_selected_ids or choice.item_id in matched_item_ids:
                continue

            labels = sorted(
                set(getattr(choice, "match_texts", ()) or (choice.normalized_name,)),
                key=len,
                reverse=True,
            )
            for label in labels:
                if not label:
                    continue
                if not self._has_supported_phrase_context(text, label):
                    continue
                matched_item_ids.append(choice.item_id)
                matched_names.append(choice.name)
                break

        return matched_item_ids, matched_names

    def _build_candidate_values(
        self,
        *,
        normalized_user_text: str,
        normalized_slot_values: list[str],
    ) -> list[OptionCandidate]:
        slot_candidates = [
            OptionCandidate(text=_normalize_side_candidate(value), source="slot_value")
            for value in normalized_slot_values
            if _normalize_side_candidate(value)
        ]
        raw_candidates = build_candidate_texts_normalized(
            normalized_user_text=_normalize_side_candidate(normalized_user_text),
            normalized_slot_values=(),
            allow_split=True,
        )
        return dedupe_option_candidates(
            slot_candidates
            + [
                OptionCandidate(text=value, source="raw_utterance")
                for value in raw_candidates
                if value
            ]
        )

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
            labels = getattr(choice, "match_texts", ()) or (choice.normalized_name,)
            confidence = max(
                (self._choice_confidence(candidate, label) for label in labels if label),
                default=0.0,
            )
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
        return score_scoped_choice(
            candidate,
            choice_name,
            reject_candidate_superset=True,
        )

    @staticmethod
    def _has_supported_phrase_context(text: str, phrase: str) -> bool:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b")
        for match in pattern.finditer(text):
            prefix = text[:match.start()].rstrip()
            suffix = text[match.end():].lstrip()

            previous_word = re.search(r"([a-z0-9]+)$", prefix)
            if previous_word and previous_word.group(1) in NEGATION_WORDS:
                continue

            next_word = re.match(r"([a-z0-9]+)", suffix)
            if next_word and next_word.group(1) in NEGATION_WORDS:
                continue

            # Multi-token phrases like "american cheese" or "plain bun" are
            # already scoped to the active group, so allow them to appear
            # inline within a larger multi-slot utterance.
            if " " in phrase:
                return True

            if previous_word and previous_word.group(1) not in SIZE_CONTEXT_WORDS | {"with", "and", "plus", "also"}:
                # Allow direct adjacency to punctuation / string start, but reject
                # arbitrary leading words to keep side matching scoped.
                continue

            if next_word and next_word.group(1) not in SIZE_CONTEXT_WORDS | {"with", "and", "plus", "also"}:
                continue

            return True

        return False

    @staticmethod
    def _group_choice_phrases(group) -> list[str]:
        phrases: list[str] = []
        seen: set[str] = set()
        for choice in group.choices:
            for label in getattr(choice, "match_texts", ()) or (choice.normalized_name,):
                normalized = normalize_text(label or "")
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                phrases.append(normalized)
        return phrases


def dedupe_option_candidates(values: list[OptionCandidate]) -> list[OptionCandidate]:
    seen: set[str] = set()
    result: list[OptionCandidate] = []
    for value in values:
        if not value.text or value.text in seen:
            continue
        seen.add(value.text)
        result.append(value)
    return result
