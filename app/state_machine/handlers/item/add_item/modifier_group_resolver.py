from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.pending_item_models import ModifierSelection
from app.utils.candidate_texts import build_candidate_texts_normalized
from app.utils.token_matcher import (
    is_controlled_partial_match,
    is_strong_token_match,
    tokenize,
)

AUTO_ACCEPT_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.60
MIN_CONFIRM_GAP = 0.08

REMOVE_PREFIXES = ("no ", "without ")
EXTRA_WORDS = {"extra", "more", "double"}
LESS_WORDS = {"less", "light"}
ON_SIDE_SUFFIXES = ("on the side", "on side")
GREEDY_CONTEXT_WORDS = {
    "with",
    "and",
    "plus",
    "also",
    "extra",
    "more",
    "double",
    "no",
    "without",
    "light",
    "less",
    "on",
    "the",
    "side",
}

# IMPORTANT: generic words that should NEVER become modifiers
GENERIC_MODIFIER_WORDS = {
    "toppings",
    "stuff",
    "things",
}


@dataclass(frozen=True, slots=True)
class ModifierGroupMatch:
    selections: list[ModifierSelection]
    unmatched_values: list[str]


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        value = (value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_modifier_slot_values_normalized(context) -> list[str]:
    slots = context.last_slots or ()
    values = []
    seen = set()

    for slot in slots:
        if str(slot.name).upper() not in {"MODIFIER", "ITEM", "MENU_ITEM"}:
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


class ModifierGroupResolver:
    def resolve(
        self,
        *,
        group,
        normalized_user_text: str,
        normalized_slot_values: list[str],
        already_selected_ids: list[str] | None = None,
        ignored_values: list[str] | None = None,
    ) -> ModifierGroupMatch:
        already_selected_ids = already_selected_ids or []
        ignored_values = dedupe_keep_order(list(ignored_values or []))
        ignored_token_set: set[str] = set()
        for value in ignored_values:
            ignored_token_set.update(tokenize(value))

        candidates = self._build_candidates(
            normalized_user_text,
            normalized_slot_values,
            ignored_values=ignored_values,
        )

        selections_by_id: dict[str, ModifierSelection] = {}
        unmatched: list[str] = []
        matched_candidate_texts: list[str] = []

        for candidate in candidates:
            parsed = self._parse(candidate)

            if not parsed["target"]:
                continue

            if parsed["target"] in GENERIC_MODIFIER_WORDS:
                continue

            scored = self._match(group, parsed["target"])
            if not scored:
                unmatched.append(candidate)
                continue

            mod_id, name, confidence = scored
            if confidence < AUTO_ACCEPT_THRESHOLD:
                unmatched.append(candidate)
                continue

            if mod_id in already_selected_ids:
                continue

            new_selection = ModifierSelection(
                modifier_id=mod_id,
                name=name,
                action=parsed["action"],
                instruction=parsed["instruction"],
            )

            previous = selections_by_id.get(mod_id)
            if previous is None or self._priority(new_selection) > self._priority(previous):
                selections_by_id[mod_id] = new_selection
                matched_candidate_texts.append(parsed["target"])

        self._greedy_scan_for_embedded_modifiers(
            group=group,
            text=normalized_user_text,
            selections_by_id=selections_by_id,
            already_selected_ids=already_selected_ids,
            ignored_tokens=ignored_token_set,
            matched_candidate_texts=matched_candidate_texts,
        )

        ordered_ids: list[str] = []
        for candidate in candidates:
            parsed = self._parse(candidate)
            scored = self._match(group, parsed["target"])
            if not scored:
                continue
            modifier_id = scored[0]
            if modifier_id in selections_by_id and modifier_id not in ordered_ids:
                ordered_ids.append(modifier_id)

        for modifier_id in selections_by_id:
            if modifier_id not in ordered_ids:
                ordered_ids.append(modifier_id)

        matched_tokens: set[str] = set()
        for selection in selections_by_id.values():
            matched_tokens.update(tokenize(normalize_text(selection.name)))
        for candidate_text in matched_candidate_texts:
            matched_tokens.update(tokenize(candidate_text))

        first_pass: list[str] = []
        for value in unmatched:
            value_tokens = set(tokenize(value))
            if value_tokens and not value_tokens.issubset(matched_tokens | ignored_token_set):
                first_pass.append(value)

        cleaned_unmatched: list[str] = []
        for value in first_pass:
            value_tokens = set(tokenize(value))
            novel_tokens = value_tokens - matched_tokens - ignored_token_set
            other_unmatched_tokens: set[str] = set()
            for other in first_pass:
                if other != value and len(other) < len(value):
                    other_unmatched_tokens.update(set(tokenize(other)) - ignored_token_set)
            if novel_tokens and novel_tokens.issubset(other_unmatched_tokens):
                continue
            cleaned_unmatched.append(value)

        return ModifierGroupMatch(
            selections=[selections_by_id[modifier_id] for modifier_id in ordered_ids],
            unmatched_values=dedupe_keep_order(cleaned_unmatched),
        )

    def _greedy_scan_for_embedded_modifiers(
        self,
        *,
        group,
        text: str,
        selections_by_id: dict[str, ModifierSelection],
        already_selected_ids: list[str],
        ignored_tokens: set[str] | None = None,
        matched_candidate_texts: list[str] | None = None,
    ) -> None:
        if not text:
            return

        ignored_tokens = ignored_tokens or set()
        matched_candidate_texts = matched_candidate_texts if matched_candidate_texts is not None else []
        text_tokens = set(tokenize(text))
        if not text_tokens:
            return

        for choice in group.choices:
            modifier_id = choice.modifier_id
            if modifier_id in selections_by_id or modifier_id in already_selected_ids:
                continue

            choice_normalized = choice.normalized_name
            if not choice_normalized:
                continue

            choice_tokens = set(tokenize(choice_normalized))
            if not choice_tokens:
                continue

            if ignored_tokens and choice_tokens.issubset(ignored_tokens):
                continue

            if not choice_tokens.issubset(text_tokens):
                continue

            if choice_normalized not in text:
                continue

            if not self._has_supported_phrase_context(text, choice_normalized):
                continue

            selections_by_id[modifier_id] = ModifierSelection(
                modifier_id=modifier_id,
                name=choice.name,
                action="add",
                instruction=None,
            )
            matched_candidate_texts.append(choice_normalized)

    def _priority(self, selection: ModifierSelection) -> int:
        if selection.action == "remove":
            return 4
        if selection.instruction == "extra":
            return 3
        if selection.instruction == "less":
            return 2
        if selection.instruction == "on_side":
            return 2
        return 1

    def _build_candidates(
        self,
        text: str,
        slot_values: list[str],
        *,
        ignored_values: list[str] | None = None,
    ) -> list[str]:
        candidates: list[str] = []
        ignored_values = ignored_values or []
        cleaned_text = self._strip_ignored_values(text, ignored_values)

        if cleaned_text:
            candidates.append(cleaned_text.strip())

        splits = build_candidate_texts_normalized(
            normalized_user_text=cleaned_text,
            normalized_slot_values=[],
            allow_split=True,
        )
        candidates.extend(splits)

        candidates.extend(
            value
            for value in slot_values
            if value and value not in ignored_values
        )

        return dedupe_keep_order(candidates)

    @staticmethod
    def _strip_ignored_values(text: str, ignored_values: list[str]) -> str:
        cleaned = (text or "").strip()
        if not cleaned or not ignored_values:
            return cleaned

        for value in sorted((v for v in ignored_values if v), key=len, reverse=True):
            cleaned = re.sub(rf"\b{re.escape(value)}\b", " ", cleaned)

        return re.sub(r"\s+", " ", cleaned).strip()

    def _parse(self, text: str):
        text = (text or "").strip()

        for prefix in REMOVE_PREFIXES:
            if text.startswith(prefix):
                return {
                    "action": "remove",
                    "instruction": None,
                    "target": text[len(prefix):].strip(),
                }

        tokens = text.split()
        if not tokens:
            return {"action": "add", "instruction": None, "target": ""}

        first = tokens[0]
        rest = " ".join(tokens[1:]).strip()

        if first in EXTRA_WORDS and rest:
            return {"action": "add", "instruction": "extra", "target": rest}

        if first in LESS_WORDS and rest:
            return {"action": "add", "instruction": "less", "target": rest}

        for suffix in ON_SIDE_SUFFIXES:
            if text.endswith(f" {suffix}"):
                return {
                    "action": "add",
                    "instruction": "on_side",
                    "target": text[: -(len(suffix) + 1)].strip(),
                }

        return {"action": "add", "instruction": None, "target": text}

    def _match(self, group, candidate):
        best = None
        best_confidence = 0.0
        second_confidence = 0.0

        for choice in group.choices:
            confidence = self._confidence(candidate, choice.normalized_name)
            if confidence > best_confidence:
                second_confidence = best_confidence
                best_confidence = confidence
                best = choice
            elif confidence > second_confidence:
                second_confidence = confidence

        if not best:
            return None

        if best_confidence < CONFIRM_THRESHOLD:
            return None

        if (
            best_confidence < AUTO_ACCEPT_THRESHOLD
            and (best_confidence - second_confidence) < MIN_CONFIRM_GAP
        ):
            return None

        return best.modifier_id, best.name, best_confidence

    def _confidence(self, candidate: str, choice_name: str) -> float:
        if candidate == choice_name:
            return 1.0

        candidate_tokens = set(tokenize(candidate))
        choice_tokens = set(tokenize(choice_name))
        if choice_tokens and choice_tokens < candidate_tokens:
            return 0.0

        score = SequenceMatcher(None, candidate, choice_name).ratio()

        if is_strong_token_match(candidate, choice_name):
            score = max(score, 0.92)

        if is_controlled_partial_match(candidate, choice_name):
            score = max(score, 0.82)

        return score

    @staticmethod
    def _has_supported_phrase_context(text: str, phrase: str) -> bool:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b")
        for match in pattern.finditer(text):
            prefix = text[:match.start()].rstrip()
            suffix = text[match.end():].lstrip()

            previous_word = re.search(r"([a-z0-9]+)$", prefix)
            if previous_word and previous_word.group(1) not in GREEDY_CONTEXT_WORDS:
                continue

            next_word = re.match(r"([a-z0-9]+)", suffix)
            if next_word and next_word.group(1) not in GREEDY_CONTEXT_WORDS:
                continue

            return True

        return False
