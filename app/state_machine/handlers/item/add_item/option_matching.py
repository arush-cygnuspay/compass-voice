from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Callable, Iterable, Sequence

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.utils.candidate_texts import build_candidate_texts_normalized
from app.utils.token_matcher import (
    is_controlled_partial_match,
    is_strong_token_match,
    tokenize,
)

_COMMON_SLOT_FALLBACK_LABELS: frozenset[str] = frozenset({"ITEM", "MENU_ITEM"})
_GROUP_NOUN_TOKENS: frozenset[str] = frozenset({"bun", "cheese", "meat"})
_SCOPED_PHRASE_BRIDGE_TOKENS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "extra",
        "large",
        "less",
        "medium",
        "mini",
        "more",
        "no",
        "plus",
        "regular",
        "remove",
        "small",
        "without",
        "with",
        "xl",
    }
)
_COMMON_LEADING_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "ok",
        "oka",
        "okay",
        "alright",
        "please",
        "just",
        "uh",
        "um",
        "well",
        "so",
    }
)
_COMMON_LEADING_FILLER_PHRASES: tuple[str, ...] = ("all right ",)


@dataclass(frozen=True, slots=True)
class OptionCandidate:
    text: str
    source: str
    slot_label: str | None = None


def strip_common_option_fillers(text: str) -> str:
    value = normalize_text(text or "")
    if not value:
        return ""

    changed = True
    while changed and value:
        changed = False

        for phrase in _COMMON_LEADING_FILLER_PHRASES:
            if value.startswith(phrase):
                value = value[len(phrase):].strip()
                changed = True
                break

        if changed:
            continue

        tokens = value.split()
        if tokens and tokens[0] in _COMMON_LEADING_FILLER_WORDS:
            value = " ".join(tokens[1:]).strip()
            changed = True

    return value


def build_slot_first_option_candidates(
    *,
    raw_utterance: str,
    slots: Sequence[Any] | None,
    allowed_slot_labels: Iterable[str],
    cleaner: Callable[[str], str] | None = None,
    allow_split: bool = True,
) -> list[OptionCandidate]:
    normalized_cleaner = cleaner or normalize_text
    relevant_labels = {str(label).upper() for label in allowed_slot_labels}
    relevant_labels |= _COMMON_SLOT_FALLBACK_LABELS

    candidates: list[OptionCandidate] = []
    seen: set[str] = set()

    def add_text(text: str | None, *, source: str, slot_label: str | None = None) -> None:
        normalized = normalized_cleaner(text or "")
        if not normalized:
            return

        expanded = [normalized]
        if allow_split:
            expanded = build_candidate_texts_normalized(
                normalized_user_text=normalized,
                normalized_slot_values=(),
                allow_split=True,
            )

        for value in expanded:
            cleaned = normalized_cleaner(value)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            candidates.append(
                OptionCandidate(
                    text=cleaned,
                    source=source,
                    slot_label=slot_label,
                )
            )

    for slot in slots or ():
        slot_label = str(getattr(slot, "name", "") or "").upper()
        if slot_label not in relevant_labels:
            continue

        slot_value = getattr(slot, "value", None)
        if isinstance(slot_value, str):
            add_text(slot_value, source="slot_value", slot_label=slot_label)

        slot_raw = getattr(slot, "raw", None)
        if isinstance(slot_raw, str):
            add_text(slot_raw, source="slot_raw", slot_label=slot_label)

    add_text(raw_utterance, source="raw_utterance")
    return candidates


def extract_slot_candidate_texts(
    *,
    slots: Sequence[Any] | None,
    allowed_slot_labels: Iterable[str],
    cleaner: Callable[[str], str] | None = None,
) -> list[str]:
    return [
        candidate.text
        for candidate in build_slot_first_option_candidates(
            raw_utterance="",
            slots=slots,
            allowed_slot_labels=allowed_slot_labels,
            cleaner=cleaner,
            allow_split=True,
        )
        if candidate.source in {"slot_value", "slot_raw"}
    ]


def build_scoped_phrase_candidates(
    *,
    raw_utterance: str,
    phrases: Iterable[str],
    cleaner: Callable[[str], str] | None = None,
    source: str = "raw_utterance",
) -> list[OptionCandidate]:
    normalized_text = normalize_text(raw_utterance or "")
    normalized_cleaner = cleaner or normalize_text
    if not normalized_text:
        return []

    phrase_map: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    seen_phrases: set[tuple[str, ...]] = set()
    for phrase in phrases:
        normalized_phrase = normalize_text(phrase or "")
        cleaned_phrase = normalized_cleaner(normalized_phrase)
        phrase_tokens = tuple(token for token in normalized_phrase.split() if token)
        if not cleaned_phrase or not phrase_tokens or phrase_tokens in seen_phrases:
            continue
        seen_phrases.add(phrase_tokens)
        phrase_map.setdefault(phrase_tokens[0], []).append((phrase_tokens, cleaned_phrase))

    if not phrase_map:
        return []

    for bucket in phrase_map.values():
        bucket.sort(key=lambda item: len(item[0]), reverse=True)

    tokens = [token for token in normalized_text.split() if token]
    candidates: list[OptionCandidate] = []
    seen: set[str] = set()
    allowed_starts: set[int] = {0}
    idx = 0
    while idx < len(tokens):
        matched = False
        if idx in allowed_starts:
            for phrase_tokens, cleaned_phrase in phrase_map.get(tokens[idx], []):
                end = idx + len(phrase_tokens)
                if tuple(tokens[idx:end]) != phrase_tokens:
                    continue
                if cleaned_phrase not in seen:
                    seen.add(cleaned_phrase)
                    candidates.append(OptionCandidate(text=cleaned_phrase, source=source))
                allowed_starts.add(end)
                idx = end
                matched = True
                break

        if matched:
            continue

        if tokens[idx] in _SCOPED_PHRASE_BRIDGE_TOKENS:
            allowed_starts.add(idx + 1)
        idx += 1

    return candidates


def score_scoped_choice(
    candidate: str,
    choice_name: str,
    *,
    reject_candidate_superset: bool = True,
) -> float:
    if not candidate or not choice_name:
        return 0.0

    if candidate == choice_name:
        return 1.0

    candidate_token_list = tokenize(candidate)
    choice_token_list = tokenize(choice_name)
    candidate_tokens = set(candidate_token_list)
    choice_tokens = set(choice_token_list)

    if reject_candidate_superset and choice_tokens and choice_tokens < candidate_tokens:
        return 0.0

    best = 0.0
    if candidate_tokens and choice_tokens:
        overlap = len(candidate_tokens & choice_tokens)
        coverage = overlap / len(choice_tokens)
        token_score = coverage
        if len(candidate_tokens) > 1 or len(choice_tokens) == 1:
            candidate_coverage = overlap / len(candidate_tokens)
            token_score = max(token_score, candidate_coverage)
        best = max(best, token_score)

    if is_strong_token_match(candidate, choice_name):
        best = max(best, 0.92)

    if is_controlled_partial_match(candidate, choice_name):
        best = max(best, 0.82)

    fuzzy = SequenceMatcher(None, candidate, choice_name).ratio()
    best = max(best, fuzzy)

    if (
        len(candidate_token_list) == len(choice_token_list) >= 2
        and candidate_token_list[0] == choice_token_list[0]
        and choice_token_list[-1] in _GROUP_NOUN_TOKENS
        and len(candidate_tokens & choice_tokens) >= len(choice_tokens) - 1
        and fuzzy >= 0.72
    ):
        best = max(best, 0.81)

    return best


def build_match_debug_payload(
    *,
    raw_utterance: str,
    candidates: Sequence[OptionCandidate],
    selected_candidate: str | None,
    matched_option: str | None,
    match_source: str | None,
    match_score: float | None,
) -> dict[str, object]:
    slot_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.source not in {"slot_value", "slot_raw"}:
            continue
        if candidate.text in seen:
            continue
        seen.add(candidate.text)
        slot_candidates.append(candidate.text)

    return {
        "raw_utterance": raw_utterance,
        "slot_candidates": slot_candidates,
        "selected_candidate": selected_candidate,
        "matched_option": matched_option,
        "match_source": match_source,
        "match_score": match_score,
    }
