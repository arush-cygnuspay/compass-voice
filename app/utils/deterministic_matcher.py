# app/state_machine/utils/deterministic_matcher.py

from __future__ import annotations

from typing import Optional, Sequence


def exact_match(
    normalized_text: str,
    lookup: dict[str, list],
):
    return lookup.get(normalized_text)


def token_match(
    normalized_text: str,
    lookup: dict[str, list],
):
    tokens = normalized_text.split()

    for token in tokens:
        if token in lookup:
            return lookup[token]

    return None


def resolve_choice(
    normalized_text: str,
    lookup: dict[str, list],
):
    """
    Strict deterministic resolution:
    1. exact match
    2. token match (controlled)
    """

    # 1. exact
    result = exact_match(normalized_text, lookup)
    if result:
        return result

    # 2. token-level match
    result = token_match(normalized_text, lookup)
    if result:
        return result

    return None


def _looks_like_skip_choice_answer(normalized_text: str) -> bool:
    text = (normalized_text or "").strip()

    skip_phrases = {
        "no",
        "none",
        "no thanks",
        "skip",
        "skip it",
        "without it",
        "nothing",
    }

    if text in skip_phrases:
        return True

    if text.startswith("no "):
        return True

    if text.startswith("without "):
        return True