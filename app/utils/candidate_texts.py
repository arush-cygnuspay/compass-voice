# app/utils/candidate_texts.py
from __future__ import annotations

import re
from typing import Iterable

from app.nlu.query_normalization.text_preprocessor import normalize_text


# Conservative separators only.
# Do not split on semantic markers like "with", "no", "without".
_CANDIDATE_SPLIT_PAT = re.compile(r"\s*(?:,| and | & |\+)\s*", re.IGNORECASE)


def _split_candidates(text: str) -> list[str]:
    """
    Conservative lexical splitting for multi-value utterances.

    Examples:
    - "cheese and jelly" -> ["cheese", "jelly"]
    - "coke, sprite" -> ["coke", "sprite"]
    """
    if not text:
        return []

    parts = _CANDIDATE_SPLIT_PAT.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def build_candidate_texts(
    *,
    user_text: str,
    slot_values: Iterable[str] | None = None,
    allow_split: bool = True,
) -> list[str]:
    """
    Backward-compatible wrapper for callers still passing raw text.

    Prefer build_candidate_texts_normalized(...) in hot-path code.
    """
    normalized_slot_values = [
        normalized
        for value in (slot_values or ())
        if (normalized := normalize_text(value))
    ]

    return build_candidate_texts_normalized(
        normalized_user_text=normalize_text(user_text or ""),
        normalized_slot_values=normalized_slot_values,
        allow_split=allow_split,
    )


def build_candidate_texts_normalized(
    *,
    normalized_user_text: str,
    normalized_slot_values: Iterable[str] | None = None,
    allow_split: bool = True,
) -> list[str]:
    """
    Build ordered candidate texts for matching using already-normalized inputs.

    Priority:
    1) normalized slot values
    2) full normalized utterance
    3) conservative split chunks (optional)

    De-duplicates while preserving order.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def add(text: str | None) -> None:
        if not text:
            return

        value = text.strip()
        if not value or value in seen:
            return

        seen.add(value)
        candidates.append(value)

    for value in normalized_slot_values or ():
        add(value)

    add(normalized_user_text)

    if allow_split and normalized_user_text:
        for chunk in _split_candidates(normalized_user_text):
            add(chunk)

    return candidates