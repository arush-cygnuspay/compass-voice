# app/state_machine/utils/token_matcher.py
from __future__ import annotations

import re

_WEAK_TOKENS = {
    "a",
    "an",
    "and",
    "or",
    "the",
    "of",
    "for",
    "to",
    "with",
    "please",
    "i",
    "will",
    "take",
    "want",
    "like",
    "would",
    "id",
    "ill",
    "oz",
    "ounce",
    "ounces",
    "sauce",
}


def _canonicalize_token(token: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", (token or "").lower())
    if not token:
        return ""

    if token.isdigit():
        return ""

    if len(token) == 1:
        return ""

    # Lightweight singularization to help ASR variants like "tomatoes" -> "tomato"
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("oes"):
        token = token[:-2]  # tomatoes -> tomato
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]

    if token in _WEAK_TOKENS:
        return ""

    return token


def tokenize(text: str) -> list[str]:
    raw_tokens = re.split(r"\s+", (text or "").strip().lower())
    tokens: list[str] = []

    for raw in raw_tokens:
        token = _canonicalize_token(raw)
        if token:
            tokens.append(token)

    return tokens


def token_overlap_score(a: str, b: str) -> int:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    return len(a_tokens & b_tokens)


def is_strong_token_match(a: str, b: str, min_overlap: int = 1) -> bool:
    """
    Directional strong match:
    - candidate/user text tokens must be a subset of choice tokens
    - single-token candidates only match single-token choices
    """
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))

    if not a_tokens or not b_tokens:
        return False

    overlap = len(a_tokens & b_tokens)
    if overlap < min_overlap:
        return False

    if a_tokens == b_tokens:
        return True

    if len(a_tokens) == 1:
        return len(b_tokens) == 1 and overlap == 1

    return a_tokens.issubset(b_tokens)


def is_controlled_partial_match(a: str, b: str) -> bool:
    """
    Directional partial fallback:
    - only after exact/token checks
    - only for multi-token, meaningful phrases
    - candidate must be contained in choice, never reverse
    """
    a_clean = " ".join(tokenize(a))
    b_clean = " ".join(tokenize(b))

    if not a_clean or not b_clean:
        return False

    if len(a_clean) < 4:
        return False

    if len(a_clean.split()) < 2:
        return False

    return a_clean in b_clean