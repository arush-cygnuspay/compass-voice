# app/nlu/matching/normalization.py
"""Single normalization/tokenization source for all matching modules.

``normalize_text`` is the general-purpose text normalizer used across the
whole NLU stack.  ``tokenize`` is the matching-specific tokenizer: it strips
stop-words, collapses plurals, and discards noise tokens so that token-overlap
comparisons are as precise as possible.
"""
from __future__ import annotations

import re

from app.nlu.query_normalization.text_preprocessor import normalize_text

__all__ = [
    "normalize_text",
    "tokenize",
    "_canonicalize_token",
    "_WEAK_TOKENS",
]

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
