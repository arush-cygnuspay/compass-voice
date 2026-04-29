# app/nlu/matching/matcher.py
"""Item/token matching orchestration.

Combines deterministic choice resolution (exact → token fallback) with
token-overlap matching predicates.  All functions operate on already-
normalized text; callers are responsible for normalization upstream.
"""
from __future__ import annotations

from app.nlu.matching.normalization import tokenize

__all__ = [
    # deterministic resolution
    "exact_match",
    "token_match",
    "resolve_choice",
    "_looks_like_skip_choice_answer",
    # token matching predicates
    "token_overlap_score",
    "is_strong_token_match",
    "is_controlled_partial_match",
]


# ── Deterministic choice resolution ──────────────────────────────────────────

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
    """Strict deterministic resolution: exact match then token match."""
    result = exact_match(normalized_text, lookup)
    if result:
        return result

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

    return False


# ── Token-overlap matching predicates ────────────────────────────────────────

def token_overlap_score(a: str, b: str) -> int:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    return len(a_tokens & b_tokens)


def is_strong_token_match(a: str, b: str, min_overlap: int = 1) -> bool:
    """Directional strong match.

    Candidate/user text tokens must be a subset of choice tokens; single-token
    candidates only match single-token choices.
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
    """Directional partial fallback.

    Only for multi-token, meaningful phrases; candidate must be contained in
    choice, never reverse.
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
