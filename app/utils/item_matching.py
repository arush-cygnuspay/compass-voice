# app/utils/item_matching.py
from __future__ import annotations


def _tokens_from_normalized(text: str) -> tuple[str, ...]:
    return tuple(part for part in text.split() if part)


def _ngrams(tokens: tuple[str, ...], n: int) -> set[tuple[str, ...]]:
    token_count = len(tokens)
    if n <= 0 or token_count < n:
        return set()

    return {
        tokens[i : i + n]
        for i in range(token_count - n + 1)
    }


def score_item_normalized(user_text: str, item_name: str) -> float:
    """
    Deterministic similarity score between already-normalized user text
    and already-normalized item text.

    Contract:
    - inputs must already be normalized upstream
    - function does no lowercase/strip/punctuation cleanup
    """
    if not user_text or not item_name:
        return 0.0

    if user_text == item_name:
        return 10.0

    user_tokens = _tokens_from_normalized(user_text)
    name_tokens = _tokens_from_normalized(item_name)

    if not user_tokens or not name_tokens:
        return 0.0

    score = 0.0

    max_n = min(len(user_tokens), len(name_tokens))
    for n in range(max_n, 0, -1):
        if _ngrams(user_tokens, n) & _ngrams(name_tokens, n):
            score = max(score, 6.0 + n / max_n)
            break

    user_token_set = set(user_tokens)
    name_token_set = set(name_tokens)
    overlap = len(user_token_set & name_token_set)

    if overlap > 0:
        coverage = overlap / len(name_tokens)
        score = max(score, 4.0 * coverage)

    score = max(score, 1.0 * overlap)
    return score


def score_item(user_text: str, item_name: str) -> float:
    """
    Backward-compatible wrapper.

    Prefer score_item_normalized(...) in hot-path code.
    """
    user_norm = (user_text or "").lower().strip()
    item_norm = (item_name or "").lower().strip()
    return score_item_normalized(user_norm, item_norm)