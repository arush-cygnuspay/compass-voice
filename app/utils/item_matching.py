# app/utils/item_matching.py

from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    return text.lower().strip()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[i : i + n])
        for i in range(len(tokens) - n + 1)
    }


def score_item(user_text: str, item_name: str) -> float:
    """
    Deterministic similarity score between user text and item name.
    """

    user_norm = _normalize(user_text)
    name_norm = _normalize(item_name)

    user_tokens = _tokens(user_text)
    name_tokens = _tokens(item_name)

    if not user_tokens or not name_tokens:
        return 0.0

    # 1️⃣ Exact match
    if user_norm == name_norm:
        return 10.0

    score = 0.0

    # 2️⃣ Ordered n-gram match (strongest fuzzy signal)
    max_n = min(len(user_tokens), len(name_tokens))
    for n in range(max_n, 0, -1):
        if _ngrams(user_tokens, n) & _ngrams(name_tokens, n):
            score = max(score, 6.0 + n / max_n)
            break

    # 3️⃣ Token coverage ratio
    overlap = len(set(user_tokens) & set(name_tokens))
    if overlap > 0:
        coverage = overlap / len(name_tokens)
        score = max(score, 4.0 * coverage)

    # 4️⃣ Raw overlap fallback
    score = max(score, 1.0 * overlap)

    return score
