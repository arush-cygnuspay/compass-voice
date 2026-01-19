# app/utils/choice_matching.py

from typing import Iterable, Optional


def _normalize(text: str) -> str:
    return text.lower().strip()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[i : i + n])
        for i in range(len(tokens) - n + 1)
    }


def match_choice(
    user_text: str,
    choices: Iterable,
) -> Optional[object]:
    """
    Deterministic fuzzy matcher for menu choices.

    Scoring signals (highest → lowest priority):
    1. Exact normalized match
    2. Longest n-gram containment (ordered substring)
    3. Token coverage ratio
    4. Token overlap count (fallback)
    """

    user_norm = _normalize(user_text)
    user_tokens = _tokens(user_text)

    if not user_tokens:
        return None

    best_choice = None
    best_score = 0.0

    for choice in choices:
        name = choice.name
        name_norm = _normalize(name)
        name_tokens = _tokens(name)

        # --------------------------------------------------
        # 1️⃣ Exact match
        # --------------------------------------------------
        if user_norm == name_norm:
            return choice

        score = 0.0

        # --------------------------------------------------
        # 2️⃣ Ordered n-gram containment
        # --------------------------------------------------
        max_n = min(len(user_tokens), len(name_tokens))
        for n in range(max_n, 0, -1):
            user_ngrams = _ngrams(user_tokens, n)
            name_ngrams = _ngrams(name_tokens, n)

            if user_ngrams & name_ngrams:
                score = 3.0 + n / max_n  # strong signal
                break

        # --------------------------------------------------
        # 3️⃣ Token coverage ratio
        # --------------------------------------------------
        overlap = len(set(user_tokens) & set(name_tokens))
        if overlap > 0:
            coverage = overlap / len(name_tokens)
            score = max(score, 2.0 * coverage)

        # --------------------------------------------------
        # 4️⃣ Fallback: raw overlap
        # --------------------------------------------------
        score = max(score, 1.0 * overlap)

        # --------------------------------------------------
        # Pick best
        # --------------------------------------------------
        if score > best_score:
            best_score = score
            best_choice = choice

    return best_choice if best_score > 0 else None
