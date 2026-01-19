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
    Deterministic, rank-stable fuzzy matcher for menu choices.

    Resolution order (strict):
    1. Exact normalized match
    2. Longest ordered n-gram containment (EARLY RETURN)
    3. Token coverage ratio
    4. Token overlap count (fallback)

    This design guarantees:
    - No ambiguity override
    - No order-dependent behavior
    - Stable results across runs
    """

    user_norm = _normalize(user_text)
    user_tokens = _tokens(user_text)

    if not user_tokens:
        return None

    # ---------------------------------------
    # 1️⃣ Exact normalized match (hard stop)
    # ---------------------------------------
    for choice in choices:
        if user_norm == _normalize(choice.name):
            return choice

    # ---------------------------------------
    # 2️⃣ Longest ordered n-gram containment
    #     (hard stop, dominance rule)
    # ---------------------------------------
    best_ngram_len = 0
    best_ngram_choice = None

    for choice in choices:
        name_tokens = _tokens(choice.name)
        max_n = min(len(user_tokens), len(name_tokens))

        for n in range(max_n, 1, -1):  # n >= 2 only
            if _ngrams(user_tokens, n) & _ngrams(name_tokens, n):
                if n > best_ngram_len:
                    best_ngram_len = n
                    best_ngram_choice = choice
                break  # stop descending n for this choice

    if best_ngram_choice:
        return best_ngram_choice

    # ---------------------------------------
    # 3️⃣ Token coverage ratio (ranked)
    # ---------------------------------------
    best_choice = None
    best_score = 0.0

    for choice in choices:
        name_tokens = _tokens(choice.name)
        overlap = len(set(user_tokens) & set(name_tokens))

        if overlap == 0:
            continue

        coverage = overlap / len(name_tokens)
        score = 2.0 * coverage  # coverage-weighted

        if score > best_score:
            best_score = score
            best_choice = choice

    if best_choice:
        return best_choice

    # ---------------------------------------
    # 4️⃣ Raw overlap fallback (last resort)
    # ---------------------------------------
    best_overlap = 0
    best_choice = None

    for choice in choices:
        overlap = len(set(user_tokens) & set(_tokens(choice.name)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_choice = choice

    return best_choice if best_overlap > 0 else None

