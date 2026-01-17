# app/utils/choice_matching.py

from typing import Iterable, Optional


def _normalize(text: str) -> str:
    return text.lower().strip()


def _tokens(text: str) -> set[str]:
    return set(_normalize(text).split())


def match_choice(
    user_text: str,
    choices: Iterable,
) -> Optional[object]:
    """
    Generic fuzzy matcher for menu choices.

    Matching priority:
    1. Phrase containment ("garlic" → "garlic mayo")
    2. Full token match
    3. Partial token overlap (fallback)
    """

    user_norm = _normalize(user_text)
    user_tokens = _tokens(user_text)

    best_match = None
    best_score = 0

    for choice in choices:
        name = choice.name
        name_norm = _normalize(name)
        name_tokens = _tokens(name)

        # 1️⃣ Phrase containment (BEST)
        if name_norm in user_norm or user_norm in name_norm:
            return choice

        # 2️⃣ All tokens present
        if name_tokens.issubset(user_tokens):
            return choice

        # 3️⃣ Partial overlap
        overlap = len(user_tokens & name_tokens)
        if overlap > best_score:
            best_score = overlap
            best_match = choice

    return best_match if best_score > 0 else None
