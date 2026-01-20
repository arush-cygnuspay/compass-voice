# app/nlu/intent_resolution/item_intent_resolver.py

"""
PURE ITEM LINGUISTIC MATCHER

Rules:
- No menu lookup
- No session
- No state
- Regex + lexical heuristics only
"""

from app.nlu.intent_patterns.add_item import (
    ADD_STRONG_VERB_PAT,
    ADD_DESIRE_PAT,
    ADD_CONFIRMATION_PAT,
    QUANTITY_ITEM_PAT,
    BARE_ITEM_PAT,
)


def match_add_item(text: str, *, allow_bare: bool = True) -> bool:
    """
    Returns True if text linguistically resembles an add-item request.

    allow_bare:
        - True  → allow ultra-loose bare item matching
        - False → disable bare item fallback
    """

    if ADD_STRONG_VERB_PAT.search(text):
        return True

    if ADD_DESIRE_PAT.search(text):
        return True

    if ADD_CONFIRMATION_PAT.search(text):
        return True

    if QUANTITY_ITEM_PAT.search(text):
        return True

    # ⚠️ LAST RESORT (dangerous)
    if allow_bare and BARE_ITEM_PAT.fullmatch(text):
        return True

    return False

