# app/nlu/intent_resolution/cart_intent_resolver.py
from app.nlu.intent_patterns.cart import (
    SHOW_CART_PAT,
    SHOW_TOTAL_PAT,
    CLEAR_CART_PAT,
    CART_ITEM_QUERY_PAT,
)
from app.nlu.intent_resolution.intent import Intent


def match_cart_intent(text: str) -> set[Intent]:
    """
    Pure cart-related linguistic matcher.

    Rules:
    - No state
    - No routing
    - No side effects
    """

    matches: set[Intent] = set()

    if CLEAR_CART_PAT.search(text):
        matches.add(Intent.CLEAR_CART)

    if SHOW_TOTAL_PAT.search(text):
        matches.add(Intent.SHOW_TOTAL)

    if SHOW_CART_PAT.search(text):
        matches.add(Intent.SHOW_CART)

    # ⚠️ Reserved for future (not routed yet)
    if CART_ITEM_QUERY_PAT.search(text):
        matches.add(Intent.SHOW_CART)

    return matches
