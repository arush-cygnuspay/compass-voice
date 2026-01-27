# app/nlu/intent_resolution/cart_intent_resolver.py
from app.nlu.intent_patterns.cart import *
from app.nlu.intent_patterns.remove_item import *
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

    # REMOVE ITEM patterns
    if REMOVE_STRONG_VERB_PAT.search(text) or REMOVE_DESIRE_PAT.search(text) or REMOVE_ITEM_PAT.search(text):
        matches.add(Intent.REMOVE_ITEM)

    return matches
