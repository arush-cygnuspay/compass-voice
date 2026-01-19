# app/nlu/intent_resolution/order_intent_resolver.py
"""
PURE ORDER LINGUISTIC MATCHER

Rules:
- No session
- No cart
- No state
- Regex only
"""

from app.nlu.intent_patterns.order import *
from app.nlu.intent_resolution.intent import Intent


def match_order_intent(text: str) -> set[Intent]:
    """
    Returns a SET of possible order-related intents.

    This function is:
    - Pure (no state awareness)
    - Linguistic only
    - Precedence-safe (caller resolves priority)
    """

    matches: set[Intent] = set()

    # 1️⃣ Order status (highest semantic priority)
    if ORDER_STATUS_PAT.search(text):
        matches.add(Intent.ORDER_STATUS)

    # 2️⃣ Explicit checkout / place order
    if ORDER_CONFIRM_EXPLICIT_PAT.search(text):
        matches.add(Intent.START_ORDER)

    # 3️⃣ Strong end-adding signals
    if END_ADDING_STRONG_PAT.search(text):
        matches.add(Intent.END_ADDING)

    # 4️⃣ Soft end-adding signals
    if END_ADDING_SOFT_PAT.search(text):
        matches.add(Intent.END_ADDING)

    # 5️⃣ Meta clarification (lowest power)
    if ORDER_META_CLARIFY_PAT.search(text):
        matches.add(Intent.META_CLARIFY)

    return matches

