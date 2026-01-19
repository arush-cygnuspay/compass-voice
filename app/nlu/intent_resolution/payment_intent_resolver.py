# app/nlu/intent_resolution/payment_intent_resolver.py
from app.nlu.intent_patterns.payment import *
from app.nlu.intent_resolution.intent import Intent


def match_payment_intent(text: str) -> set[Intent]:
    """
    Matches payment-related user intents.

    Pure, linguistic only.
    No state awareness.
    """
    matches: set[Intent] = set()

    if PAYMENT_DONE_PAT.search(text):
        matches.add(Intent.PAYMENT_DONE)

    if PAYMENT_REQUEST_PAT.search(text):
        matches.add(Intent.PAYMENT_REQUEST)

    return matches
