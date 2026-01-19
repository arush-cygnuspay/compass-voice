# app/nlu/intent_resolution/intent_resolver.py

from typing import Set

from app.nlu.intent_resolution.common_intent_resolver import match_yes_no
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.intent_resolution.item_intent_resolver import match_add_item
from app.nlu.intent_resolution.order_intent_resolver import match_order_intent
from app.nlu.intent_resolution.payment_intent_resolver import match_payment_intent
from app.nlu.intent_resolution.priority import INTENT_PRIORITY
from app.state_machine.conversation_state import ConversationState
from app.utils.text_utils import normalize_text


def resolve_intent(text: str, state: ConversationState) -> IntentResult:
    """
    Resolve linguistic intent from raw user text.

    GUARANTEES:
    - Pure
    - Deterministic
    - Regex-based only
    """

    if not text:
        return IntentResult(Intent.UNKNOWN, "")

    normalized = normalize_text(text)
    matches: set[Intent] = set()

    # ----------------------------------
    # YES / NO (state-sensitive)
    # ----------------------------------
    yn = match_yes_no(normalized)
    if yn:
        matches.add(yn)

    # ----------------------------------
    # ORDER intents (always allowed)
    # ----------------------------------
    matches |= match_order_intent(normalized)
    matches |= match_payment_intent(normalized)

    # ----------------------------------
    # ADD ITEM (ONLY when IDLE)
    # ----------------------------------
    if state == ConversationState.IDLE:
        if match_add_item(normalized):
            matches.add(Intent.ADD_ITEM)

    # ----------------------------------
    # PRIORITY RESOLUTION
    # ----------------------------------
    for intent in INTENT_PRIORITY:
        if intent in matches:
            return IntentResult(intent=intent, raw_text=normalized)

    return IntentResult(Intent.UNKNOWN, normalized)

