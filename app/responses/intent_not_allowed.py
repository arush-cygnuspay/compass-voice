# app/responses/intent_not_allowed.py

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.flow_sets import ADD_ITEM_FLOW_STATES


def handle_intent_not_allowed(payload: dict) -> str:
    state = payload.get("state")
    intent = payload.get("intent")

    if state == ConversationState.WAITING_FOR_ORDER_TYPE:
        return "Please say pickup or delivery."

    if state in ADD_ITEM_FLOW_STATES and intent == Intent.SHOW_CART:
        return "Finish this item first, or say cancel."

    if state in ADD_ITEM_FLOW_STATES:
        return "Please finish this item, or say cancel."

    if state == ConversationState.CONFIRMING_ORDER:
        return "Please confirm the order, or say cancel."

    if state == ConversationState.WAITING_FOR_PAYMENT:
        return "Please complete payment, or say cancel."

    return "I didn’t catch that. Please say it again."