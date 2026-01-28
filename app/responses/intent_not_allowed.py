# app/responses/intent_not_allowed.py
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.conversation_state import ConversationState
from app.state_machine.flow_sets import ADD_ITEM_FLOW_STATES

def handle_intent_not_allowed(payload: dict) -> str:
    state = payload.get("state")
    intent = payload.get("intent")

    if state in ADD_ITEM_FLOW_STATES and intent == Intent.SHOW_CART:
        return "Let’s finish adding this item before viewing your cart."

    # ---- Add Item Flow ----
    if state in ADD_ITEM_FLOW_STATES:
        return (
            "You’re currently adding an item.\n"
            "Please finish adding it, or say **cancel** to stop."
        )

    # ---- Order Confirmation ----
    if state == ConversationState.CONFIRMING_ORDER:
        return (
            "You’re reviewing your order.\n"
            "Please confirm to proceed or say cancel."
        )

    # ---- Payment ----
    if state == ConversationState.WAITING_FOR_PAYMENT:
        return (
            "Your order is awaiting payment.\n"
            "Please complete the payment or say cancel."
        )

    # ---- Fallback (should be rare) ----
    return "Sorry, that action isn’t allowed right now."
