# app/responses/flow_control_responses.py
from app.state_machine.conversation_state import ConversationState


def flow_guard_finish_current_step(payload: dict) -> str:
    """
    User tried a blocked global action while in an active flow.
    Keep the prompt short for telephony.
    """
    step = payload.get("current_step", "step")
    item_name = payload.get("item_name")

    if item_name:
        return f"Finish the {step} for {item_name}, or say cancel."

    return "Finish this step, or say cancel."


def flow_guard_confirm_cancel(payload: dict) -> str:
    """
    Ask the user to confirm flow cancellation.
    Keep the wording concise and direct.
    """
    item_name = payload.get("item_name")

    if item_name:
        return f"Cancel {item_name}? Say yes or no."

    return "Cancel this? Say yes or no."


def flow_guard_cancelled(_: dict | None = None) -> str:
    """
    Flow was cancelled and context reset.
    """
    return "Okay, cancelled. What next?"