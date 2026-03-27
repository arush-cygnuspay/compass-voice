# app/responses/flow_control_responses.py

from app.state_machine.conversation_state import ConversationState


def flow_guard_finish_current_step(payload: dict) -> str:
    """
    User attempted a forbidden global action while mid-flow.
    """
    step = payload.get("current_step", "this step")
    item_name = payload.get("item_name")

    if item_name:
        return f"Please finish the {step} for your {item_name}, or say cancel."

    return "Please finish this step, choose an option, or say cancel."


def flow_guard_confirm_cancel(payload: dict) -> str:
    """
    Ask the user to confirm cancelling the current flow.
    """
    item_name = payload.get("item_name")

    if item_name:
        return f"Do you want to cancel {item_name}? Please say yes or no."

    return "Do you want to cancel this? Please say yes or no."


def flow_guard_cancelled(_: dict | None = None) -> str:
    """
    Flow was cancelled and context reset.
    """
    return "Okay, cancelled. What would you like next?"