# app/responses/flow_control_responses.py

def flow_guard_finish_current_step(payload: dict) -> str:
    """
    User tried a blocked action during an active flow.
    Keep the prompt short, but still natural for telephony.
    """
    item_name = payload.get("item_name")

    if item_name:
        return f"Please finish {item_name}, or say cancel."

    return "Please finish this step, or say cancel."


def flow_guard_confirm_cancel(payload: dict) -> str:
    """
    Confirm whether the user wants to cancel the current flow.
    """
    item_name = payload.get("item_name")

    if item_name:
        return f"Do you want to cancel {item_name}? Please say yes or no."

    return "Do you want to cancel this? Please say yes or no."


def flow_guard_cancelled(_: dict | None = None) -> str:
    """
    Flow was cancelled and context was reset.
    """
    return "Okay, cancelled. What would you like next?"