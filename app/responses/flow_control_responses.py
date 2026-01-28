# app/responses/flow_control_responses.py

from app.state_machine.conversation_state import ConversationState

def flow_guard_finish_current_step(payload: dict) -> str:
    """
    User attempted a forbidden global action while mid-flow.
    """
    state = payload.get("state")
    step = payload.get("current_step", "this step")
    item_name = payload.get("item_name")

    if item_name:
        return (
            f"Let’s finish selecting {step} for your {item_name} first.\n"
            "You can say an option, ask for choices, or say cancel."
        )

    return (
        f"Let’s finish this step first.\n"
        "You can choose an option, ask for choices, or say cancel."
    )


def flow_guard_confirm_cancel(payload: dict) -> str:
    """
    Ask the user to confirm cancelling the current flow.
    """
    item_name = payload.get("item_name")

    if item_name:
        return (
            f"Do you want to stop adding {item_name}?\n"
            "Please say yes to cancel or no to continue."
        )

    return (
        "Do you want to cancel what we’re doing right now?\n"
        "Please say yes to cancel or no to continue."
    )


def flow_guard_cancelled(_: dict | None = None) -> str:
    """
    Flow was cancelled and context reset.
    """
    return (
        "Alright, I’ve cancelled that.\n"
        "What would you like to do next?"
    )
