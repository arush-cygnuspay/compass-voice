from __future__ import annotations

from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


class ResumePromptBuilder:
    """
    Reconstruct the pending question for the active flow
    without mutating session state.
    """

    def build(self, session: Session) -> tuple[str, dict | None] | None:
        state = session.conversation_state
        context = session.conversation_context

        if state == ConversationState.WAITING_FOR_SIZE:
            return "ask_for_size", None

        if state == ConversationState.WAITING_FOR_SIDE:
            return "ask_for_side", None

        if state == ConversationState.WAITING_FOR_SIDE_SIZE:
            return "ask_for_side_size", None

        if state == ConversationState.WAITING_FOR_MODIFIER:
            return "ask_for_modifier", None

        if state == ConversationState.WAITING_FOR_QUANTITY:
            return "ask_for_quantity", {
                "item_name": context.current_item_name or "this item",
            }

        return None