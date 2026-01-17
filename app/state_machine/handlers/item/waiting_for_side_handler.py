# app/state_machine/handlers/item/waiting_for_side_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.menu_repository import MenuRepository


class WaitingForSideHandler(BaseHandler):
    """
    Handles side selection for items that require sides.
    Enforces selection one side group at a time.
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
    ) -> HandlerResult:

        # Cancellation is always allowed
        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        item = self.menu_repo.items.get(context.current_item_id)
        if not item:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        # Find the next incomplete side group
        next_group = None
        for group in item.side_groups:
            selected = context.selected_side_groups.get(group.group_id, [])
            if group.is_required and len(selected) < group.min_selector:
                next_group = group
                break

        if not next_group:
            # All required side groups are satisfied
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIZE,
                response_key="side_selection_complete",
            )

        # Try to match user utterance against choices
        user_text_lower = user_text.lower()
        matched_choice = None

        for choice in next_group.choices:
            if choice.name.lower() in user_text_lower:
                matched_choice = choice
                break

        if not matched_choice:
            # Invalid or unrelated input → re-prompt same group
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_SIDE,
                response_key="repeat_side_options",
            )

        # We have a valid side choice → ask for confirmation
        context.awaiting_confirmation_for = {
            "type": "side",
            "group_id": next_group.group_id,
            "value_id": matched_choice.item_id,
        }

        return HandlerResult(
            next_state=ConversationState.CONFIRMING,
            response_key="confirm_side_selection",
        )
