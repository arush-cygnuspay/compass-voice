# app/state_machine/handlers/item/idle_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.menu_repository import MenuRepository


class IdleHandler(BaseHandler):
    """
    Handles user intents when the system is in IDLE state.
    Entry point for ADD_ITEM flow.
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
    ) -> HandlerResult:

        if intent != Intent.ADD_ITEM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="idle_unhandled_intent",
            )

        # Attempt to resolve item from menu
        item = self.menu_repo.find_item_by_name(user_text)

        if not item:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_not_found",
            )

        # Store candidate item and move to confirmation
        context.candidate_item_id = item.item_id
        context.awaiting_confirmation_for = {
            "type": "item",
            "value_id": item.item_id,
        }

        return HandlerResult(
            next_state=ConversationState.CONFIRMING,
            response_key="confirm_item_candidate",
        )
