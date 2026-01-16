# app/state_machine/handlers/item/add_item_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent import Intent
from app.menu.repository import MenuRepository


class AddItemHandler(BaseHandler):
    """
    Handles initial item addition intent.
    Resolves item using MenuRepository (entity-first).
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
                response_key="unhandled_intent",
            )

        # -----------------------------
        # Resolve item from menu
        # -----------------------------
        item = self.menu_repo.resolve_item(user_text)

        if not item:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_not_found",
            )

        # -----------------------------
        # Ask for confirmation
        # -----------------------------
        context.candidate_item_id = item.item_id
        context.awaiting_confirmation_for = {
            "type": "item",
            "value_id": item.item_id,
        }

        return HandlerResult(
            next_state=ConversationState.CONFIRMING,
            response_key="confirm_item",
        )
