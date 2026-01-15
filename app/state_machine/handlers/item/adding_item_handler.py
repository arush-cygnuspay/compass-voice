# app/state_machine/handlers/item/adding_item_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.nlu.intent import Intent
from app.state_machine.context import ConversationContext


class AddingItemHandler(BaseHandler):
    """
    Handles the ADDING_ITEM conversational state.
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
    ) -> HandlerResult:
        return HandlerResult(
            next_state=ConversationState.CONFIRMING,
            response_key="confirm_item_candidate",
        )
