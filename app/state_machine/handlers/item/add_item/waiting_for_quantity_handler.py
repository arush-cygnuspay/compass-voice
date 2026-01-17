# app/state_machine/handlers/item/waiting_for_quantity_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


class WaitingForQuantityHandler(BaseHandler):
    """
    Handles quantity selection for the current item.
    Final step of the add-item flow.
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
    ) -> HandlerResult:

        # Global cancel
        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        # Try parsing numeric quantity
        quantity = None
        text = user_text.lower().strip()

        if text.isdigit():
            quantity = int(text)
        else:
            quantity = NUMBER_WORDS.get(text)

        if not quantity or quantity <= 0:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="repeat_quantity_prompt",
            )

        # Ask for confirmation
        context.awaiting_confirmation_for = {
            "type": "quantity",
            "value": quantity,
        }

        return HandlerResult(
            next_state=ConversationState.CONFIRMING,
            response_key="confirm_quantity",
        )
