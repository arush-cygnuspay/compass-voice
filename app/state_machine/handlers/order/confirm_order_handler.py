# app/state_machine/handlers/order/confirm_order_handler.py
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.base_handler import BaseHandler
from app.state_machine.context import ConversationContext
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class ConfirmOrderHandler(BaseHandler):
    """
    Handles order confirmation (yes / no).
    """

    def handle(self, intent, context, user_text, session=None):

        if intent in {Intent.DENY, Intent.CANCEL}:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_cancelled",
                reset_context=True,
            )

        if intent == Intent.CONFIRM:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_link_sent",
            )

        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="repeat_order_confirmation",
        )

