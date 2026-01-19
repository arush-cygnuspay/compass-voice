# app/state_machine/handlers/payment/waiting_for_payment_handler.py
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.base_handler import BaseHandler
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class WaitingForPaymentHandler(BaseHandler):
    def handle(self, intent, context, user_text, session=None):

        if intent == Intent.PAYMENT_DONE:
            session.cart.clear()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_completed",
                reset_context=True,
            )

        if intent == Intent.PAYMENT_REQUEST:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_link_sent",
            )

        if intent in {Intent.DENY, Intent.CANCEL}:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_cancelled",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_PAYMENT,
            response_key="waiting_for_payment",
        )
