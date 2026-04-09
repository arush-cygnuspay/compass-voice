# app/state_machine/handlers/payment/waiting_for_payment_handler.py

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class WaitingForPaymentHandler(BaseHandler):
    """
    Handles the payment waiting state.

    Behavior:
    - payment done / confirm -> complete order and clear cart
    - payment request / payment methods -> resend payment link
    - payment status / review intents -> remind user payment is pending
    - cancel_order -> cancel checkout flow, keep cart
    - deny / cancel -> remain waiting for payment
    """

    RESEND_PAYMENT_INTENTS = {
        Intent.PAYMENT_REQUEST,
        Intent.PAYMENT_METHODS_QUERY,
    }

    STATUS_INTENTS = {
        Intent.PAYMENT_STATUS,
        Intent.SHOW_CART,
        Intent.SHOW_TOTAL,
        Intent.REVIEW_ORDER,
        Intent.ORDER_STATUS_GENERAL,
        Intent.ORDER_PROCESSING_STATUS,
        Intent.ORDER_PLACEMENT_STATUS,
        Intent.ORDER_ERROR_STATUS,
    }

    COMPLETE_PAYMENT_INTENTS = {
        Intent.PAYMENT_DONE,
    }

    def handle(self, intent, context, user_text, session=None):
        if session is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if session.conversation_state != ConversationState.WAITING_FOR_PAYMENT:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if intent in self.COMPLETE_PAYMENT_INTENTS:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_completed",
                reset_context=True,
                command={"type": "CLEAR_CART"},
            )

        if intent == Intent.CANCEL_ORDER:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_cancelled",
            )

        if intent in self.RESEND_PAYMENT_INTENTS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_link_sent",
            )

        if intent in self.STATUS_INTENTS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="waiting_for_payment",
            )

        if intent in {Intent.DENY, Intent.CANCEL}:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="waiting_for_payment",
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_PAYMENT,
            response_key="waiting_for_payment",
        )