# app/state_machine/handlers/order/confirm_order_handler.py

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class ConfirmOrderHandler(BaseHandler):
    """
    Handles final order confirmation before payment.

    Behavior:
    - confirm / checkout-like intents -> move to WAITING_FOR_PAYMENT
    - review intents -> repeat order summary
    - payment status -> payment has not started yet
    - cancel order / deny / cancel -> exit checkout flow, keep cart
    """

    CHECKOUT_CONFIRM_INTENTS = {
        Intent.CONFIRM,
        Intent.CHECKOUT,
        Intent.CONFIRM_ORDER,
        Intent.FINISH_ORDER,
        Intent.PAYMENT_REQUEST,
    }

    REVIEW_INTENTS = {
        Intent.SHOW_CART,
        Intent.SHOW_TOTAL,
        Intent.REVIEW_ORDER,
    }

    def __init__(self, cart_summary_builder):
        self.cart_summary_builder = cart_summary_builder

    def handle(self, intent, context, user_text, session=None):
        if session is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if session.conversation_state != ConversationState.CONFIRMING_ORDER:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if session.cart.is_empty():
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_empty",
            )

        if intent in self.CHECKOUT_CONFIRM_INTENTS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_link_sent",
            )

        if intent == Intent.PAYMENT_STATUS:
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="payment_not_started",
            )

        if intent in {Intent.CANCEL_ORDER, Intent.DENY, Intent.CANCEL}:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_cancelled",
            )

        payload = self.cart_summary_builder.build(session.cart)

        if intent in self.REVIEW_INTENTS:
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="confirm_order_summary",
                response_payload=payload,
            )

        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="confirm_order_summary",
            response_payload=payload,
        )