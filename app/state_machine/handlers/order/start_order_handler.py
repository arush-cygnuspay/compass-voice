# app/state_machine/handlers/order/start_order_handler.py

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class StartOrderHandler(BaseHandler):
    """
    Entry point for order review / checkout initiation from IDLE.

    Behavior:
    - checkout-like intents -> show summary and move to CONFIRMING_ORDER
    - cancel_order in IDLE -> nothing active to cancel
    - payment/order status in IDLE -> no active payment/order procedure
    """

    ENTRY_INTENTS = {
        Intent.START_ORDER,
        Intent.END_ADDING,
        Intent.CHECKOUT,
        Intent.CONFIRM_ORDER,
        Intent.FINISH_ORDER,
        Intent.PAYMENT_REQUEST,
        Intent.REVIEW_ORDER,
    }

    STATUS_INTENTS = {
        Intent.PAYMENT_STATUS,
        Intent.ORDER_STATUS_GENERAL,
        Intent.ORDER_PLACEMENT_STATUS,
        Intent.ORDER_PROCESSING_STATUS,
        Intent.ORDER_ERROR_STATUS,
    }

    def __init__(self, cart_summary_builder):
        self.cart_summary_builder = cart_summary_builder

    def handle(self, intent, context, user_text, session=None):
        if session is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if intent == Intent.CANCEL_ORDER:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="no_active_order_to_cancel",
            )

        if intent in self.STATUS_INTENTS:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="no_active_payment",
            )

        if intent not in self.ENTRY_INTENTS:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="intent_not_allowed",
            )

        if session.cart.is_empty():
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_empty",
            )

        payload = self.cart_summary_builder.build(session.cart)

        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="confirm_order_summary",
            response_payload=payload,
        )