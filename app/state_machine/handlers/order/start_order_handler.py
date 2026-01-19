# app/state_machine/handlers/order/start_order_handler.py
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.base_handler import BaseHandler
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class StartOrderHandler(BaseHandler):
    """
    Entry point when user indicates they are done adding items.
    """

    def __init__(self, cart_summary_builder):
        self.cart_summary_builder = cart_summary_builder

    def handle(self, intent, context, user_text, session=None):

        if intent not in {Intent.END_ADDING, Intent.START_ORDER}:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="unhandled_intent",
            )

        if session.cart.is_empty():
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_empty",
            )

        summary = self.cart_summary_builder.build(session.cart)

        return HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="confirm_order_summary",
            response_payload=summary,
        )


