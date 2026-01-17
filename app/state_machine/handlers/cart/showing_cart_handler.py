# app/state_machine/handlers/cart/showing_cart_handler.py

from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.nlu.intent_resolution.intent import Intent
from app.cart.cart import Cart


class ShowingCartHandler(BaseHandler):
    """
    Handles showing the current cart contents to the user.
    Read-only flow.
    """

    def __init__(self, cart: Cart) -> None:
        self.cart = cart

    def handle(self, intent: Intent, context) -> HandlerResult:
        # Defensive: this handler should only be called for SHOW_CART
        if intent != Intent.SHOW_CART:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_unhandled_intent",
            )

        if self.cart.is_empty():
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_empty",
            )

        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="cart_show",
        )
