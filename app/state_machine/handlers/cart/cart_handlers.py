# app/state_machine/handlers/cart/cart_handlers.py

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class CartHandler(BaseHandler):
    """
    Cart utility handler.

    Behavior:
    - SHOW_CART: read-only
    - SHOW_TOTAL: read-only
    - CLEAR_CART: confirmation-gated destructive action
    """

    def __init__(self, cart_summary_builder: CartSummaryBuilder):
        self.cart_summary_builder = cart_summary_builder

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        current_state = session.conversation_state if session else ConversationState.IDLE

        if intent == Intent.SHOW_CART:
            return HandlerResult(
                next_state=current_state,
                response_key="show_cart",
                response_payload=self.cart_summary_builder.build(session.cart),
            )

        if intent == Intent.SHOW_TOTAL:
            return HandlerResult(
                next_state=current_state,
                response_key="show_total",
                response_payload=self.cart_summary_builder.build(session.cart),
            )

        if intent == Intent.CLEAR_CART:
            if session is None or session.cart.is_empty():
                return HandlerResult(
                    next_state=current_state,
                    response_key="cart_empty",
                )

            context.return_state = current_state
            context.awaiting_flow_confirmation = True
            context.interrupt_proposal = None
            context.awaiting_confirmation_for = {"type": "clear_cart"}

            return HandlerResult(
                next_state=ConversationState.CANCELLATION_CONFIRMATION,
                response_key="confirm_clear_cart",
            )

        return HandlerResult(
            next_state=current_state,
            response_key="intent_not_allowed",
        )