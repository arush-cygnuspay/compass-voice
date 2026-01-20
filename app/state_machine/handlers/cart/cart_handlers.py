# app/state_machine/handlers/cart/cart_handlers.py

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.context import ConversationContext
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class CartHandler(BaseHandler):
    """
    Read-only cart utility handler.
    Can safely interrupt tasks and MUST return to previous state.
    """

    def __init__(self, cart_summary_builder: CartSummaryBuilder):
        self.cart_summary_builder = cart_summary_builder

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session = None,
    ) -> HandlerResult:

        # 🔁 Store return_state ONLY for read-only cart views
        if intent in {Intent.SHOW_CART, Intent.SHOW_TOTAL}:
            if context.return_state is None:
                context.return_state = session.conversation_state

        # -------------------------
        # SHOW CART
        # -------------------------
        if intent == Intent.SHOW_CART:
            return HandlerResult(
                next_state=ConversationState.SHOWING_CART,
                response_key="show_cart",
                response_payload=self.cart_summary_builder.build(session.cart),
            )

        # -------------------------
        # SHOW TOTAL
        # -------------------------
        if intent == Intent.SHOW_TOTAL:
            return HandlerResult(
                next_state=ConversationState.SHOWING_TOTAL,
                response_key="show_total",
                response_payload=self.cart_summary_builder.build(session.cart),
            )

        # -------------------------
        # CLEAR CART (destructive)
        # -------------------------
        if intent == Intent.CLEAR_CART:

            # Requires confirmation if order is being confirmed
            if session.conversation_state == ConversationState.CONFIRMING_ORDER:
                return HandlerResult(
                    next_state=ConversationState.CANCELLATION_CONFIRMATION,
                    response_key="confirm_clear_cart",
                )

            # Safe immediate clear from IDLE
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_cleared",
                command={"type": "CLEAR_CART"},
            )

        # Defensive fallback
        return HandlerResult(
            next_state=session.conversation_state,
            response_key="intent_not_allowed",
        )


class ShowingCartHandler(BaseHandler):
    def handle(self, intent, context, user_text, session=None):

        if intent in {Intent.CONFIRM, Intent.CANCEL, Intent.DENY}:
            next_state = context.return_state or ConversationState.IDLE
            context.return_state = None

            return HandlerResult(
                next_state=next_state,
                response_key="resume_previous_state",
            )

        return HandlerResult(
            next_state=ConversationState.SHOWING_CART,
            response_key="awaiting_ack",
        )


class ShowingTotalHandler(BaseHandler):
    def handle(self, intent, context, user_text, session=None):

        if intent in {Intent.CONFIRM, Intent.CANCEL, Intent.DENY}:
            next_state = context.return_state or ConversationState.IDLE
            context.return_state = None

            return HandlerResult(
                next_state=next_state,
                response_key="resume_previous_state",
            )

        return HandlerResult(
            next_state=ConversationState.SHOWING_TOTAL,
            response_key="awaiting_ack",
        )
