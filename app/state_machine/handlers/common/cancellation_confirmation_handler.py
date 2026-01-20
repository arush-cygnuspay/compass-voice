# app/state_machine/handlers/common/cancellation_confirmation_handler.py

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.base_handler import BaseHandler
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class CancellationConfirmationHandler(BaseHandler):
    """
    Confirms destructive flow-level actions (e.g. CLEAR_CART).
    Does NOT deal with item-building confirmations.
    """

    def handle(self, intent, context, user_text, session=None):

        # 🛑 Safety guard: wrong state
        if session.conversation_state != ConversationState.CANCELLATION_CONFIRMATION:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        # ✅ User confirms destructive action
        if intent == Intent.CONFIRM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_cleared",
                command={"type": "CLEAR_CART"},
                reset_context=True,
            )

        # ❌ User denies destructive action
        if intent in {Intent.DENY, Intent.CANCEL}:
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ORDER,
                response_key="resume_order_confirmation",
            )

        # Repeat confirmation if unclear
        return HandlerResult(
            next_state=ConversationState.CANCELLATION_CONFIRMATION,
            response_key="confirm_clear_cart",
        )

