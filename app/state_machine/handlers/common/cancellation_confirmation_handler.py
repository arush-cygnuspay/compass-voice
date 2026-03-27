# app/state_machine/handlers/common/cancellation_confirmation_handler.py

from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.conversation_context import ConversationContext
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


class CancellationConfirmationHandler(BaseHandler):
    """
    Confirms flow-level cancellation / interruption actions.

    Handles:
    - destructive confirmations like CLEAR_CART
    - cancelling the current add-item flow
    - cancelling current item confirmation in order to switch to a new request

    Does NOT resolve item ambiguity itself. That belongs to ConfirmingHandler.
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        if session is None or session.conversation_state != ConversationState.CANCELLATION_CONFIRMATION:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        # -------------------------------------------------
        # 1) User confirms cancellation / destructive action
        # -------------------------------------------------
        if intent == Intent.CONFIRM:
            if self._is_clear_cart_confirmation(context):
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="cart_cleared",
                    command={"type": "CLEAR_CART"},
                    reset_context=True,
                )

            interrupt_proposal = context.interrupt_proposal

            context.reset_task()
            context.interrupt_proposal = interrupt_proposal

            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_cancelled_successfully",
            )

        # -------------------------------------------------
        # 2) User denies cancellation => resume prior flow
        # -------------------------------------------------
        if intent in {Intent.DENY, Intent.CANCEL}:
            resume_state = context.return_state or ConversationState.IDLE

            if self._is_clear_cart_confirmation(context):
                context.awaiting_confirmation_for = None
                context.awaiting_flow_confirmation = False
                context.return_state = None

                return HandlerResult(
                    next_state=resume_state,
                    response_key="clear_cart_cancelled",
                )

            return HandlerResult(
                next_state=resume_state,
                response_key="continue_current_item_after_cancel_denied",
                response_payload={
                    "field_name": context.current_prompt_field or "option",
                    "available_choices": list(context.available_choices_values),
                },
            )

        # -------------------------------------------------
        # 3) Unclear answer => repeat the appropriate confirmation
        # -------------------------------------------------
        if self._is_clear_cart_confirmation(context):
            return HandlerResult(
                next_state=ConversationState.CANCELLATION_CONFIRMATION,
                response_key="confirm_clear_cart",
            )

        item_name = context.current_item_name or "this item"
        if context.interrupt_proposal is not None:
            return HandlerResult(
                next_state=ConversationState.CANCELLATION_CONFIRMATION,
                response_key="confirm_cancel_current_item_for_new_request",
                response_payload={"item_name": item_name},
            )

        return HandlerResult(
            next_state=ConversationState.CANCELLATION_CONFIRMATION,
            response_key="confirm_cancel_current_item",
            response_payload={"item_name": item_name},
        )

    def _is_clear_cart_confirmation(self, context: ConversationContext) -> bool:
        confirmation = context.awaiting_confirmation_for or {}
        return confirmation.get("type") == "clear_cart"