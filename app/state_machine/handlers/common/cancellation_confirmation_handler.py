# app/state_machine/handlers/common/cancellation_confirmation_handler.py

from __future__ import annotations

import re

from app.intent.confirmation_utils import resolve_confirmation_decision
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.common.waiting_for_quantity_handler import (
    WaitingForQuantityHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.utils.quantity_detection import normalize_quantity


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

        if (
            not self._is_clear_cart_confirmation(context)
            and context.return_state == ConversationState.WAITING_FOR_QUANTITY
            and self._looks_like_quantity_reply(user_text, context)
        ):
            self._clear_flow_confirmation_state(context)
            return WaitingForQuantityHandler().handle(
                intent=Intent.UNKNOWN,
                context=context,
                user_text=user_text,
                session=session,
            )

        decision = resolve_confirmation_decision(
            context.last_nlu,
            user_text,
            resolved_intent=intent,
            expect_confirmation=True,
        )

        # -------------------------------------------------
        # 1) User confirms cancellation / destructive action
        # -------------------------------------------------
        if decision == "affirm":
            if self._is_clear_cart_confirmation(context):
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="cart_cleared",
                    command={"type": "CLEAR_CART"},
                    reset_context=True,
                )

            interrupt_proposal = context.interrupt_proposal

            context.reset_task()
            # Clear the multi-item queue on cancellation — user is
            # abandoning the whole multi-item request.
            context.clear_item_queue()
            context.interrupt_proposal = interrupt_proposal

            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_cancelled_successfully",
            )

        # -------------------------------------------------
        # 2) User denies cancellation => resume prior flow
        # -------------------------------------------------
        if decision == "cancel":
            self._clear_flow_confirmation_state(context)
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        if decision == "deny":
            resume_state = context.return_state or ConversationState.IDLE

            if self._is_clear_cart_confirmation(context):
                self._clear_flow_confirmation_state(context)

                return HandlerResult(
                    next_state=resume_state,
                    response_key="clear_cart_cancelled",
                )

            self._clear_flow_confirmation_state(context)

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

    def _clear_flow_confirmation_state(self, context: ConversationContext) -> None:
        context.awaiting_confirmation_for = None
        context.awaiting_flow_confirmation = False
        context.return_state = None
        context.interrupt_proposal = None

    def _looks_like_quantity_reply(
        self,
        user_text: str,
        context: ConversationContext,
    ) -> bool:
        slots = getattr(context, "last_slots", None)
        if slots:
            for slot in slots:
                if str(getattr(slot, "name", "")).upper() == "QUANTITY":
                    return True

        normalized = " ".join((user_text or "").lower().split())
        if not normalized:
            return False

        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = " ".join(normalized.split())

        for prefix in (
            "i said ",
            "its ",
            "it is ",
            "just ",
            "only ",
            "quantity is ",
            "make it ",
        ):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                break

        if not normalized:
            return False

        if len(normalized.split()) > 4:
            return False

        return normalize_quantity(normalized) is not None
