# app/state_machine/handlers/common/waiting_for_quantity_handler.py
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.conversation_context import ConversationContext, InterruptProposal
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.utils.quantity_detection import detect_quantity


SOFT_SWITCH_INTENTS: set[Intent] = {
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.SHOW_MENU,
    Intent.ASK_MENU_INFO,
    Intent.ASK_PRICE,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.START_ORDER,
    Intent.END_ADDING,
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
    Intent.REVIEW_ORDER,
    Intent.PAYMENT_REQUEST,
}


class WaitingForQuantityHandler(BaseHandler):
    """
    Resolve quantity while the add-item flow is waiting for it.

    Rules:
    - prefer the already-resolved QUANTITY slot when available
    - fallback to text-based quantity detection
    - accept short direct answers like "2" or "three"
    - if quantity is resolved, continue canonical add-item flow immediately
    - if quantity is not resolved and the user tries a different flow-level action,
      route through cancellation confirmation
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        pending = context.pending_add_item

        if pending is None or not context.current_item_id:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        normalized_user_text = (user_text or "").strip()

        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_cancelled_successfully",
            )

        if intent == Intent.ASK_OPTIONS:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="ask_for_quantity",
                response_payload={"item_name": pending.item_name},
            )

        extracted_quantity = self._extract_quantity_from_context_or_text(
            context=context,
            user_text=normalized_user_text,
        )

        if extracted_quantity is not None:
            if extracted_quantity <= 0:
                return HandlerResult(
                    next_state=ConversationState.WAITING_FOR_QUANTITY,
                    response_key="invalid_quantity_option",
                    response_payload={"item_name": pending.item_name},
                )

            context.quantity = extracted_quantity

            if context.current_prompt_field == "quantity":
                context.current_prompt_field = None

            if context.available_choices_kind == "quantity":
                context.available_choices_kind = None
                context.available_choices_values = ()

            step = determine_next_add_item_step(context)
            return self._step_to_result(context, step)

        if intent in SOFT_SWITCH_INTENTS:
            context.awaiting_flow_confirmation = True
            context.return_state = ConversationState.WAITING_FOR_QUANTITY
            context.interrupt_proposal = InterruptProposal(
                text=normalized_user_text,
                predicted_main_intent=None,
                predicted_sub_intent=intent.value,
            )

            return HandlerResult(
                next_state=ConversationState.CANCELLATION_CONFIRMATION,
                response_key="confirm_cancel_current_item_for_new_request",
                response_payload={"item_name": pending.item_name},
            )

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_QUANTITY,
            response_key="invalid_quantity_option",
            response_payload={"item_name": pending.item_name},
        )

    def _extract_quantity_from_context_or_text(
        self,
        *,
        context: ConversationContext,
        user_text: str,
    ) -> int | None:
        slots = getattr(context, "last_slots", ()) or ()
        for slot in slots:
            slot_name = str(getattr(slot, "name", "")).upper()
            if slot_name != "QUANTITY":
                continue

            value = getattr(slot, "value", None)

            if isinstance(value, int):
                return value

            if isinstance(value, str):
                stripped = value.strip()
                if stripped.isdigit():
                    return int(stripped)

        quantity_info = detect_quantity(user_text)
        if not quantity_info:
            return None

        if quantity_info.get("type") == "vague":
            return None

        value = quantity_info.get("value")
        if isinstance(value, int):
            return value

        return None

    def _step_to_result(self, context: ConversationContext, step) -> HandlerResult:
        pending = context.pending_add_item
        if pending is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        if step.next_state == ConversationState.FINALIZING_ADD_ITEM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload={
                    "item_name": pending.item_name,
                    "quantity": context.quantity or 1,
                },
                command=build_add_item_command(context),
                reset_context=True,
            )

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=step.response_payload,
        )