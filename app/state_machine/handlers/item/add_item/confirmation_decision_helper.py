# app/state_machine/handlers/item/add_item/confirmation_decision_helper.py
from __future__ import annotations

from app.menu.models import MenuItem
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class ConfirmationDecisionHelper:
    """Interprets the next-step determination and builds the final HandlerResult.

    Receives the post-prefill context and the prefill summaries, calls
    determine_next_add_item_step(), and returns the appropriate HandlerResult:
    either an immediate add-to-cart (FINALIZING_ADD_ITEM) or a waiting-state
    prompt for the next unresolved group.
    """

    def build_handler_result(
        self,
        *,
        context: ConversationContext,
        item: MenuItem,
        prefilled_summary: str,
        prefill_feedback: str,
        prefill_debug: dict,
    ) -> HandlerResult:
        step = determine_next_add_item_step(context)

        if step.next_state == ConversationState.FINALIZING_ADD_ITEM:
            payload: dict = {
                "item_name": item.name,
                "quantity": context.quantity or 1,
                "prefilled_summary": prefilled_summary,
                "prefill_debug": prefill_debug,
            }
            if prefill_feedback:
                payload["prefill_feedback"] = prefill_feedback
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload=payload,
                command=build_add_item_command(context),
                reset_context=True,
            )

        payload = dict(step.response_payload or {})
        if prefilled_summary:
            payload["prefilled_summary"] = prefilled_summary
            payload["prefilled_item_name"] = item.name
        if prefill_feedback:
            payload["prefill_feedback"] = prefill_feedback
        payload["prefill_debug"] = prefill_debug

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=payload,
        )
