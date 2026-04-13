# app/core/flow_control/flow_control_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.core.flow_control.flow_decision import FlowAction, FlowDecision
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


DELIVERY_GATING_STATES: set[ConversationState] = {
    ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
    ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
}

ACTIVE_TASK_STATES: set[ConversationState] = {
    ConversationState.WAITING_FOR_ORDER_TYPE,
    ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
    ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
    ConversationState.CONFIRMING_ITEM,
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
    ConversationState.REMOVING_ITEM,
    ConversationState.MODIFYING_ITEM,
    ConversationState.CONFIRMING_ORDER,
    ConversationState.WAITING_FOR_PAYMENT,
    ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
}

MID_ITEM_BLOCKING_STATES: set[ConversationState] = {
    ConversationState.CONFIRMING_ITEM,
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
    ConversationState.REMOVING_ITEM,
    ConversationState.MODIFYING_ITEM,
}

CHECKOUT_ATTEMPT_INTENTS: set[Intent] = {
    Intent.START_ORDER,
    Intent.END_ADDING,
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
    Intent.PAYMENT_REQUEST,
    Intent.REVIEW_ORDER,
}

READ_ONLY_INTERRUPT_INTENTS: set[Intent] = {
    Intent.ASK_PRICE,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.AVAILABILITY_QUERY,
}


@dataclass(frozen=True, slots=True)
class _GuardPayload:
    state: ConversationState
    context: ConversationContext

    def cancel_payload(self) -> Mapping[str, Any]:
        return {
            "state": self.state.value,
            "item_name": self.context.current_item_name,
        }


class FlowControlPolicy:
    def evaluate(
        self,
        *,
        state: ConversationState,
        intent: Intent,
        context: ConversationContext,
    ) -> FlowDecision:
        payload = _GuardPayload(state=state, context=context)

        if intent in CHECKOUT_ATTEMPT_INTENTS and state in MID_ITEM_BLOCKING_STATES:
            return FlowDecision(
                action=FlowAction.BLOCK,
                response_key="checkout_blocked_finish_current_item",
                response_payload={"state": state.value},
            )

        if (
            intent in {Intent.CANCEL, Intent.CANCEL_ORDER}
            and state in ACTIVE_TASK_STATES
            and state != ConversationState.CANCELLATION_CONFIRMATION
        ):
            return FlowDecision(
                action=FlowAction.CANCEL,
                response_key="flow_guard_confirm_cancel",
                response_payload=payload.cancel_payload(),
            )

        if state in DELIVERY_GATING_STATES:
            return FlowDecision(action=FlowAction.PASS)

        if (
            intent in READ_ONLY_INTERRUPT_INTENTS
            and state in ACTIVE_TASK_STATES
            and state != ConversationState.CANCELLATION_CONFIRMATION
        ):
            return FlowDecision(
                action=FlowAction.HANDLE_READONLY_INTERRUPT,
            )

        return FlowDecision(action=FlowAction.PASS)