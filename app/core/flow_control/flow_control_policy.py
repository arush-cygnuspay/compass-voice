# app/core/flow_control/flow_control_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.core.flow_control.flow_decision import FlowAction, FlowDecision
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


from app.state_machine.flow_sets import (
    ACTIVE_TASK_STATES,
    CHECKOUT_ATTEMPT_INTENTS,
    DELIVERY_GATING_STATES,
    MID_ITEM_BLOCKING_STATES,
    READ_ONLY_INTERRUPT_INTENTS,
)


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
