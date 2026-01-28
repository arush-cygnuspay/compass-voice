# app/core/flow_control/flow_control_policy.py

from app.core.flow_control.flow_decision import FlowDecision, FlowAction
from app.core.flow_control.slot_interaction import SlotInteraction
from app.nlu.choice_signals.choice_signals import ChoiceSignal
from app.nlu.choice_signals.resolver import resolve_choice_signal
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.context import ConversationContext
from app.state_machine.conversation_state import ConversationState

WAITING_STATES = {
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
}


FORBIDDEN_GLOBAL_INTENTS = {
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.ADD_ITEM,
    Intent.REMOVE_ITEM,
    Intent.START_ORDER,
    Intent.CLEAR_CART,
    Intent.PAYMENT_REQUEST,
}


CLARIFICATION_INTENTS = {
    Intent.ASK_MENU_INFO,
    Intent.SHOW_MENU,
    Intent.ASK_PRICE,
    Intent.UNKNOWN,
}


class FlowControlPolicy:
    """
    Governs mid-flow behavior.
    Decides whether an intent should pass, be rewritten, blocked, or cancelled.
    """

    # app/core/flow_control/flow_control_policy.py

    def evaluate(
            self,
            *,
            state: ConversationState,
            intent: Intent,
            context: ConversationContext,
    ) -> FlowDecision:
        """
        Central authority for mid-flow governance.

        This layer:
        - Prevents illegal transitions
        - Normalizes slot-level interactions
        - Keeps handlers simple and deterministic
        """

        # --------------------------------------------------
        # 1. Explicit intent-level CANCEL (highest priority)
        # --------------------------------------------------
        if intent == Intent.CANCEL and state != ConversationState.IDLE:
            return FlowDecision(
                action=FlowAction.CANCEL,
                response_key="flow_guard_confirm_cancel",
                response_payload={
                    "item_name": context.current_item_name,
                },
            )

        # --------------------------------------------------
        # 2. Slot-level interactions (OPTIONS / SKIP)
        #    These are NOT intents — they are control signals
        # --------------------------------------------------
        if state in WAITING_STATES:
            signal = resolve_choice_signal(context.last_user_text)

            if signal == ChoiceSignal.ASK_OPTIONS:
                return FlowDecision(
                    action=FlowAction.PASS,
                    slot_interaction=SlotInteraction.ASK_OPTIONS,
                )

            if signal == ChoiceSignal.DENY:
                return FlowDecision(
                    action=FlowAction.PASS,
                    slot_interaction=SlotInteraction.SKIP,
                )

        # --------------------------------------------------
        # 3. Confirm / deny must always reach handlers
        # --------------------------------------------------
        if intent in {Intent.CONFIRM, Intent.DENY}:
            return FlowDecision(action=FlowAction.PASS)

        # --------------------------------------------------
        # 4. Clarification / help during slot filling
        # --------------------------------------------------
        if state in WAITING_STATES and intent in CLARIFICATION_INTENTS:
            return FlowDecision(
                action=FlowAction.REWRITE,
                effective_intent=Intent.UNKNOWN,
            )

        # --------------------------------------------------
        # 5. Forbidden global actions mid-flow
        # --------------------------------------------------
        if state in WAITING_STATES and intent in FORBIDDEN_GLOBAL_INTENTS:
            return FlowDecision(
                action=FlowAction.BLOCK,
                response_key="flow_guard_finish_current_step",
                response_payload={
                    "state": state.name,
                    "current_step": state.name.lower().replace("waiting_for_", ""),
                    "item_name": context.current_item_name,
                },
            )

        # --------------------------------------------------
        # 6. Default: allow routing
        # --------------------------------------------------
        return FlowDecision(action=FlowAction.PASS)
