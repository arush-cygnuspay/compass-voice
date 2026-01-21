# app/state_machine/state_router.py

from typing import Dict, Set

from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.route_result import RouteResult


class StateRouter:
    """
    Authoritative router that decides whether a detected intent
    is allowed to execute in the current conversation state.

    IMPORTANT:
    - Cart utilities are overlays, NOT states
    - SHOWING_* states do not exist
    """

    def __init__(self) -> None:
        # Only TASK states belong here
        self._allowed_intents: Dict[ConversationState, Set[Intent]] = {
            ConversationState.GREETING: {
                Intent.CONFIRM,
                Intent.UNKNOWN,
            },

            ConversationState.IDLE: {
                Intent.ADD_ITEM,
                Intent.MODIFY_ITEM,
                Intent.REMOVE_ITEM,
                Intent.SHOW_MENU,
                Intent.END_ADDING,
                Intent.START_ORDER,
            },

            ConversationState.WAITING_FOR_SIDE: {
                Intent.CONFIRM,
                Intent.DENY,
                Intent.CANCEL,
                Intent.UNKNOWN,
            },

            ConversationState.WAITING_FOR_MODIFIER: {
                Intent.CONFIRM,
                Intent.DENY,
                Intent.CANCEL,
                Intent.UNKNOWN,
            },

            ConversationState.WAITING_FOR_SIZE: {
                Intent.CONFIRM,
                Intent.DENY,
                Intent.CANCEL,
                Intent.UNKNOWN,
            },

            ConversationState.WAITING_FOR_QUANTITY: {
                Intent.CONFIRM,
                Intent.DENY,
                Intent.CANCEL,
                Intent.UNKNOWN,
            },

            ConversationState.MODIFYING_ITEM: {
                Intent.CONFIRM,
                Intent.DENY,
                Intent.CANCEL,
                Intent.UNKNOWN,
            },

            ConversationState.REMOVING_ITEM: {
                Intent.CONFIRM,
                Intent.DENY,
                Intent.CANCEL,
            },

            ConversationState.CONFIRMING_ORDER: {
                Intent.CONFIRM,
                Intent.DENY,
                Intent.CANCEL,
            },

            ConversationState.CANCELLATION_CONFIRMATION: {
                Intent.CONFIRM,
                Intent.DENY,
            },

            ConversationState.WAITING_FOR_PAYMENT: {
                Intent.PAYMENT_DONE,
                Intent.DENY,
                Intent.CANCEL,
                Intent.UNKNOWN,
            },

            ConversationState.ERROR_RECOVERY: {
                Intent.CONFIRM,
                Intent.CANCEL,
                Intent.UNKNOWN,
            },
        }

    def route(self, state: ConversationState, intent_result: IntentResult) -> RouteResult:
        intent = intent_result.intent

        # --------------------------------------------------
        # CART OVERLAYS (GLOBAL, NON-TASK)
        # --------------------------------------------------
        if intent in {Intent.SHOW_CART, Intent.SHOW_TOTAL}:
            if state == ConversationState.WAITING_FOR_PAYMENT:
                return RouteResult(allowed=False)

            return RouteResult(
                allowed=True,
                handler_name="cart_handler",
            )

        # --------------------------------------------------
        # CLEAR CART (DESTRUCTIVE, STRICT)
        # --------------------------------------------------
        if intent == Intent.CLEAR_CART:
            if state in {
                ConversationState.IDLE,
                ConversationState.CONFIRMING_ORDER,
            }:
                return RouteResult(
                    allowed=True,
                    handler_name="cart_handler",
                )

            return RouteResult(allowed=False)

        # --------------------------------------------------
        # ADD ITEM (ENTRY POINT)
        # --------------------------------------------------
        if state == ConversationState.IDLE and intent == Intent.ADD_ITEM:
            return RouteResult(
                allowed=True,
                handler_name="add_item_handler",
            )

        # --------------------------------------------------
        # END ADDING / START ORDER
        # --------------------------------------------------
        if state == ConversationState.IDLE and intent in {
            Intent.END_ADDING,
            Intent.START_ORDER,
        }:
            return RouteResult(
                allowed=True,
                handler_name="start_order_handler",
            )

        # --------------------------------------------------
        # PAYMENT FLOW
        # --------------------------------------------------
        if state == ConversationState.WAITING_FOR_PAYMENT:
            return RouteResult(
                allowed=True,
                handler_name="waiting_for_payment_handler",
            )

        # --------------------------------------------------
        # DEFAULT TASK-STATE ROUTING
        # --------------------------------------------------
        allowed_intents = self._allowed_intents.get(state, set())
        if intent in allowed_intents:
            return RouteResult(
                allowed=True,
                handler_name=f"{state.name.lower()}_handler",
            )

        # --------------------------------------------------
        # GLOBAL CANCEL (TASKS ONLY)
        # --------------------------------------------------
        if intent == Intent.CANCEL and state != ConversationState.IDLE:
            return RouteResult(
                allowed=True,
                handler_name="cancel_handler",
            )

        return RouteResult(allowed=False)
