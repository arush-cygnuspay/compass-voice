# app/state_machine/state_router.py

from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.route_result import RouteResult


WAITING_STATE_HANDLERS: dict[ConversationState, str] = {
    ConversationState.WAITING_FOR_SIDE: "waiting_for_side_handler",
    ConversationState.WAITING_FOR_SIDE_SIZE: "waiting_for_side_size_handler",
    ConversationState.WAITING_FOR_MODIFIER: "waiting_for_modifier_handler",
    ConversationState.WAITING_FOR_SIZE: "waiting_for_size_handler",
    ConversationState.WAITING_FOR_QUANTITY: "waiting_for_quantity_handler",
}

DIRECT_STATE_HANDLERS: dict[ConversationState, str] = {
    ConversationState.CONFIRMING_ITEM: "confirming_handler",
    ConversationState.REMOVING_ITEM: "removing_item_handler",
    ConversationState.CONFIRMING_ORDER: "confirming_order_handler",
    ConversationState.WAITING_FOR_PAYMENT: "waiting_for_payment_handler",
    ConversationState.CANCELLATION_CONFIRMATION: "cancellation_confirmation_handler",
}

IDLE_CHECKOUT_INTENTS: set[Intent] = {
    Intent.START_ORDER,
    Intent.END_ADDING,
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
    Intent.PAYMENT_REQUEST,
    Intent.REVIEW_ORDER,
}

IDLE_CART_INTENTS: set[Intent] = {
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.CLEAR_CART,
}

IDLE_MENU_INFO_INTENTS: set[Intent] = {
    Intent.BROWSE_MENU,
    Intent.BROWSE_CATEGORY,
    Intent.ASK_ITEM_INFO,
    Intent.ASK_OPTIONS,
    Intent.AVAILABILITY_QUERY,
    Intent.RECOMMENDATION_QUERY,
    Intent.SHOW_MENU,
    Intent.ASK_MENU_INFO,
}

GREETING_INTENTS: set[Intent] = {
    Intent.CONFIRM,
    Intent.UNKNOWN,
    Intent.GREETING,
    Intent.MORNING,
    Intent.AFTERNOON,
    Intent.EVENING,
    Intent.NIGHT,
}


IDLE_ORDER_SUPPORT_INTENTS: set[Intent] = {
    Intent.CANCEL_ORDER,
    Intent.PAYMENT_STATUS,
    Intent.ORDER_STATUS_GENERAL,
    Intent.ORDER_PLACEMENT_STATUS,
    Intent.ORDER_PROCESSING_STATUS,
    Intent.ORDER_ERROR_STATUS,
}


class StateRouter:
    """
    Deterministic router.

    Rules:
    - waiting states own the turn by default
    - direct task states map to a single dedicated handler
    - IDLE routes only valid entry intents
    - exceptional mid-flow interrupts are handled before routing by FlowControlPolicy / TurnEngine
    - everything else is rejected
    """

    def route(self, state: ConversationState, intent_result: IntentResult) -> RouteResult:
        intent = intent_result.intent

        # 1) Waiting states own the turn by default.
        # Mid-flow exceptions like ASK_PRICE / SHOW_CART / SHOW_TOTAL
        # must be intercepted earlier by FlowControlPolicy + TurnEngine.
        if state in WAITING_STATE_HANDLERS:
            return RouteResult(
                allowed=True,
                handler_name=WAITING_STATE_HANDLERS[state],
            )

        # 2) Direct task states
        if state in DIRECT_STATE_HANDLERS:
            return RouteResult(
                allowed=True,
                handler_name=DIRECT_STATE_HANDLERS[state],
            )

        # 3) IDLE entry routing
        if state == ConversationState.IDLE:
            if intent == Intent.ADD_ITEM:
                return RouteResult(
                    allowed=True,
                    handler_name="add_item_handler",
                )

            if intent == Intent.REMOVE_ITEM:
                return RouteResult(
                    allowed=True,
                    handler_name="remove_item_handler",
                )

            if intent in IDLE_CHECKOUT_INTENTS:
                return RouteResult(
                    allowed=True,
                    handler_name="start_order_handler",
                )

            if intent in IDLE_ORDER_SUPPORT_INTENTS:
                return RouteResult(
                    allowed=True,
                    handler_name="start_order_handler",
                )

            if intent in IDLE_CART_INTENTS:
                return RouteResult(
                    allowed=True,
                    handler_name="cart_handler",
                )

        # 4) Greeting
        if state == ConversationState.GREETING:
            if intent in GREETING_INTENTS:
                return RouteResult(
                    allowed=True,
                    handler_name="confirming_handler",
                )

            return RouteResult(
                allowed=False,
                reason=f"intent {intent.value} is not allowed in greeting",
            )

        # 5) Error recovery
        if state == ConversationState.ERROR_RECOVERY:
            return RouteResult(
                allowed=False,
                reason=f"no routing available from state {state.value}",
            )

        return RouteResult(
            allowed=False,
            reason=f"no route for state={state.value}, intent={intent.value}",
        )