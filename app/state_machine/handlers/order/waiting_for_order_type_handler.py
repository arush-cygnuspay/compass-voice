# app/state_machine/handlers/common/waiting_for_order_type_handler.py
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler

PICKUP_WORDS = {
    "pickup",
    "pick up",
    "carryout",
    "carry out",
    "takeout",
    "take out",
    "collection",
}

DELIVERY_WORDS = {
    "delivery",
    "deliver",
    "drop off",
    "dropoff",
    "send it",
}


class WaitingForOrderTypeHandler(BaseHandler):
    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        normalized = " ".join((user_text or "").strip().lower().split())

        order_type = self._resolve_order_type(normalized)
        if order_type is None:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_ORDER_TYPE,
                response_key="repeat_order_type",
                response_payload=None,
            )

        context.order_type = order_type
        context.delivery_address_required = order_type == "delivery"
        context.delivery_address_confirmed = False

        if order_type == "pickup":
            context.delivery_address.reset_for_new_delivery()
            context.onboarding_complete = True
            context.current_prompt_field = None

            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="order_type_captured_pickup",
                response_payload={"order_type": "pickup"},
            )

        context.delivery_address.reset_for_new_delivery()
        context.onboarding_complete = False
        context.current_prompt_field = "delivery_area"

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
            response_key="ask_for_delivery_area",
            response_payload={"order_type": "delivery"},
        )

    def _resolve_order_type(self, text: str) -> str | None:
        if not text:
            return None

        for phrase in DELIVERY_WORDS:
            if phrase in text:
                return "delivery"

        for phrase in PICKUP_WORDS:
            if phrase in text:
                return "pickup"

        return None