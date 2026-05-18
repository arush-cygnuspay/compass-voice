# app/state_machine/handlers/order/waiting_for_order_type_handler.py
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.common.order_type_resolver import OrderTypeResolver
from app.state_machine.flow_sets import ORDERING_INTENTS as _ORDER_TYPE_ORDERING_INTENTS
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.common.preorder_redirect_utils import (
    looks_like_ordering_request,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class WaitingForOrderTypeHandler(BaseHandler):
    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        normalized = " ".join((user_text or "").strip().lower().split())

        # ── Ordering intents before order type selected → redirect ──
        if (
            intent in _ORDER_TYPE_ORDERING_INTENTS
            or looks_like_ordering_request(context, normalized)
        ):
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_ORDER_TYPE,
                response_key="ordering_blocked_need_order_type",
            )

        order_match = OrderTypeResolver.resolve(user_text)
        order_type = order_match.order_type if order_match is not None else None

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

        return HandlerResult(
            next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
            response_key="ask_for_delivery_area",
            response_payload={"order_type": "delivery"},
            prompt_field="delivery_area",
        )
