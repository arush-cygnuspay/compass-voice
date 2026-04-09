from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.common.order_type_resolver import OrderTypeResolver
from app.state_machine.conversation_context import ConversationContext
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler


class WaitingForOrderTypeHandler(BaseHandler):
    """
    Resolve the mandatory pre-order fulfillment type.

    Supports:
    - pickup / delivery as a standalone response
    - combined utterances like:
      'delivery and one pizza'
      'pickup add a burger'
      'it is pickup can i get fries'
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        match = OrderTypeResolver.resolve(user_text)

        if match is None:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_ORDER_TYPE,
                response_key="repeat_order_type",
                response_payload=None,
            )

        context.order_type = match.order_type
        context.onboarding_complete = True
        context.delivery_address_required = match.order_type == "delivery"
        context.delivery_address_confirmed = False

        response_key = (
            "order_type_captured_pickup"
            if match.order_type == "pickup"
            else "order_type_captured_delivery"
        )

        payload = {
            "order_type": match.order_type,
        }

        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key=response_key,
            response_payload=payload,
        )