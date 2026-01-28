# app/state_machine/handlers/common/waiting_for_quantity_handler.py

from app.session.session import Session
from app.state_machine.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.conversation_state import ConversationState
from app.state_machine.context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.utils.quantity_detection import detect_quantity


class WaitingForQuantityHandler(BaseHandler):
    """
    Global quantity handler.
    Used by ADD, MODIFY, REMOVE flows.
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session = None,
    ) -> HandlerResult:

        # -------- Cancel --------
        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        # -------- Extract quantity --------
        quantity_info = detect_quantity(user_text)

        if not quantity_info:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="repeat_quantity_prompt",
            )

        if quantity_info["type"] == "vague":
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="clarify_exact_quantity",
            )

        quantity = quantity_info["value"]

        if quantity is None or quantity <= 0:
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="repeat_quantity_prompt",
            )

        # -------- Persist quantity --------
        context.quantity = quantity

        # -------- Commit ADD --------
        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            command={
                "type": "ADD_ITEM_TO_CART",
                "payload": {
                    "item_id": context.current_item_id,
                    "quantity": quantity,
                    "variant_id": context.selected_variant_id,
                    "sides": context.selected_side_groups,
                    "modifiers": context.selected_modifier_groups,
                },
            },
            response_payload={
                "item_id": context.current_item_id,
                "item_name": context.current_item_name,
                "quantity": quantity,
            },
            reset_context=True,
        )
