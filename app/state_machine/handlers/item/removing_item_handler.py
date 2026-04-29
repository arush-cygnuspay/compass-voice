from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.control_intent_resolver import (
    ControlIntentKind,
    control_intent_to_confirmation_decision,
    log_control_intent_event,
    resolve_control_intent,
)
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class RemovingItemHandler(BaseHandler):
    """
    Handles confirmation for item removal.
    User confirms or denies removing the item from cart.
    """

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session = None,
    ) -> HandlerResult:
        confirmation = context.awaiting_confirmation_for
        control_intent = resolve_control_intent(
            user_text,
            intent,
            getattr(context.last_nlu, "model_sub_intent", None),
            ConversationState.REMOVING_ITEM,
            context,
            nlu_result=context.last_nlu,
            intent_confidence=context.last_intent_confidence,
        )
        decision = control_intent_to_confirmation_decision(control_intent)

        if not confirmation or confirmation.get("type") != "remove_item":
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if control_intent is not None and control_intent.kind == ControlIntentKind.META_CLARIFY:
            log_control_intent_event(
                "meta_clarify_repeated",
                state=ConversationState.REMOVING_ITEM.value,
                field_name="remove_item_confirmation",
            )
            return HandlerResult(
                next_state=ConversationState.REMOVING_ITEM,
                response_key="repeat_remove_confirmation",
            )

        if control_intent is not None and control_intent.kind == ControlIntentKind.OPTIONS_REQUEST:
            log_control_intent_event(
                "options_requested",
                state=ConversationState.REMOVING_ITEM.value,
                field_name="remove_item_confirmation",
            )
            return HandlerResult(
                next_state=ConversationState.REMOVING_ITEM,
                response_key="repeat_remove_confirmation",
            )

        if decision == "cancel":
            log_control_intent_event(
                "control_intent_action",
                state=ConversationState.REMOVING_ITEM.value,
                action="cancel_item_removal",
                kind=ControlIntentKind.CANCEL.value,
            )
            context.reset_item_scope()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        if decision == "deny":
            context.reset_item_scope()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_removal_cancelled",
            )

        if decision == "affirm":
            cart_item_id = confirmation.get("cart_item_id")

            if not cart_item_id:
                return HandlerResult(
                    next_state=ConversationState.ERROR_RECOVERY,
                    response_key="cart_item_id_missing",
                )

            item_name = confirmation.get("item_name", "item")
            context.reset_item_scope()

            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_removed_successfully",
                response_payload={
                    "item_name": item_name,
                },
                command={
                    "type": "REMOVE_ITEM_FROM_CART",
                    "payload": {
                        "cart_item_id": cart_item_id,
                    },
                },
            )

        return HandlerResult(
            next_state=ConversationState.REMOVING_ITEM,
            response_key="repeat_remove_confirmation",
        )
