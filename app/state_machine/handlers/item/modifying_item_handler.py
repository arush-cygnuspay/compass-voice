from __future__ import annotations

from app.core.pending_action import PendingAction
from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import (
    build_pending_add_item,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class ModifyingItemHandler(BaseHandler):
    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session = None,
    ) -> HandlerResult:
        if session is None:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        confirmation = context.awaiting_confirmation_for or {}
        action_type = confirmation.get("type")

        if not action_type:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        if intent == Intent.CANCEL:
            context.reset()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="action_cancelled",
            )

        if action_type == "replace_item_collect":
            if intent == Intent.DENY:
                context.reset()
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="item_replacement_cancelled",
                )

            replacement_item = self._resolve_replacement_item(user_text)
            if replacement_item is None:
                return HandlerResult(
                    next_state=ConversationState.MODIFYING_ITEM,
                    response_key="ask_replacement_item",
                    response_payload={"item_name": confirmation.get("item_name", "that item")},
                )

            context.awaiting_confirmation_for = {
                **confirmation,
                "type": "replace_item",
                "replacement_item_id": replacement_item.item_id,
                "replacement_item_name": replacement_item.name,
            }
            return HandlerResult(
                next_state=ConversationState.MODIFYING_ITEM,
                response_key="confirm_replace_item",
                response_payload={
                    "item_name": confirmation.get("item_name", "that item"),
                    "replacement_item_name": replacement_item.name,
                },
            )

        if intent == Intent.DENY:
            context.reset()
            cancel_key = "item_modification_cancelled" if action_type == "modify_item" else "item_replacement_cancelled"
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key=cancel_key,
            )

        if intent != Intent.CONFIRM:
            repeat_key = "confirm_modify_item" if action_type == "modify_item" else "confirm_replace_item"
            return HandlerResult(
                next_state=ConversationState.MODIFYING_ITEM,
                response_key=repeat_key,
                response_payload={
                    "item_name": confirmation.get("item_name", "that item"),
                    "replacement_item_name": confirmation.get("replacement_item_name"),
                },
            )

        target_item_id = confirmation.get("item_id")
        if action_type == "replace_item":
            target_item_id = confirmation.get("replacement_item_id")

        if not target_item_id:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        try:
            item = self.menu_repo.get_item(target_item_id)
        except KeyError:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="item_context_missing",
            )

        cart_item_id = confirmation.get("cart_item_id")
        if not cart_item_id:
            return HandlerResult(
                next_state=ConversationState.ERROR_RECOVERY,
                response_key="confirmation_state_error",
            )

        return self._enter_add_flow_for_item(
            context=context,
            item=item,
            cart_item_id=cart_item_id,
        )

    def _resolve_replacement_item(self, user_text: str):
        result = self.menu_repo.resolve_menu_query(user_text, limit=5)
        if result.type == MenuQueryType.ITEM and result.item is not None:
            return result.item
        if (
            result.type == MenuQueryType.CATEGORY_SINGLE_ITEM
            and result.items
            and len(result.items) == 1
        ):
            return result.items[0]
        return None

    def _enter_add_flow_for_item(
        self,
        *,
        context: ConversationContext,
        item,
        cart_item_id: str,
    ) -> HandlerResult:
        context.reset_task()
        context.pending_action = PendingAction.MODIFY_ITEM
        context.current_item_id = item.item_id
        context.current_item_name = item.name
        context.candidate_item_id = item.item_id
        context.pending_add_item = build_pending_add_item(item)

        step = determine_next_add_item_step(context)

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=step.response_payload,
            command={
                "type": "REMOVE_ITEM_FROM_CART",
                "payload": {
                    "cart_item_id": cart_item_id,
                },
            },
        )
