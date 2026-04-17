# app/state_machine/handlers/item/remove_item_handler.py
from app.session.session import Session
from app.core.pending_action import PendingAction
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.conversation_context import ConversationContext
from app.nlu.intent_resolution.intent import Intent
from app.menu.repository import MenuRepository
from app.state_machine.handlers.item.cart_edit_support import (
    extract_item_slot_values,
    match_cart_item_from_text,
    resolve_menu_item_from_text,
    split_replacement_request,
)


class RemoveItemHandler(BaseHandler):
    """
    Handles basic cart edit intents from idle.

    Supported:
    - remove item
    - undo last item
    - replace item
    - modify item (rebuild the same base item from scratch)
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session = None,
    ) -> HandlerResult:

        session_id = session.session_id if session else "n/a"

        if intent not in {
            Intent.REMOVE_ITEM,
            Intent.REPLACE_ITEM,
            Intent.MODIFY_ITEM,
            Intent.UNDO_LAST,
        }:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="unhandled_intent",
            )

        # Check if cart is empty
        if session.cart.is_empty():
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="cart_is_empty",
            )

        # -----------------------------
        # Resolve item from cart
        # -----------------------------
        if intent == Intent.UNDO_LAST:
            cart_items = session.cart.get_items()
            matched_cart_item = cart_items[-1] if cart_items else None
        else:
            matched_cart_item = self._match_cart_item(user_text, context, session)

        if not matched_cart_item:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_not_found_in_cart",
            )

        menu_item = self.menu_repo.get_item(matched_cart_item.item_id)
        context.current_item_name = menu_item.name

        if intent in {Intent.REMOVE_ITEM, Intent.UNDO_LAST}:
            context.pending_action = PendingAction.REMOVE_ITEM
            context.candidate_item_id = matched_cart_item.cart_item_id
            context.current_item_id = matched_cart_item.item_id
            context.awaiting_confirmation_for = {
                "type": "remove_item",
                "cart_item_id": matched_cart_item.cart_item_id,
                "item_id": matched_cart_item.item_id,
                "item_name": menu_item.name,
            }

            return HandlerResult(
                next_state=ConversationState.REMOVING_ITEM,
                response_key="confirm_remove_item",
                response_payload={
                    "item_name": menu_item.name,
                    "quantity": matched_cart_item.quantity,
                },
            )

        if intent == Intent.MODIFY_ITEM:
            context.pending_action = PendingAction.MODIFY_ITEM
            context.candidate_item_id = matched_cart_item.cart_item_id
            context.current_item_id = matched_cart_item.item_id
            context.awaiting_confirmation_for = {
                "type": "modify_item",
                "cart_item_id": matched_cart_item.cart_item_id,
                "item_id": matched_cart_item.item_id,
                "item_name": menu_item.name,
            }
            return HandlerResult(
                next_state=ConversationState.MODIFYING_ITEM,
                response_key="confirm_modify_item",
                response_payload={"item_name": menu_item.name},
            )

        replacement_item = self._resolve_replacement_item(
            user_text=user_text,
            context=context,
            cart_item=matched_cart_item,
        )
        if replacement_item is None:
            context.pending_action = PendingAction.MODIFY_ITEM
            context.candidate_item_id = matched_cart_item.cart_item_id
            context.current_item_id = matched_cart_item.item_id
            context.awaiting_confirmation_for = {
                "type": "replace_item_collect",
                "cart_item_id": matched_cart_item.cart_item_id,
                "item_id": matched_cart_item.item_id,
                "item_name": menu_item.name,
            }
            return HandlerResult(
                next_state=ConversationState.MODIFYING_ITEM,
                response_key="ask_replacement_item",
                response_payload={"item_name": menu_item.name},
            )

        context.pending_action = PendingAction.MODIFY_ITEM
        context.candidate_item_id = matched_cart_item.cart_item_id
        context.current_item_id = matched_cart_item.item_id
        context.awaiting_confirmation_for = {
            "type": "replace_item",
            "cart_item_id": matched_cart_item.cart_item_id,
            "item_id": matched_cart_item.item_id,
            "item_name": menu_item.name,
            "replacement_item_id": replacement_item.item_id,
            "replacement_item_name": replacement_item.name,
        }
        return HandlerResult(
            next_state=ConversationState.MODIFYING_ITEM,
            response_key="confirm_replace_item",
            response_payload={
                "item_name": menu_item.name,
                "replacement_item_name": replacement_item.name,
            },
        )

    def _match_cart_item(
        self,
        user_text: str,
        context: ConversationContext,
        session: Session,
    ):
        slot_candidates = extract_item_slot_values(context)
        left_text, _ = split_replacement_request(user_text)
        candidates = slot_candidates + ([left_text] if left_text else []) + [user_text]
        return match_cart_item_from_text(
            menu_repo=self.menu_repo,
            session=session,
            candidate_texts=candidates,
        )

    def _resolve_replacement_item(
        self,
        *,
        user_text: str,
        context: ConversationContext,
        cart_item,
    ):
        slot_candidates = extract_item_slot_values(context)
        old_menu_item = self.menu_repo.get_item(cart_item.item_id)
        exclude_item_ids = {cart_item.item_id}

        for candidate in slot_candidates:
            resolved = resolve_menu_item_from_text(
                self.menu_repo,
                candidate,
                exclude_item_ids=exclude_item_ids,
            )
            if resolved is not None:
                return resolved

        _, replacement_text = split_replacement_request(user_text)
        if replacement_text:
            resolved = resolve_menu_item_from_text(
                self.menu_repo,
                replacement_text,
                exclude_item_ids=exclude_item_ids,
            )
            if resolved is not None:
                return resolved

        if old_menu_item.name and replacement_text == old_menu_item.name:
            return None

        return None
