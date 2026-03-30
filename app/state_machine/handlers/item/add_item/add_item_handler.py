from __future__ import annotations

import re
from typing import Sequence

from app.core.pending_action import PendingAction
from app.menu.models import MenuItem
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.repository import MenuRepository
from app.menu.slot_helpers import first_slot_value
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.conversation_context import ConversationContext
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.utils.quantity_detection import normalize_quantity
from app.utils.token_matcher import is_controlled_partial_match, is_strong_token_match

SIZE_WORDS = (
    "extra large",
    "large",
    "medium",
    "small",
    "regular",
    "mini",
    "xl",
)

ITEM_FILLER_PREFIXES: tuple[str, ...] = (
    "i want ",
    "i want a ",
    "i want an ",
    "i would like ",
    "i would like a ",
    "i would like an ",
    "i would like to order ",
    "i would like to get ",
    "i will take ",
    "ill take ",
    "can i get ",
    "give me ",
    "add ",
    "get ",
    "bring ",
    "make it ",
    "a ",
    "an ",
    "the ",
)


class AddItemHandler(BaseHandler):
    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        if intent != Intent.ADD_ITEM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="unhandled_intent",
            )

        normalized_user_text = self._normalize_item_request_text(user_text)

        context.reset_task()
        context.pending_action = PendingAction.ADD_ITEM
        context.awaiting_flow_confirmation = False
        context.interrupt_proposal = None
        context.awaiting_confirmation_for = None

        quantity = self._extract_quantity(normalized_user_text)
        if quantity is not None and quantity > 0:
            context.quantity = quantity

        slots = self._get_last_slots(context)
        item_slot_value = first_slot_value(slots, "ITEM", "MENU_ITEM")
        category_slot_value = first_slot_value(slots, "CATEGORY", "MENU_CATEGORY")

        if item_slot_value or category_slot_value:
            result = self.menu_repo.resolve_menu_query_from_slots_normalized(
                normalized_user_text=normalized_user_text,
                slots=slots,
                fallback_to_text=False,
                limit=5,
            )
            requested_item_text = item_slot_value or category_slot_value or normalized_user_text

            return self._route_menu_query_result(
                context=context,
                result=result,
                requested_item_text=normalize_text(requested_item_text),
                original_user_text=normalized_user_text,
                slots=slots,
            )

        result = self.menu_repo.resolve_menu_query_normalized(
            normalized_user_text,
            limit=5,
        )

        return self._route_menu_query_result(
            context=context,
            result=result,
            requested_item_text=normalized_user_text,
            original_user_text=normalized_user_text,
            slots=slots,
        )

    def _route_menu_query_result(
        self,
        *,
        context: ConversationContext,
        result: MenuQueryResult,
        requested_item_text: str,
        original_user_text: str,
        slots: Sequence[SlotValue],
    ) -> HandlerResult:
        rtype = result.type

        if rtype == MenuQueryType.ITEM and result.item is not None:
            return self._enter_add_flow_for_item(
                context=context,
                item=result.item,
                user_text=original_user_text,
                slots=slots,
            )

        if (
            rtype == MenuQueryType.CATEGORY_SINGLE_ITEM
            and result.items
            and len(result.items) == 1
        ):
            return self._enter_add_flow_for_item(
                context=context,
                item=result.items[0],
                user_text=original_user_text,
                slots=slots,
            )

        if rtype == MenuQueryType.CATEGORY:
            candidate_items = result.items or []
            payload = {
                "reason": "category_detected",
                "query": requested_item_text,
                "category_id": result.category_id,
                "category_name": result.category_name,
                "candidate_item_ids": [item.item_id for item in candidate_items],
                "candidate_item_names": [item.name for item in candidate_items],
            }

            context.awaiting_confirmation_for = {
                "type": "item",
                "reason": "category_detected",
                "query": requested_item_text,
                "category_id": result.category_id,
                "category_name": result.category_name,
                "candidate_item_ids": payload["candidate_item_ids"],
                "candidate_item_names": payload["candidate_item_names"],
            }

            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key="confirm_item_from_category",
                response_payload=payload,
            )

        if rtype in {MenuQueryType.ITEM_AMBIGUOUS, MenuQueryType.CATEGORY_AMBIGUOUS}:
            payload = {
                "reason": "multiple_matches",
                "query": requested_item_text,
            }

            if result.matched_items:
                payload["candidate_item_ids"] = [item.item_id for item in result.matched_items]
                payload["candidate_item_names"] = [item.name for item in result.matched_items]

            if result.matched_categories:
                payload["candidate_category_names"] = [
                    category.get("name")
                    for category in result.matched_categories
                    if category.get("name")
                ]

            context.awaiting_confirmation_for = {
                "type": "item",
                "reason": "multiple_matches",
                "query": requested_item_text,
                "candidate_item_ids": payload.get("candidate_item_ids", []),
                "candidate_item_names": payload.get("candidate_item_names", []),
                "candidate_category_names": payload.get("candidate_category_names", []),
            }

            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key="confirm_item_ambiguous",
                response_payload=payload,
            )

        context.reset_task()
        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_not_found",
            response_payload={
                "query": requested_item_text,
                "suggested_item_names": [item.name for item in (result.suggested_items or [])],
                "suggested_category_names": [
                    category.get("name")
                    for category in (result.suggested_categories or [])
                    if category.get("name")
                ],
            },
        )

    def _enter_add_flow_for_item(
        self,
        *,
        context: ConversationContext,
        item: MenuItem,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> HandlerResult:
        context.current_item_id = item.item_id
        context.current_item_name = item.name
        context.candidate_item_id = item.item_id
        context.pending_add_item = build_pending_add_item(item)

        self._prefill_item_variant(
            context=context,
            user_text=user_text,
            slots=slots,
        )

        step = determine_next_add_item_step(context)

        if step.next_state == ConversationState.FINALIZING_ADD_ITEM:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload={
                    "item_name": item.name,
                    "quantity": context.quantity or 1,
                },
                command=build_add_item_command(context),
                reset_context=True,
            )

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=step.response_payload,
        )

    def _prefill_item_variant(
        self,
        *,
        context: ConversationContext,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> None:
        pending = context.pending_add_item
        if pending is None or not pending.item_variants:
            return

        requested_size = self._extract_requested_size(user_text=user_text, slots=slots)
        if not requested_size:
            return

        matched_variant = self._match_variant_label(
            requested_size=requested_size,
            pending_variants=pending.item_variants,
        )
        if matched_variant is None:
            return

        context.selected_variant_id = matched_variant.variant_id
        context.size_target = None

    def _extract_requested_size(
        self,
        *,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> str | None:
        slot_size = first_slot_value(slots, "SIZE")
        if isinstance(slot_size, str) and slot_size.strip():
            return normalize_text(slot_size)

        normalized_user_text = user_text or ""
        for size in SIZE_WORDS:
            if re.search(rf"\b{re.escape(size)}\b", normalized_user_text):
                return normalize_text(size)

        return None

    def _match_variant_label(
        self,
        *,
        requested_size: str,
        pending_variants,
    ) -> object | None:
        if not requested_size:
            return None

        for variant in pending_variants:
            if variant.normalized_name == requested_size:
                return variant

        for variant in pending_variants:
            if is_strong_token_match(requested_size, variant.normalized_name):
                return variant

        for variant in pending_variants:
            if is_controlled_partial_match(requested_size, variant.normalized_name):
                return variant

        return None

    def _get_last_slots(self, context: ConversationContext) -> Sequence[SlotValue]:
        return context.last_slots or ()

    def _extract_quantity(self, text: str) -> int | None:
        try:
            return normalize_quantity(text)
        except Exception:
            return None

    def _normalize_item_request_text(self, text: str) -> str:
        normalized = normalize_text(text or "")
        if not normalized:
            return ""

        changed = True
        while changed and normalized:
            changed = False
            for prefix in ITEM_FILLER_PREFIXES:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
                    changed = True
                    break

        return normalized