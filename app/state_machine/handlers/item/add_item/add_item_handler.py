# app/state_machine/handlers/item/add_item/add_item_handler.py
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
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.state_machine.handlers.item.add_item.side_group_resolver import (
    SideGroupResolver,
    extract_side_slot_values_normalized,
)
from app.state_machine.handlers.item.add_item.modifier_group_resolver import (
    ModifierGroupResolver,
    extract_modifier_slot_values_normalized,
)
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
        self.side_resolver = SideGroupResolver()
        self.modifier_resolver = ModifierGroupResolver()

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
        slots = self._get_last_slots(context)
        item_slot_value = first_slot_value(slots, "ITEM", "MENU_ITEM")
        category_slot_value = first_slot_value(slots, "CATEGORY", "MENU_CATEGORY")

        context.reset_task()
        context.pending_action = PendingAction.ADD_ITEM
        context.awaiting_flow_confirmation = False
        context.interrupt_proposal = None
        context.awaiting_confirmation_for = None

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

        # 1) prefill main item size if present
        self._prefill_item_variant(
            context=context,
            user_text=user_text,
            slots=slots,
        )

        # 2) prefill side selections from first utterance
        self._prefill_side_groups(
            context=context,
            normalized_user_text=user_text,
        )

        # 3) prefill side sizes for already selected side items
        self._prefill_selected_side_variants(
            context=context,
            user_text=user_text,
            slots=slots,
        )

        # 4) prefill modifiers (structured: add/remove/extra/less)
        self._prefill_modifier_groups(
            context=context,
            normalized_user_text=user_text,
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

    def _prefill_side_groups(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
    ) -> None:
        pending = context.pending_add_item
        if pending is None or not pending.side_groups:
            return

        slot_values = extract_side_slot_values_normalized(context)

        for group in pending.side_groups:
            existing_ids = list(context.selected_side_groups.get(group.group_id, []))
            if existing_ids:
                continue

            resolution = self.side_resolver.resolve(
                group=group,
                normalized_user_text=normalized_user_text,
                normalized_slot_values=slot_values,
                already_selected_ids=existing_ids,
            )
            if not resolution.matched_item_ids:
                continue

            capped_ids = resolution.matched_item_ids[: int(group.max_selector or 1)]
            context.selected_side_groups[group.group_id] = capped_ids
            context.skipped_side_groups.discard(group.group_id)

    def _prefill_selected_side_variants(
        self,
        *,
        context: ConversationContext,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> None:
        """
        Prefill side size only when it is safe.

        Current safe rule:
        - exactly one selected side item exists that needs a variant
        - exactly one size expression can be extracted
        - that size matches one of that side item's available variants

        This covers first-turn utterances like:
        - "2 chicken burgers with small coke"
        - "burger with medium sprite"
        """
        pending = context.pending_add_item
        if pending is None:
            return

        selected_variant_side_choices = []
        for group in pending.side_groups:
            for selected_item_id in context.selected_side_groups.get(group.group_id, []):
                choice = group.choices_by_item_id.get(selected_item_id)
                if choice is None:
                    continue
                if choice.pricing_mode != "variant":
                    continue
                if selected_item_id in context.selected_side_variants:
                    continue
                if not choice.variants:
                    continue
                selected_variant_side_choices.append(choice)

        if len(selected_variant_side_choices) != 1:
            return

        requested_size = self._extract_requested_size(user_text=user_text, slots=slots)
        if not requested_size:
            return

        side_choice = selected_variant_side_choices[0]
        matched_variant = self._match_variant_label(
            requested_size=requested_size,
            pending_variants=side_choice.variants,
        )
        if matched_variant is None:
            return

        context.selected_side_variants[side_choice.item_id] = matched_variant.variant_id

    def _prefill_modifier_groups(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
    ) -> None:
        pending = context.pending_add_item
        if pending is None or not pending.modifier_groups:
            return

        slot_values = extract_modifier_slot_values_normalized(context)

        for group in pending.modifier_groups:
            existing_selections = list(context.selected_modifier_groups.get(group.group_id, []))
            existing_ids = [sel.modifier_id for sel in existing_selections]
            if existing_selections:
                continue

            resolution = self.modifier_resolver.resolve(
                group=group,
                normalized_user_text=normalized_user_text,
                normalized_slot_values=slot_values,
                already_selected_ids=existing_ids,
            )
            if not resolution.selections:
                continue

            capped = resolution.selections[: int(group.max_selector or 1)]
            context.selected_modifier_groups[group.group_id] = capped
            context.skipped_modifier_groups.discard(group.group_id)

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
        slot_size = first_slot_value(slots, "SIZE", "VARIANT")
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
