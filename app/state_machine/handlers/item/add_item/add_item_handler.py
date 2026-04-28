# app/state_machine/handlers/item/add_item/add_item_handler.py
from __future__ import annotations

import logging
import re
from collections import deque
from typing import Sequence

from app.core.pending_action import PendingAction
from app.menu.models import MenuItem
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.repository import MenuRepository
from app.menu.slot_helpers import first_slot_value
from app.nlu.intent_resolution.intent import Intent
from app.nlu.multi_item_parser import parse_multi_item_utterance, ParsedItemSegment
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.nlu.slot_consumption import consume_slot_or_fallback
from app.session.session import Session
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import QueuedItemRequest
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.add_item_flow import (
    build_add_item_command,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.state_machine.handlers.item.add_item.side_group_resolver import (
    SideGroupResolver,
    build_side_option_candidates,
    extract_side_slot_values_normalized,
)
from app.state_machine.handlers.item.add_item.modifier_group_resolver import (
    ModifierGroupResolver,
    build_modifier_option_candidates,
    extract_modifier_slot_values_normalized,
)
from app.state_machine.handlers.item.add_item.multi_group_prefill import (
    MultiGroupPrefillEngine,
    PrefillResult,
)
from app.state_machine.handlers.item.add_item.option_matching import build_scoped_phrase_candidates
from app.utils.quantity_detection import UNIT_PATTERN, extract_leading_quantity_phrase, normalize_quantity
from app.utils.token_matcher import (
    is_controlled_partial_match,
    is_strong_token_match,
    tokenize,
)

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

logger = logging.getLogger(__name__)


def _parse_quantity_value(raw: str) -> int | None:
    """Coerce a raw QUANTITY slot string to a positive int (or None)."""
    text = (raw or "").strip()
    if not text:
        return None
    if text.isdigit():
        value = int(text)
        return value if value > 0 else None
    coerced = normalize_quantity(text)
    if isinstance(coerced, int) and coerced > 0:
        return coerced
    return None


class PendingItemCaptureHelper:
    def __init__(
        self,
        side_resolver: SideGroupResolver | None = None,
        modifier_resolver: ModifierGroupResolver | None = None,
    ) -> None:
        self.side_resolver = side_resolver or SideGroupResolver()
        self.modifier_resolver = modifier_resolver or ModifierGroupResolver()

    def prefill_quantity(
        self,
        *,
        context: ConversationContext,
        user_text: str,
    ) -> bool:
        if isinstance(context.quantity, int) and context.quantity > 0:
            return False

        for slot in context.last_slots or ():
            if str(getattr(slot, "name", "")).upper() != "QUANTITY":
                continue

            value = getattr(slot, "value", None)
            if isinstance(value, int) and value > 0:
                context.quantity = value
                return True
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.isdigit() and int(stripped) > 0:
                    context.quantity = int(stripped)
                    return True

        normalized_quantity = self._infer_quantity_from_text(
            context=context,
            user_text=user_text,
        )
        if isinstance(normalized_quantity, int) and normalized_quantity > 0:
            context.quantity = normalized_quantity
            return True

        return False

    def _infer_quantity_from_text(
        self,
        *,
        context: ConversationContext,
        user_text: str,
    ) -> int | None:
        normalized_text = normalize_text(user_text or "")
        if not normalized_text:
            return None

        def _extract_quantity_via_regex() -> int | None:
            # Exact quantity answers and explicit units are safe.
            if normalized_text.isdigit():
                return int(normalized_text)

            if re.fullmatch(r"(a|an|single|couple|one|two|three|four|five|six|seven|eight|nine|ten)", normalized_text):
                return normalize_quantity(normalized_text)

            if re.search(rf"\b(\d+|a|an|single|couple|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:{UNIT_PATTERN})\b", normalized_text):
                return normalize_quantity(normalized_text)

            if "dozen" in normalized_text:
                return normalize_quantity(normalized_text)

            leading = extract_leading_quantity_phrase(normalized_text)
            if leading is None:
                return None

            quantity, remainder, token = leading
            if not remainder:
                return quantity

            pending = context.pending_add_item
            if pending is None:
                return None

            pending_item_name = normalize_text(pending.item_name or "")
            if pending_item_name.startswith(token):
                return None

            item_slot_values = [
                normalize_text(str(getattr(slot, "value", "") or ""))
                for slot in (context.last_slots or ())
                if str(getattr(slot, "name", "")).upper() in {"ITEM", "MENU_ITEM"}
            ]
            candidate_names = [pending_item_name, *item_slot_values]
            candidate_names = [value for value in candidate_names if value]
            if not candidate_names:
                return None

            remainder_tokens = set(tokenize(remainder))
            if not remainder_tokens:
                return None

            for candidate_name in candidate_names:
                candidate_tokens = set(tokenize(candidate_name))
                if not candidate_tokens:
                    continue
                if candidate_tokens.issubset(remainder_tokens):
                    return quantity

            return None

        resolution = consume_slot_or_fallback(
            slots=context.last_slots or (),
            slot_labels=("QUANTITY",),
            fallback=_extract_quantity_via_regex,
            parse=_parse_quantity_value,
            consumer_site="add_item_handler.quantity",
        )
        return resolution.value

    def prefill_side_groups(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
        start_index: int = 0,
    ) -> list[dict]:
        pending = context.pending_add_item
        if pending is None or not pending.side_groups:
            return []

        slot_values = extract_side_slot_values_normalized(context)
        feedback: list[dict] = []

        for group in pending.side_groups[start_index:]:
            existing_ids = list(context.selected_side_groups.get(group.group_id, []))
            if existing_ids:
                continue

            resolution = self.side_resolver.resolve(
                group=group,
                normalized_user_text=normalized_user_text,
                option_candidates=build_side_option_candidates(context, normalized_user_text),
                normalized_slot_values=slot_values,
                already_selected_ids=existing_ids,
            )
            unmatched_values = self._clean_prefill_unmatched_values(
                list(resolution.unmatched_values),
                item_name=pending.item_name,
            )
            min_selector, max_selector = effective_group_selector_bounds(group)
            accepted_limit = max_selector if max_selector > 0 else len(resolution.matched_item_ids)

            if not resolution.matched_item_ids and not unmatched_values:
                continue

            capped_ids = resolution.matched_item_ids[:accepted_limit]
            dropped_ids = resolution.matched_item_ids[accepted_limit:]
            accepted_names = [
                group.choices_by_item_id[item_id].name
                for item_id in capped_ids
                if item_id in group.choices_by_item_id
            ]
            dropped_names = [
                group.choices_by_item_id[item_id].name
                for item_id in dropped_ids
                if item_id in group.choices_by_item_id
            ]

            over_max = bool(dropped_ids)
            if capped_ids and not over_max:
                context.selected_side_groups[group.group_id] = capped_ids
                context.skipped_side_groups.discard(group.group_id)

            feedback.append(
                {
                    "group_id": group.group_id,
                    "kind": "side",
                    "accepted_names": [] if over_max else accepted_names,
                    "requested_names": accepted_names + dropped_names,
                    "dropped_names": dropped_names,
                    "unmatched_names": unmatched_values,
                    "max_selector": max_selector,
                    "min_selector": min_selector,
                    "over_max": over_max,
                }
            )

        return feedback

    def prefill_selected_side_variants(
        self,
        *,
        context: ConversationContext,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> None:
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

        requested_size = self.extract_requested_size(user_text=user_text, slots=slots)
        if not requested_size:
            return

        side_choice = selected_variant_side_choices[0]
        matched_variant = self.match_variant_label(
            requested_size=requested_size,
            pending_variants=side_choice.variants,
        )
        if matched_variant is None:
            return

        context.selected_side_variants[side_choice.item_id] = matched_variant.variant_id

    def prefill_modifier_groups(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
        start_index: int = 0,
    ) -> list[dict]:
        pending = context.pending_add_item
        if pending is None or not pending.modifier_groups:
            return []

        slot_values = extract_modifier_slot_values_normalized(context)
        ignored_values = self._prefill_ignored_modifier_values(context)
        feedback: list[dict] = []

        for group in pending.modifier_groups[start_index:]:
            existing_selections = list(context.selected_modifier_groups.get(group.group_id, []))
            existing_ids = [sel.modifier_id for sel in existing_selections]
            if existing_selections:
                continue

            resolution = self.modifier_resolver.resolve(
                group=group,
                normalized_user_text=normalized_user_text,
                option_candidates=build_modifier_option_candidates(context, normalized_user_text),
                normalized_slot_values=slot_values,
                already_selected_ids=existing_ids,
                ignored_values=ignored_values,
                known_choice_phrases=self.all_modifier_choice_phrases(pending),
            )
            unmatched_values = self._clean_prefill_unmatched_values(
                list(resolution.unmatched_values),
                item_name=pending.item_name,
            )
            min_selector, max_selector = effective_group_selector_bounds(group)
            accepted_limit = max_selector if max_selector > 0 else len(resolution.selections)

            if not resolution.selections and not unmatched_values:
                continue

            capped = resolution.selections[:accepted_limit]
            dropped = resolution.selections[accepted_limit:]
            over_max = bool(dropped)

            if capped and not over_max:
                context.selected_modifier_groups[group.group_id] = capped
                context.skipped_modifier_groups.discard(group.group_id)

            feedback.append(
                {
                    "group_id": group.group_id,
                    "kind": "modifier",
                    "accepted_names": [] if over_max else [sel.name for sel in capped],
                    "requested_names": [sel.name for sel in resolution.selections],
                    "dropped_names": [sel.name for sel in dropped],
                    "unmatched_names": unmatched_values,
                    "max_selector": max_selector,
                    "min_selector": min_selector,
                    "over_max": over_max,
                }
            )

        return feedback

    def prefill_item_variant(
        self,
        *,
        context: ConversationContext,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> None:
        pending = context.pending_add_item
        if pending is None or not pending.item_variants:
            return

        requested_size = self.extract_requested_size(user_text=user_text, slots=slots)
        if not requested_size:
            return

        matched_variant = self.match_variant_label(
            requested_size=requested_size,
            pending_variants=pending.item_variants,
        )
        if matched_variant is None:
            return

        context.selected_variant_id = matched_variant.variant_id
        context.size_target = None

    @staticmethod
    def all_modifier_choice_phrases(pending) -> list[str]:
        phrases: list[str] = []
        seen: set[str] = set()
        for group in pending.modifier_groups:
            for choice in group.choices:
                for value in getattr(choice, "match_texts", ()) or (choice.normalized_name,):
                    if value and value not in seen:
                        seen.add(value)
                        phrases.append(value)
        return phrases

    @staticmethod
    def collect_matched_names(feedback_entries: list[dict]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for entry in feedback_entries:
            for name in entry.get("accepted_names") or []:
                cleaned = str(name).strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                names.append(cleaned)
        return names

    @staticmethod
    def extract_requested_size(
        *,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> str | None:
        normalized_user_text = user_text or ""

        def _extract_size_via_regex() -> str | None:
            for size in SIZE_WORDS:
                if re.search(rf"\b{re.escape(size)}\b", normalized_user_text):
                    return normalize_text(size)
            return None

        resolution = consume_slot_or_fallback(
            slots=slots,
            slot_labels=("SIZE", "VARIANT"),
            fallback=_extract_size_via_regex,
            parse=lambda raw: normalize_text(raw) or None,
            consumer_site="add_item_handler.size",
        )
        return resolution.value

    @staticmethod
    def match_variant_label(
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

    @staticmethod
    def _clean_prefill_unmatched_values(
        values: list[str],
        *,
        item_name: str,
    ) -> list[str]:
        normalized_item_name = normalize_text(item_name)
        return [
            value
            for value in values
            if value and normalize_text(value) != normalized_item_name
        ]

    @staticmethod
    def _prefill_ignored_modifier_values(context: ConversationContext) -> list[str]:
        pending = context.pending_add_item
        if pending is None:
            return []

        ignored: list[str] = []
        normalized_item_name = normalize_text(pending.item_name)
        if normalized_item_name:
            ignored.append(normalized_item_name)

        for group in pending.side_groups:
            for selected_item_id in context.selected_side_groups.get(group.group_id, []):
                choice = group.choices_by_item_id.get(selected_item_id)
                if choice and choice.normalized_name:
                    ignored.append(choice.normalized_name)
                    ignored.extend(getattr(choice, "match_texts", ()) or ())

        seen: set[str] = set()
        result: list[str] = []
        for value in ignored:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result


class AddItemHandler(BaseHandler):
    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo
        self.side_resolver = SideGroupResolver()
        self.modifier_resolver = ModifierGroupResolver()
        self.capture_helper = PendingItemCaptureHelper(
            side_resolver=self.side_resolver,
            modifier_resolver=self.modifier_resolver,
        )
        # Unified, segment-scoped prefill engine. Resolves every option
        # phrase in a segment against ALL valid groups for the pending item
        # in one pass — see multi_group_prefill.py for the rationale.
        self.prefill_engine = MultiGroupPrefillEngine()

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

        slot_aligned_user_text = normalize_text(user_text or "")
        normalized_user_text = self._normalize_item_request_text(user_text)
        slots = self._get_last_slots(context)

        # ── Multi-item detection ──────────────────────────────
        # If the user said multiple items in one go, queue the extras
        # and process only the first item now.
        multi_segments = parse_multi_item_utterance(
            slot_aligned_user_text,
            slots,
            menu_store=getattr(self.menu_repo, "store", None),
        )
        if len(multi_segments) >= 2:
            return self._handle_multi_item_utterance(
                context=context,
                segments=multi_segments,
                full_user_text=normalized_user_text,
            )

        # ── Single-item path (unchanged) ─────────────────────
        return self._handle_single_item(
            context=context,
            normalized_user_text=normalized_user_text,
            slots=slots,
        )

    def _handle_single_item(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
        slots: tuple[SlotValue, ...] | list[SlotValue],
    ) -> HandlerResult:
        """Original single-item add flow."""
        item_slot_value = first_slot_value(slots, "ITEM", "MENU_ITEM")
        category_slot_value = first_slot_value(slots, "CATEGORY", "MENU_CATEGORY")
        modifier_slot_value = first_slot_value(slots, "MODIFIER")

        context.reset_task()
        context.pending_action = PendingAction.ADD_ITEM
        context.awaiting_flow_confirmation = False
        context.interrupt_proposal = None
        context.awaiting_confirmation_for = None

        if (
            not item_slot_value
            and not category_slot_value
            and modifier_slot_value
            and self._looks_like_modifier_only_request(
                normalized_user_text=normalized_user_text,
                modifier_value=str(modifier_slot_value),
            )
        ):
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="modifier_requires_item_context",
                response_payload={"modifier_name": str(modifier_slot_value).strip()},
            )

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

    def _handle_multi_item_utterance(
        self,
        *,
        context: ConversationContext,
        segments: list[ParsedItemSegment],
        full_user_text: str,
    ) -> HandlerResult:
        """
        Handle a multi-item utterance.

        Strategy:
        1. Queue all items after the first one (preserving their slots).
        2. Build an acknowledgment summary of everything heard (with details).
        3. Start the add-item flow for the first item.
        """
        first_segment = segments[0]
        remaining_segments = segments[1:]

        # Queue the remaining items — preserve segment slots for better
        # modifier/side prefilling when dequeued.
        context.pending_item_queue = deque(
            QueuedItemRequest(
                raw_text=seg.raw_text,
                item_slot_value=seg.item_slot_value,
                quantity=seg.quantity,
                acknowledged=False,
                segment_slots=seg.slots or (),
            )
            for seg in remaining_segments
        )

        # Build detailed summary of what we heard (include modifiers/sides)
        item_summaries = []
        for seg in segments:
            item_summaries.append(self._build_segment_summary(seg))

        # Now process the first item through the normal single-item flow
        first_slots = first_segment.slots if first_segment.slots else self._get_last_slots(context)

        context.reset_task()
        context.pending_action = PendingAction.ADD_ITEM
        context.awaiting_flow_confirmation = False
        context.interrupt_proposal = None
        context.awaiting_confirmation_for = None
        context.last_slots = tuple(first_slots)

        if first_segment.quantity and first_segment.quantity > 0:
            context.quantity = first_segment.quantity

        # Resolve the first item
        first_text = first_segment.raw_text
        item_slot_value = first_segment.item_slot_value
        category_slot_value = first_slot_value(first_slots, "CATEGORY", "MENU_CATEGORY")

        if item_slot_value:
            result = self.menu_repo.resolve_menu_query_from_slots_normalized(
                normalized_user_text=first_text,
                slots=first_slots,
                fallback_to_text=True,
                limit=5,
            )
            requested_item_text = normalize_text(item_slot_value)
        else:
            result = self.menu_repo.resolve_menu_query_normalized(first_text, limit=5)
            requested_item_text = first_text

        handler_result = self._route_menu_query_result(
            context=context,
            result=result,
            requested_item_text=requested_item_text,
            original_user_text=first_text,
            slots=first_slots,
        )

        # Wrap the response with a multi-item acknowledgment prefix
        queue_count = len(context.pending_item_queue)
        payload = dict(handler_result.response_payload or {})
        payload["multi_item_ack"] = True
        payload["heard_items_summary"] = item_summaries
        payload["queue_count"] = queue_count
        payload["current_item_name"] = first_segment.item_slot_value or payload.get("item_name", "")
        payload["queued_item_names"] = [
            seg.item_slot_value or seg.raw_text for seg in remaining_segments
        ]

        return HandlerResult(
            next_state=handler_result.next_state,
            response_key=handler_result.response_key,
            response_payload=payload,
            command=handler_result.command,
            reset_context=handler_result.reset_context,
        )

    @staticmethod
    def _looks_like_modifier_only_request(
        *,
        normalized_user_text: str,
        modifier_value: str,
    ) -> bool:
        modifier_normalized = normalize_text(modifier_value or "")
        if not modifier_normalized:
            return False

        if normalized_user_text == modifier_normalized:
            return True

        prefixes = (
            "add ",
            "with ",
            "extra ",
            "more ",
            "light ",
            "less ",
            "no ",
            "without ",
            "hold ",
            "hold the ",
            "remove ",
            "remove the ",
        )
        for prefix in prefixes:
            if normalized_user_text == f"{prefix}{modifier_normalized}".strip():
                return True

        return False

    @staticmethod
    def _build_segment_summary(seg: ParsedItemSegment) -> str:
        """Build a concise spoken summary like '2 chicken tacos'."""
        qty_prefix = f"{seg.quantity} " if seg.quantity and seg.quantity > 1 else ""
        item_name = seg.item_slot_value or ""
        raw = (seg.raw_text or "").strip()
        return f"{qty_prefix}{item_name}".strip() or raw

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
        missing_groups_before_prefill = self._missing_group_names(context)
        prefill_user_text = self._prefill_segment_text_for_item(
            item_name=item.name,
            user_text=user_text,
        )

        # 0) Quantity is still handled separately because it can come from
        #    a leading numeric (e.g. "2 chicken tacos with...") rather than
        #    an option phrase.
        self.capture_helper.prefill_quantity(
            context=context,
            user_text=prefill_user_text,
        )

        # 1) Unified, segment-scoped prefill across variants + sides + modifiers.
        #    Every candidate phrase in the segment is scored against EVERY
        #    valid target on this item; the highest scoring binding wins,
        #    regardless of NLU slot label. This fixes cases like:
        #      "chicken taco with coke steak and chicken"
        #    where "coke" must bind to the Can Drinks side group even though
        #    NLU may emit it as ITEM, and "steak" must bind to Additional
        #    Meat without dragging "with"/"and" tokens into the score.
        prefill_result: PrefillResult = self.prefill_engine.prefill(
            pending=context.pending_add_item,
            segment_text=user_text,
            slots=tuple(slots or ()),
        )
        self._apply_prefill_result(context=context, result=prefill_result)

        # 2) Side sizes are dependent on which side choices were just bound,
        #    so resolve them after the unified pass.
        self._prefill_selected_side_variants(
            context=context,
            user_text=prefill_user_text,
            slots=slots,
        )

        # Build a spoken summary of everything that was pre-captured
        prefilled_summary = self._build_prefilled_summary(context)
        prefill_feedback = self._build_prefill_feedback_summary(
            context,
            list(prefill_result.feedback),
            unresolved_phrases=prefill_result.unresolved_phrases,
        )
        missing_groups_after_prefill = self._missing_group_names(context)
        prefill_debug = self._build_prefill_debug_payload(
            context=context,
            segment_text=user_text,
            missing_groups_before_prefill=missing_groups_before_prefill,
            missing_groups_after_prefill=missing_groups_after_prefill,
            engine_debug=prefill_result.debug,
        )
        logger.debug("pending_item_prefill %s", prefill_debug)

        step = determine_next_add_item_step(context)

        if step.next_state == ConversationState.FINALIZING_ADD_ITEM:
            payload = {
                "item_name": item.name,
                "quantity": context.quantity or 1,
                "prefilled_summary": prefilled_summary,
                "prefill_debug": prefill_debug,
            }
            if prefill_feedback:
                payload["prefill_feedback"] = prefill_feedback
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
                response_payload=payload,
                command=build_add_item_command(context),
                reset_context=True,
            )

        payload = dict(step.response_payload or {})
        if prefilled_summary:
            payload["prefilled_summary"] = prefilled_summary
            payload["prefilled_item_name"] = item.name
        if prefill_feedback:
            payload["prefill_feedback"] = prefill_feedback
        payload["prefill_debug"] = prefill_debug

        return HandlerResult(
            next_state=step.next_state,
            response_key=step.response_key,
            response_payload=payload,
        )

    @staticmethod
    def _build_prefilled_summary(context: ConversationContext) -> str:
        """
        Build a short spoken summary of what was auto-captured from the
        user's utterance, e.g. "with Coke, extra Cheese, Bacon".

        Returns empty string if nothing was prefilled.
        """
        pending = context.pending_add_item
        if pending is None:
            return ""

        parts: list[str] = []

        # Variant / size
        if context.selected_variant_id and pending.item_variants_by_id:
            variant = pending.item_variants_by_id.get(context.selected_variant_id)
            if variant:
                parts.append(variant.name)

        # Side items
        for group in pending.side_groups:
            selected_ids = context.selected_side_groups.get(group.group_id, [])
            for sid in selected_ids:
                choice = group.choices_by_item_id.get(sid)
                if choice:
                    # Include side size if also prefilled
                    side_variant_id = context.selected_side_variants.get(sid)
                    if side_variant_id and choice.variants_by_id:
                        sv = choice.variants_by_id.get(side_variant_id)
                        if sv:
                            parts.append(f"{sv.name} {choice.name}")
                            continue
                    parts.append(choice.name)

        # Modifiers
        for group in pending.modifier_groups:
            for sel in context.selected_modifier_groups.get(group.group_id, []):
                if sel.action == "remove":
                    parts.append(f"no {sel.name}")
                elif sel.instruction == "extra":
                    parts.append(f"extra {sel.name}")
                elif sel.instruction == "less":
                    parts.append(f"less {sel.name}")
                else:
                    parts.append(sel.name)

        if not parts:
            return ""

        if len(parts) == 1:
            return f"with {parts[0]}"
        if len(parts) == 2:
            return f"with {parts[0]} and {parts[1]}"
        return f"with {', '.join(parts[:-1])}, and {parts[-1]}"

    @staticmethod
    def _format_feedback_names(values: list[str]) -> str:
        clean = [str(value).strip() for value in values if str(value).strip()]
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} and {clean[1]}"
        return f"{', '.join(clean[:-1])}, and {clean[-1]}"

    def _build_prefill_feedback_summary(
        self,
        context: ConversationContext,
        feedback_entries: list[dict],
        *,
        unresolved_phrases: list[str] | None = None,
    ) -> str:
        parts: list[str] = []
        pending = context.pending_add_item
        item_name = pending.item_name if pending else ""
        # Scope the "I couldn't find X" feedback to the current item only
        # when the user is actually in a multi-item flow; otherwise the
        # extra scope reads as awkward redundancy. We treat a non-empty
        # pending_item_queue as the signal — that means OTHER items are
        # waiting to be processed.
        scope_to_item = bool(getattr(context, "pending_item_queue", None))

        for entry in feedback_entries:
            accepted_names = entry.get("accepted_names") or []
            requested_names = entry.get("requested_names") or []
            dropped_names = entry.get("dropped_names") or []
            max_selector = int(entry.get("max_selector", 0) or 0)

            if dropped_names:
                dropped_text = self._format_feedback_names(requested_names or dropped_names)
                accepted_text = self._format_feedback_names(accepted_names)
                if accepted_text and max_selector > 0:
                    parts.append(
                        f"I heard {accepted_text} and {dropped_text}, but you can only pick {max_selector} there."
                    )
                elif accepted_text:
                    parts.append(f"I heard {accepted_text} and {dropped_text}, so I'll ask you to choose.")
                elif max_selector > 0:
                    parts.append(f"I heard {dropped_text}, but you can only pick {max_selector} there.")
                else:
                    parts.append(f"I heard {dropped_text}, so I'll ask you to choose.")

        cleaned_unresolved = self._collapse_unresolved_for_feedback(
            unresolved_phrases or [],
            pending=pending,
        )
        if cleaned_unresolved:
            unresolved_text = self._format_feedback_names(cleaned_unresolved)
            if scope_to_item and item_name:
                parts.append(f"For the {item_name}, I couldn't find {unresolved_text}.")
            else:
                parts.append(f"I couldn't find {unresolved_text}.")

        return " ".join(parts).strip()

    def _collapse_prefill_unmatched_names(
        self,
        context: ConversationContext,
        feedback_entries: list[dict],
    ) -> list[str]:
        pending = context.pending_add_item
        if pending is None:
            return []

        ignored_tokens: set[str] = set(tokenize(normalize_text(pending.item_name)))
        ignored_tokens.update(
            {
                "with",
                "extra",
                "more",
                "double",
                "no",
                "without",
                "light",
                "less",
                "on",
                "the",
                "side",
                "and",
                "plus",
                "plu",
                "also",
                "als",
                "large",
                "medium",
                "small",
                "regular",
                "mini",
                "xl",
            }
        )
        known_choice_tokens: set[str] = set()
        known_phrases: set[str] = {normalize_text(pending.item_name)}
        matched_feedback_tokens: set[str] = set()
        raw_unmatched_values: list[str] = []

        for group in pending.side_groups:
            for choice in group.choices:
                known_choice_tokens.update(tokenize(choice.normalized_name))
                known_phrases.add(choice.normalized_name)

        for group in pending.modifier_groups:
            for choice in group.choices:
                known_choice_tokens.update(tokenize(choice.normalized_name))
                known_phrases.add(choice.normalized_name)

        for variant in pending.item_variants:
            known_choice_tokens.update(tokenize(variant.normalized_name))
            known_phrases.add(variant.normalized_name)

        for entry in feedback_entries:
            for accepted_name in entry.get("accepted_names") or []:
                matched_feedback_tokens.update(tokenize(normalize_text(accepted_name)))
            for dropped_name in entry.get("dropped_names") or []:
                matched_feedback_tokens.update(tokenize(normalize_text(dropped_name)))
            for unmatched_name in entry.get("unmatched_names") or []:
                if unmatched_name:
                    raw_unmatched_values.append(str(unmatched_name).strip())

        deduped_values = self._dedupe_keep_order(raw_unmatched_values)
        if not deduped_values:
            return []

        first_pass: list[str] = []
        known_tokens = ignored_tokens | known_choice_tokens | matched_feedback_tokens
        for value in deduped_values:
            normalized_value = normalize_text(value)
            value_tokens = set(tokenize(normalized_value))
            novel_tokens = value_tokens - known_tokens
            if novel_tokens or self._should_keep_raw_unmatched_phrase(
                normalized_value,
                known_phrases,
            ):
                first_pass.append(value)

        cleaned_values: list[str] = []
        for value in first_pass:
            normalized_value = normalize_text(value)
            value_tokens = set(tokenize(normalized_value)) - known_tokens
            if not value_tokens:
                if self._should_keep_raw_unmatched_phrase(normalized_value, known_phrases):
                    cleaned_values.append(value)
                continue

            covered_by_shorter = False
            for other in first_pass:
                if other == value or len(other) >= len(value):
                    continue
                other_tokens = set(tokenize(normalize_text(other))) - known_tokens
                if value_tokens and value_tokens.issubset(other_tokens):
                    covered_by_shorter = True
                    break

            if not covered_by_shorter:
                cleaned_values.append(value)

        final_values: list[str] = []
        canonical_seen: set[str] = set()
        for value in cleaned_values:
            canonical = self._canonicalize_unmatched_phrase(
                normalize_text(value),
                ignored_tokens=ignored_tokens,
                matched_feedback_tokens=matched_feedback_tokens,
            )
            if canonical and canonical in canonical_seen:
                continue
            if canonical:
                canonical_seen.add(canonical)
            final_values.append(value)

        return self._dedupe_keep_order(final_values)

    @staticmethod
    def _dedupe_keep_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            cleaned = str(value).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
        return result

    @staticmethod
    def _should_keep_raw_unmatched_phrase(
        normalized_value: str,
        known_phrases: set[str],
    ) -> bool:
        if not normalized_value or normalized_value in known_phrases:
            return False

        return normalized_value.startswith("no ") or normalized_value.startswith("without ")

    @staticmethod
    def _canonicalize_unmatched_phrase(
        normalized_value: str,
        *,
        ignored_tokens: set[str],
        matched_feedback_tokens: set[str],
    ) -> str:
        tokens = [
            token
            for token in tokenize(normalized_value)
            if token not in ignored_tokens and token not in matched_feedback_tokens
        ]
        return " ".join(tokens).strip()

    @staticmethod
    def _clean_prefill_unmatched_values(
        values: list[str],
        *,
        item_name: str,
    ) -> list[str]:
        normalized_item_name = normalize_text(item_name)
        return [
            value
            for value in values
            if value and normalize_text(value) != normalized_item_name
        ]

    @staticmethod
    def _prefill_ignored_modifier_values(context: ConversationContext) -> list[str]:
        pending = context.pending_add_item
        if pending is None:
            return []

        ignored: list[str] = []
        normalized_item_name = normalize_text(pending.item_name)
        if normalized_item_name:
            ignored.append(normalized_item_name)

        for group in pending.side_groups:
            for selected_item_id in context.selected_side_groups.get(group.group_id, []):
                choice = group.choices_by_item_id.get(selected_item_id)
                if choice and choice.normalized_name:
                    ignored.append(choice.normalized_name)
                    ignored.extend(getattr(choice, "match_texts", ()) or ())

        seen: set[str] = set()
        result: list[str] = []
        for value in ignored:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _build_prefill_debug_payload(
        self,
        *,
        context: ConversationContext,
        segment_text: str,
        missing_groups_before_prefill: list[str],
        missing_groups_after_prefill: list[str],
        engine_debug: dict | None = None,
    ) -> dict[str, object]:
        resolved_group_values = self._resolved_group_values(context)
        pending = context.pending_add_item
        candidate_text = segment_text
        if pending is not None:
            candidate_text = self._prefill_segment_text_for_item(
                item_name=pending.item_name,
                user_text=segment_text,
            )
        engine_debug = engine_debug or {}
        candidate_phrases = (
            engine_debug.get("candidate_phrases")
            or self._collect_prefill_candidate_phrases(
                context=context,
                segment_text=candidate_text,
            )
        )
        return {
            "segment_text": segment_text,
            "segment_text_after_item": engine_debug.get("segment_text_after_item", candidate_text),
            "candidate_phrases": list(candidate_phrases),
            "resolved_group_values": resolved_group_values,
            "pending_item_prefill_before_missing_groups": resolved_group_values,
            "missing_groups_before_prefill": missing_groups_before_prefill,
            "missing_groups_after_prefill": missing_groups_after_prefill,
            "skipped_groups_because_prefilled": [
                group_name
                for group_name in missing_groups_before_prefill
                if group_name not in missing_groups_after_prefill
            ],
            "bindings": engine_debug.get("bindings", []),
        }

    def _collect_prefill_candidate_phrases(
        self,
        *,
        context: ConversationContext,
        segment_text: str,
    ) -> list[str]:
        pending = context.pending_add_item
        if pending is None:
            return []

        candidates: list[str] = []
        seen: set[str] = set()

        def add(value: str | None) -> None:
            normalized = normalize_text(value or "")
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        for candidate in build_side_option_candidates(context, segment_text):
            add(candidate.text)
        for candidate in build_modifier_option_candidates(context, segment_text):
            add(candidate.text)

        known_phrases: list[str] = []
        for group in pending.side_groups:
            for choice in group.choices:
                known_phrases.extend(getattr(choice, "match_texts", ()) or (choice.normalized_name,))
        for group in pending.modifier_groups:
            for choice in group.choices:
                known_phrases.extend(getattr(choice, "match_texts", ()) or (choice.normalized_name,))
        known_phrases.extend(variant.normalized_name for variant in pending.item_variants)

        for candidate in build_scoped_phrase_candidates(
            raw_utterance=segment_text,
            phrases=known_phrases,
        ):
            add(candidate.text)

        return candidates

    @staticmethod
    def _resolved_group_values(context: ConversationContext) -> dict[str, list[str]]:
        pending = context.pending_add_item
        if pending is None:
            return {}

        resolved: dict[str, list[str]] = {}
        if context.selected_variant_id and context.selected_variant_id in pending.item_variants_by_id:
            resolved["Size"] = [pending.item_variants_by_id[context.selected_variant_id].name]

        for group in pending.side_groups:
            selected_ids = context.selected_side_groups.get(group.group_id, [])
            if not selected_ids:
                continue
            resolved[group.name] = [
                group.choices_by_item_id[item_id].name
                for item_id in selected_ids
                if item_id in group.choices_by_item_id
            ]

        for group in pending.modifier_groups:
            selections = context.selected_modifier_groups.get(group.group_id, [])
            if not selections:
                continue
            resolved[group.name] = [selection.name for selection in selections]

        if isinstance(context.quantity, int) and context.quantity > 0:
            resolved["Quantity"] = [str(context.quantity)]

        return resolved

    def _missing_group_names(self, context: ConversationContext) -> list[str]:
        pending = context.pending_add_item
        if pending is None:
            return []

        missing: list[str] = []
        if pending.item_variants and not context.selected_variant_id:
            missing.append("Size")

        for group in pending.side_groups:
            selected = context.selected_side_groups.get(group.group_id, ())
            skipped = group.group_id in context.skipped_side_groups
            min_selector, _ = effective_group_selector_bounds(group)
            if bool(getattr(group, "is_required", False)):
                if len(selected) < min_selector:
                    missing.append(group.name)
            elif not selected and not skipped:
                missing.append(group.name)

        for group in pending.modifier_groups:
            selections = context.selected_modifier_groups.get(group.group_id, ())
            skipped = group.group_id in context.skipped_modifier_groups
            min_selector, _ = effective_group_selector_bounds(group)
            if bool(getattr(group, "is_required", False)):
                if len(selections) < min_selector:
                    missing.append(group.name)
            elif not selections and not skipped:
                missing.append(group.name)

        if not isinstance(context.quantity, int) or context.quantity <= 0:
            missing.append("Quantity")

        return missing

    def _apply_prefill_result(
        self,
        *,
        context: ConversationContext,
        result: PrefillResult,
    ) -> None:
        """
        Apply the unified prefill engine result onto the context.

        This is the single point where engine output is committed to the
        FSM-visible state. Anything written here will be picked up by
        determine_next_add_item_step when computing the next missing group.
        """
        if result.variant_id:
            context.selected_variant_id = result.variant_id
            context.size_target = None

        for group_id, item_ids in result.side_selections.items():
            if not item_ids:
                continue
            context.selected_side_groups[group_id] = list(item_ids)
            context.skipped_side_groups.discard(group_id)

        for group_id, selections in result.modifier_selections.items():
            if not selections:
                continue
            context.selected_modifier_groups[group_id] = list(selections)
            context.skipped_modifier_groups.discard(group_id)

    @staticmethod
    def _collapse_unresolved_for_feedback(
        unresolved_phrases: list[str],
        *,
        pending,
    ) -> list[str]:
        """Filter unresolved phrases for a user-facing "couldn't find" message.

        Keeps the ORIGINAL phrase (e.g. "no sauce", "american cheese") so
        the user hears what they said back, but uses the residual non-
        ignored tokens as a *canonical* dedup key so composite phrases
        ("american cheese coke") don't repeat content already covered by a
        shorter phrase ("american cheese").
        """
        if not unresolved_phrases:
            return []

        ignored_tokens: set[str] = set()
        if pending is not None:
            ignored_tokens.update(tokenize(normalize_text(pending.item_name)))
            for group in pending.side_groups:
                for choice in group.choices:
                    ignored_tokens.update(tokenize(choice.normalized_name))
                    for label in (getattr(choice, "match_texts", ()) or ()):
                        ignored_tokens.update(tokenize(label))
            for group in pending.modifier_groups:
                for choice in group.choices:
                    ignored_tokens.update(tokenize(choice.normalized_name))
                    for label in (getattr(choice, "match_texts", ()) or ()):
                        ignored_tokens.update(tokenize(label))
            for variant in pending.item_variants:
                ignored_tokens.update(tokenize(variant.normalized_name))

        # Connector / instruction words that are never user-facing labels.
        ignored_tokens.update(
            {
                "with", "and", "plus", "also", "or",
                "extra", "more", "double", "less", "light",
                "on", "the", "side",
                "a", "an",
            }
        )
        # NOTE: "no"/"without"/"hold"/"remove" are intentionally NOT in
        # ignored_tokens so that "no sauce" survives as "no sauce" (not
        # canonicalised to "sauce") in the user-facing text. Their tokens
        # only filter the canonical dedup key below.
        canonical_ignored = ignored_tokens | {"no", "without", "hold", "remove"}

        result: list[str] = []
        seen_canonical: set[str] = set()
        seen_phrases: set[str] = set()
        for phrase in unresolved_phrases:
            normalized = normalize_text(phrase or "").strip()
            if not normalized or normalized in seen_phrases:
                continue
            canonical = " ".join(
                t for t in tokenize(normalized) if t not in canonical_ignored
            )
            if not canonical or canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            seen_phrases.add(normalized)
            result.append(normalized)
        return result

    def _prefill_side_groups(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
    ) -> list[dict]:
        return self.capture_helper.prefill_side_groups(
            context=context,
            normalized_user_text=normalized_user_text,
        )

    def _prefill_selected_side_variants(
        self,
        *,
        context: ConversationContext,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> None:
        self.capture_helper.prefill_selected_side_variants(
            context=context,
            user_text=user_text,
            slots=slots,
        )

    def _prefill_modifier_groups(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
    ) -> list[dict]:
        return self.capture_helper.prefill_modifier_groups(
            context=context,
            normalized_user_text=normalized_user_text,
        )

    @staticmethod
    def _all_modifier_choice_phrases(pending) -> list[str]:
        return PendingItemCaptureHelper.all_modifier_choice_phrases(pending)

    def _prefill_item_variant(
        self,
        *,
        context: ConversationContext,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> None:
        self.capture_helper.prefill_item_variant(
            context=context,
            user_text=user_text,
            slots=slots,
        )

    def _extract_requested_size(
        self,
        *,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> str | None:
        return self.capture_helper.extract_requested_size(
            user_text=user_text,
            slots=slots,
        )

    def _match_variant_label(
        self,
        *,
        requested_size: str,
        pending_variants,
    ) -> object | None:
        return self.capture_helper.match_variant_label(
            requested_size=requested_size,
            pending_variants=pending_variants,
        )

    def _get_last_slots(self, context: ConversationContext) -> Sequence[SlotValue]:
        return context.last_slots or ()

    def _prefill_segment_text_for_item(self, *, item_name: str, user_text: str) -> str:
        normalized = self._normalize_item_request_text(user_text)
        item_normalized = normalize_text(item_name or "")
        if not normalized or not item_normalized:
            return normalized
        if normalized.startswith(item_normalized):
            remainder = normalized[len(item_normalized):].strip()
            return remainder or normalized
        return normalized

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
