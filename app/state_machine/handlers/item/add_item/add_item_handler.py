# app/state_machine/handlers/item/add_item/add_item_handler.py
from __future__ import annotations

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
    extract_side_slot_values_normalized,
)
from app.state_machine.handlers.item.add_item.modifier_group_resolver import (
    ModifierGroupResolver,
    extract_modifier_slot_values_normalized,
)
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
    def _build_segment_summary(seg: ParsedItemSegment) -> str:
        """Build a spoken summary like '2 chicken tacos with coke and bacon'."""
        qty_prefix = f"{seg.quantity} " if seg.quantity and seg.quantity > 1 else ""
        item_name = seg.item_slot_value or ""

        # Extract detail tokens from raw_text that are NOT the item name
        # e.g. "chicken taco with coke and extra american cheese" → "with coke and extra american cheese"
        raw = (seg.raw_text or "").strip()
        detail_suffix = ""
        if item_name and raw:
            norm_item = normalize_text(item_name)
            norm_raw = normalize_text(raw)
            # Strip quantity prefix from raw for matching
            stripped = re.sub(r"^\d+\s+", "", norm_raw).strip()
            idx = stripped.find(norm_item)
            if idx >= 0:
                after = stripped[idx + len(norm_item):].strip()
                if after:
                    detail_suffix = f" {after}"

        return f"{qty_prefix}{item_name}{detail_suffix}".strip() or raw

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
        side_feedback = self._prefill_side_groups(
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
        modifier_feedback = self._prefill_modifier_groups(
            context=context,
            normalized_user_text=user_text,
        )

        # Build a spoken summary of everything that was pre-captured
        prefilled_summary = self._build_prefilled_summary(context)
        prefill_feedback = self._build_prefill_feedback_summary(
            context,
            side_feedback + modifier_feedback,
        )

        step = determine_next_add_item_step(context)

        if step.next_state == ConversationState.FINALIZING_ADD_ITEM:
            payload = {
                "item_name": item.name,
                "quantity": context.quantity or 1,
                "prefilled_summary": prefilled_summary,
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
    ) -> str:
        parts: list[str] = []

        for entry in feedback_entries:
            accepted_names = entry.get("accepted_names") or []
            dropped_names = entry.get("dropped_names") or []
            max_selector = int(entry.get("max_selector", 0) or 0)

            if dropped_names:
                dropped_text = self._format_feedback_names(dropped_names)
                accepted_text = self._format_feedback_names(accepted_names)
                if accepted_text and max_selector > 0:
                    parts.append(
                        f"I kept {accepted_text} and left off {dropped_text} because you can only pick {max_selector}."
                    )
                elif accepted_text:
                    parts.append(f"I kept {accepted_text} and left off {dropped_text}.")
                elif max_selector > 0:
                    parts.append(f"I left off {dropped_text} because you can only pick {max_selector}.")
                else:
                    parts.append(f"I left off {dropped_text}.")

        unmatched_names = self._collapse_prefill_unmatched_names(context, feedback_entries)
        if unmatched_names:
            parts.append(
                f"I couldn't find {self._format_feedback_names(unmatched_names)}."
            )

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

        seen: set[str] = set()
        result: list[str] = []
        for value in ignored:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _prefill_side_groups(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
    ) -> list[dict]:
        pending = context.pending_add_item
        if pending is None or not pending.side_groups:
            return []

        slot_values = extract_side_slot_values_normalized(context)
        feedback: list[dict] = []

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

            if capped_ids:
                context.selected_side_groups[group.group_id] = capped_ids
                context.skipped_side_groups.discard(group.group_id)
            elif unmatched_values and not getattr(group, "is_required", False):
                context.skipped_side_groups.add(group.group_id)

            if dropped_names or unmatched_values:
                feedback.append(
                    {
                        "accepted_names": accepted_names,
                        "dropped_names": dropped_names,
                        "unmatched_names": unmatched_values,
                        "max_selector": max_selector,
                        "min_selector": min_selector,
                    }
                )

        return feedback

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
    ) -> list[dict]:
        pending = context.pending_add_item
        if pending is None or not pending.modifier_groups:
            return []

        slot_values = extract_modifier_slot_values_normalized(context)
        ignored_values = self._prefill_ignored_modifier_values(context)
        feedback: list[dict] = []

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
                ignored_values=ignored_values,
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

            if capped:
                context.selected_modifier_groups[group.group_id] = capped
                context.skipped_modifier_groups.discard(group.group_id)
            elif unmatched_values and not getattr(group, "is_required", False):
                context.skipped_modifier_groups.add(group.group_id)

            if dropped or unmatched_values:
                feedback.append(
                    {
                        "accepted_names": [sel.name for sel in capped],
                        "dropped_names": [sel.name for sel in dropped],
                        "unmatched_names": unmatched_values,
                        "max_selector": max_selector,
                        "min_selector": min_selector,
                    }
                )

        return feedback

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
