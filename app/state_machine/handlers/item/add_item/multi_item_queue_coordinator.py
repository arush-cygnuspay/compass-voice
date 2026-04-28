# app/state_machine/handlers/item/add_item/multi_item_queue_coordinator.py
"""Multi-item queue coordinator.

Handles utterances that contain multiple item requests ("a burger and a coke
and two fries").  Queues items after the first one, then runs the first item
through ItemResolutionHandler, and wraps the result with a multi-item
acknowledgement payload.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Sequence

from app.core.pending_action import PendingAction
from app.menu.repository import MenuRepository
from app.menu.slot_helpers import first_slot_value
from app.nlu.multi_item_parser import ParsedItemSegment
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.item_resolution_handler import (
    ItemResolutionHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import QueuedItemRequest


class MultiItemQueueCoordinator:
    """Owns the multi-item split, queue setup, and first-item kickoff.

    Strategy:
    1. Queue all items after the first one (preserving their segment slots).
    2. Process the first item through ItemResolutionHandler.
    3. Wrap the result with a multi-item acknowledgement payload.
    """

    def __init__(
        self,
        menu_repo: MenuRepository,
        item_resolution_handler: ItemResolutionHandler,
    ) -> None:
        self.menu_repo = menu_repo
        self.item_resolution_handler = item_resolution_handler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(
        self,
        *,
        context: ConversationContext,
        segments: list[ParsedItemSegment],
        get_last_slots: Callable[[ConversationContext], Sequence[SlotValue]],
    ) -> HandlerResult:
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
        item_summaries = [self._build_segment_summary(seg) for seg in segments]

        # Set up context for the first item
        first_slots = first_segment.slots if first_segment.slots else get_last_slots(context)

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

        handler_result = self.item_resolution_handler.resolve_item_and_enter_flow(
            context=context,
            result=result,
            requested_item_text=requested_item_text,
            original_user_text=first_text,
            slots=first_slots,
        )

        # Wrap the response with a multi-item acknowledgement prefix
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

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_segment_summary(seg: ParsedItemSegment) -> str:
        """Build a concise spoken summary like '2 chicken tacos'."""
        qty_prefix = f"{seg.quantity} " if seg.quantity and seg.quantity > 1 else ""
        item_name = seg.item_slot_value or ""
        raw = (seg.raw_text or "").strip()
        return f"{qty_prefix}{item_name}".strip() or raw
