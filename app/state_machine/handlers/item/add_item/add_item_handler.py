# app/state_machine/handlers/item/add_item/add_item_handler.py
"""Thin coordinator for the add-item FSM path.

Responsibilities:
  - Normalise the raw utterance.
  - Detect multi-item utterances and delegate to MultiItemQueueCoordinator.
  - Run the single-item path: context reset, modifier-only guard, menu query,
    then delegate to ItemResolutionHandler.

All item-resolution, prefill, confirmation-decision, and queue-management
logic live in the dedicated modules imported below.
"""
from __future__ import annotations

import logging
from typing import Sequence

from app.core.pending_action import PendingAction
from app.menu.slot_helpers import first_slot_value
from app.nlu.intent_resolution.intent import Intent
from app.nlu.multi_item_parser import parse_multi_item_utterance
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.handlers.item.add_item.confirmation_decision_helper import (
    ConfirmationDecisionHelper,
)
from app.state_machine.handlers.item.add_item.item_resolution_handler import (
    ItemResolutionHandler,
)
from app.state_machine.handlers.item.add_item.modifier_group_resolver import ModifierGroupResolver
from app.state_machine.handlers.item.add_item.multi_group_prefill import MultiGroupPrefillEngine
from app.state_machine.handlers.item.add_item.multi_item_queue_coordinator import (
    MultiItemQueueCoordinator,
)
from app.state_machine.handlers.item.add_item.prefill_orchestrator import (
    PendingItemCaptureHelper,
    PrefillOrchestrator,
    normalize_item_request_text,
)
from app.state_machine.handlers.item.add_item.side_group_resolver import SideGroupResolver
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.menu.repository import MenuRepository

logger = logging.getLogger(__name__)

# Re-export so that waiting-state handlers that do
#   from app.state_machine.handlers.item.add_item.add_item_handler import PendingItemCaptureHelper
# continue to work without modification.
__all__ = ["AddItemHandler", "PendingItemCaptureHelper"]


class AddItemHandler(BaseHandler):
    """Thin coordinator: normalise → detect multi-item → resolve → prefill → decide."""

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

        side_resolver = SideGroupResolver()
        modifier_resolver = ModifierGroupResolver()
        capture_helper = PendingItemCaptureHelper(
            side_resolver=side_resolver,
            modifier_resolver=modifier_resolver,
        )
        prefill_engine = MultiGroupPrefillEngine()
        confirmation_helper = ConfirmationDecisionHelper()

        self.prefill_orchestrator = PrefillOrchestrator(
            capture_helper=capture_helper,
            prefill_engine=prefill_engine,
            confirmation_helper=confirmation_helper,
        )
        self.item_resolution_handler = ItemResolutionHandler(
            menu_repo=menu_repo,
            prefill_orchestrator=self.prefill_orchestrator,
        )
        self.multi_item_coordinator = MultiItemQueueCoordinator(
            menu_repo=menu_repo,
            item_resolution_handler=self.item_resolution_handler,
        )

    # ------------------------------------------------------------------
    # BaseHandler entry point
    # ------------------------------------------------------------------

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
        normalized_user_text = normalize_item_request_text(user_text)
        slots = self._get_last_slots(context)

        multi_segments = parse_multi_item_utterance(
            slot_aligned_user_text,
            slots,
            menu_store=getattr(self.menu_repo, "store", None),
        )
        if len(multi_segments) >= 2:
            return self.multi_item_coordinator.handle(
                context=context,
                segments=multi_segments,
                get_last_slots=self._get_last_slots,
            )

        return self._handle_single_item(
            context=context,
            normalized_user_text=normalized_user_text,
            slots=slots,
        )

    # ------------------------------------------------------------------
    # Single-item path
    # ------------------------------------------------------------------

    def _handle_single_item(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
        slots: tuple[SlotValue, ...] | list[SlotValue],
    ) -> HandlerResult:
        item_slot_value = first_slot_value(slots, "ITEM", "MENU_ITEM")
        category_slot_value = first_slot_value(slots, "CATEGORY", "MENU_CATEGORY")
        modifier_slot_value = first_slot_value(slots, "MODIFIER")

        context.reset_task()
        context.pending_action = PendingAction.ADD_ITEM

        if (
            not item_slot_value
            and not category_slot_value
            and modifier_slot_value
            and ItemResolutionHandler.looks_like_modifier_only_request(
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
            return self.item_resolution_handler.resolve_item_and_enter_flow(
                context=context,
                result=result,
                requested_item_text=normalize_text(requested_item_text),
                original_user_text=normalized_user_text,
                slots=slots,
            )

        result = self.menu_repo.resolve_menu_query_normalized(normalized_user_text, limit=5)
        return self.item_resolution_handler.resolve_item_and_enter_flow(
            context=context,
            result=result,
            requested_item_text=normalized_user_text,
            original_user_text=normalized_user_text,
            slots=slots,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def capture_helper(self) -> PendingItemCaptureHelper:
        """Expose the capture helper for tests that probe prefill internals."""
        return self.prefill_orchestrator.capture_helper

    @staticmethod
    def _get_last_slots(context: ConversationContext) -> Sequence[SlotValue]:
        return context.last_slots or ()
