# app/state_machine/handlers/item/add_item/item_resolution_handler.py
"""Item resolution: maps a raw MenuQueryResult to a HandlerResult.

Owns the routing logic that was previously in AddItemHandler._route_menu_query_result
and the modifier-only-request guard (_looks_like_modifier_only_request).

On a successful single-item match it delegates to PrefillOrchestrator to
initialise the pending item, apply prefill, and determine the next FSM step.
"""
from __future__ import annotations

from typing import Sequence

from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.repository import MenuRepository
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.prefill_orchestrator import PrefillOrchestrator
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class ItemResolutionHandler:
    """Routes a MenuQueryResult to the appropriate HandlerResult.

    Called by AddItemHandler (single-item path) and MultiItemQueueCoordinator
    (first-item path of a multi-item utterance) after the menu query has
    already been executed by the caller.
    """

    def __init__(
        self,
        menu_repo: MenuRepository,
        prefill_orchestrator: PrefillOrchestrator,
    ) -> None:
        self.menu_repo = menu_repo
        self.prefill_orchestrator = prefill_orchestrator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_item_and_enter_flow(
        self,
        *,
        context: ConversationContext,
        result: MenuQueryResult,
        requested_item_text: str,
        original_user_text: str,
        slots: Sequence[SlotValue],
    ) -> HandlerResult:
        """Route a menu query result to the correct HandlerResult.

        Unambiguous single-item matches proceed immediately to
        PrefillOrchestrator.enter_add_flow_for_item().  Everything else
        (category, ambiguous, not-found) returns an appropriate prompt or
        error HandlerResult.
        """
        rtype = result.type

        if rtype == MenuQueryType.ITEM and result.item is not None:
            return self.prefill_orchestrator.enter_add_flow_for_item(
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
            return self.prefill_orchestrator.enter_add_flow_for_item(
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

        context.reset_item_scope()
        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_not_found",
            response_payload={
                "query": requested_item_text,
                "suggested_item_names": [
                    item.name for item in (result.suggested_items or [])
                ],
                "suggested_category_names": [
                    category.get("name")
                    for category in (result.suggested_categories or [])
                    if category.get("name")
                ],
            },
        )

    # ------------------------------------------------------------------
    # Static guard (used by AddItemHandler before the menu query)
    # ------------------------------------------------------------------

    @staticmethod
    def looks_like_modifier_only_request(
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
