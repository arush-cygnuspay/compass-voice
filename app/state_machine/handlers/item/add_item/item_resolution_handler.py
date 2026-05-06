# app/state_machine/handlers/item/add_item/item_resolution_handler.py
"""Item resolution: maps a raw MenuQueryResult to a HandlerResult.

Owns the routing logic that was previously in AddItemHandler._route_menu_query_result
and the modifier-only-request guard (_looks_like_modifier_only_request).

On a successful single-item match it delegates to PrefillOrchestrator to
initialise the pending item, apply prefill, and determine the next FSM step.

Architecture note
-----------------
``resolve_item_and_enter_flow`` is now a thin two-step adapter:
  1. Calls ``OrderingDecisionEngine.decide_from_menu_result`` — pure, no
     side effects, returns an ``OrderingDecision``.
  2. Calls ``_execute_decision`` — effectful: mutates ``ConversationContext``,
     constructs ``HandlerResult``, calls ``PrefillOrchestrator`` for ADD_ITEM.

The original ``MenuQueryResult`` is forwarded to ``_execute_decision`` so that
``MenuItem`` objects (which carry side/modifier groups) can be used directly
without an extra repo round-trip.  ``OrderingDecision`` intentionally keeps
only primitive fields (item_id, item_name) per the DTO contract.

External behaviour is unchanged — callers see the same ``HandlerResult``
shapes as before.
"""
from __future__ import annotations

from typing import Sequence

from app.contracts.ordering_decision import OrderingAction, OrderingDecision
from app.core.ordering_decision_engine import OrderingDecisionEngine
from app.menu.models import MenuItem
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
        *,
        engine: OrderingDecisionEngine | None = None,
    ) -> None:
        self.menu_repo = menu_repo
        self.prefill_orchestrator = prefill_orchestrator
        self._engine = engine or OrderingDecisionEngine(menu_repo)

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

        Delegates the pure routing decision to ``OrderingDecisionEngine``,
        then executes it (context mutations + HandlerResult construction).
        """
        decision = self._engine.decide_from_menu_result(
            result=result,
            requested_item_text=requested_item_text,
            context=context,
            slots=slots,
        )
        return self._execute_decision(
            decision,
            result=result,
            context=context,
            original_user_text=original_user_text,
            slots=slots,
        )

    # ------------------------------------------------------------------
    # Decision executor (effectful — mutates context)
    # ------------------------------------------------------------------

    def _execute_decision(
        self,
        decision: OrderingDecision,
        *,
        result: MenuQueryResult,
        context: ConversationContext,
        original_user_text: str,
        slots: Sequence[SlotValue],
    ) -> HandlerResult:
        """Translate an ``OrderingDecision`` into a ``HandlerResult``.

        The original ``MenuQueryResult`` is available here so the executor
        can resolve ``MenuItem`` objects for the ADD_ITEM path without an
        extra repo lookup.  All ``ConversationContext`` mutations live here.
        """
        action = decision.action

        # ── Exact match / single-item category ───────────────────────────
        if action == OrderingAction.ADD_ITEM:
            item = self._item_from_result(result)
            if item is None:
                return HandlerResult(
                    next_state=ConversationState.ERROR_RECOVERY,
                    response_key="item_context_missing",
                )
            return self.prefill_orchestrator.enter_add_flow_for_item(
                context=context,
                item=item,
                user_text=original_user_text,
                slots=slots,
            )

        # ── Category disambiguation ───────────────────────────────────────
        if action == OrderingAction.ASK_ITEM_CONFIRMATION and decision.reason == "category_detected":
            candidate_ids = [c.item_id for c in decision.candidates]
            candidate_names = [c.item_name for c in decision.candidates]
            payload = {
                "reason": "category_detected",
                "query": decision.query,
                "category_id": decision.category_id,
                "category_name": decision.category_name,
                "candidate_item_ids": candidate_ids,
                "candidate_item_names": candidate_names,
            }
            context.awaiting_confirmation_for = {
                "type": "item",
                "reason": "category_detected",
                "query": decision.query,
                "category_id": decision.category_id,
                "category_name": decision.category_name,
                "candidate_item_ids": candidate_ids,
                "candidate_item_names": candidate_names,
            }
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key="confirm_item_from_category",
                response_payload=payload,
            )

        # ── Ambiguous (multiple items / categories) ───────────────────────
        if action == OrderingAction.ASK_ITEM_CONFIRMATION and decision.reason == "multiple_matches":
            candidate_ids = [c.item_id for c in decision.candidates]
            candidate_names = [c.item_name for c in decision.candidates]
            payload: dict = {
                "reason": "multiple_matches",
                "query": decision.query,
            }
            if candidate_ids:
                payload["candidate_item_ids"] = candidate_ids
                payload["candidate_item_names"] = candidate_names
            if decision.matched_category_names:
                payload["candidate_category_names"] = list(decision.matched_category_names)
            context.awaiting_confirmation_for = {
                "type": "item",
                "reason": "multiple_matches",
                "query": decision.query,
                "candidate_item_ids": candidate_ids,
                "candidate_item_names": candidate_names,
                "candidate_category_names": list(decision.matched_category_names),
            }
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key="confirm_item_ambiguous",
                response_payload=payload,
            )

        # ── NOT_FOUND paths: bump counter (engine was pure — no side effects) ──
        normalized_query = normalize_text(decision.query or "")
        if normalized_query:
            context.bump_not_found(normalized_query)

        # ── Escalation ────────────────────────────────────────────────────
        if action == OrderingAction.ESCALATE_TO_AGENT:
            context.reset_item_scope()
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_not_found_escalation",
            )

        # ── Near-miss confirmation prompt ─────────────────────────────────
        if action == OrderingAction.ASK_ITEM_CONFIRMATION and decision.reason == "near_miss_suggestion":
            context.awaiting_confirmation_for = {
                "type": "item",
                "reason": "near_miss_suggestion",
                "value_id": decision.item_id,
                "value_name": decision.item_name,
                "query": decision.query,
                "suggestions": list(decision.suggestions),
                "tier": decision.tier,
            }
            return HandlerResult(
                next_state=ConversationState.CONFIRMING_ITEM,
                response_key="item_not_found_near_miss",
                response_payload={
                    "item_name": decision.item_name,
                    "tier": decision.tier,
                },
            )

        # ── Low-confidence / no match: suggest alternatives ───────────────
        context.reset_item_scope()
        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_not_found",
            response_payload={
                "query": decision.query,
                "suggestions": list(decision.suggestions),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _item_from_result(result: MenuQueryResult) -> MenuItem | None:
        """Extract the resolved MenuItem from an ADD_ITEM-path result."""
        if result.item is not None:
            return result.item
        if result.items and len(result.items) >= 1:
            return result.items[0]
        return None

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
