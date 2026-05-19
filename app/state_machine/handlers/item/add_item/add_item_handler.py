# app/state_machine/handlers/item/add_item/add_item_handler.py
"""Thin coordinator for the add-item FSM path.

Responsibilities:
  - Normalise the raw utterance.
  - Route complex compound turns through the GPT / local multi-item planners.
  - Guard unsafe NLU slots before the legacy slot parser executes them.
  - Run the single-item path: context reset, modifier-only guard, menu query,
    then delegate to ItemResolutionHandler.

Execution order for add_item turns
------------------------------------
1. SmartTurnPlanner (GPT) — if enabled and triggered.
     Fix: now handles len(items) >= 2 via apply_multi_item_plan().
2. GPT Add-Item Planner (Phase 4) — if enabled and triggered.
     Fix: no longer collapses multi-item plans to items[0].
3. Local heuristic multi-item planner — plan_multi_item_order().
     Handles typos ("cicken"), size-word boundaries, article boundaries.
4. Unsafe-slot guard — slot_pairing_looks_broken().
     Records whether slots are broken but does NOT return immediately.
     Prevents the legacy slot parser from executing bad slots.
5. Legacy slot-based multi-item parser — parse_multi_item_utterance().
     Only reached when slots are SAFE (broken_reason is None).
6. Compound fallback policy — decide_compound_fallback().
     Only evaluated when slots ARE broken.  Decides whether to:
       - Fall through to single-item path (item+modifier/side utterances,
         recoverable slot issues like low-confidence)
       - Return "what's the first item?" (ambiguous compound)
       - Return "one at a time" (repeated failure escalation)
7. Single-item path.
     Reached when slots are safe AND legacy parser found < 2 segments,
     OR when slots are broken AND compound policy chose EXECUTE_VALID_PLAN
     or EXECUTE_PARTIAL_AND_CLARIFY (e.g. "burger with fries").

Every branch emits an `add_item_handler_path_taken` log event so the
execution path is always visible in structured logs.

All item-resolution, prefill, confirmation-decision, and queue-management
logic live in the dedicated modules imported below.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from app.core.pending_action import PendingAction
from app.menu.slot_helpers import first_slot_value
from app.nlu.intent_resolution.intent import Intent
from app.nlu.multi_item_parser import ParsedItemSegment, parse_multi_item_utterance
from app.services.compound_turn_policy import CompoundFallbackDecision, decide_compound_fallback
from app.services.multi_item_order_planner import plan_multi_item_order
from app.services.slot_safety_guard import slot_pairing_looks_broken
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
from app.state_machine.handlers.item.add_item.multi_item_plan_executor import apply_multi_item_plan
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

# Path names for add_item_handler_path_taken log events.
_PATH_SMART_PLANNER = "smart_planner"
_PATH_GPT_ADD_ITEM_PLANNER = "gpt_add_item_planner"
_PATH_LOCAL_MULTI_ITEM_PLANNER = "local_multi_item_planner"
_PATH_LEGACY_SLOT_PARSER = "legacy_slot_parser"
_PATH_FALLBACK_CLARIFY = "fallback_clarify"
_PATH_SINGLE_ITEM = "single_item"


class AddItemHandler(BaseHandler):
    """Thin coordinator: normalise → plan → guard → resolve → prefill → decide."""

    def __init__(
        self,
        menu_repo: MenuRepository,
        *,
        gpt_planner: "Any | None" = None,
    ) -> None:
        self.menu_repo = menu_repo
        # Phase 4: GPT Add-Item Planner (shadow or inline, default None=disabled).
        # Injected at construction time — no import at module level to avoid
        # circular imports.  When None, the planner path is completely skipped.
        self._gpt_planner = gpt_planner

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
        menu_store = getattr(self.menu_repo, "store", None)
        local_confidence = float(getattr(context, "last_intent_confidence", 1.0) or 1.0)
        session_id = getattr(session, "session_id", None) if session else None

        # ── Step 1: SmartTurnPlanner (GPT, compound / correction) ─────────
        # Enabled via SMART_TURN_PLANNER_ENABLED=true.
        # FIX: _apply_smart_plan_add_item now routes len(items) >= 2 through
        # apply_multi_item_plan() instead of discarding items[1..N].
        _stp_result = self._try_smart_planner_add_item(
            user_text=slot_aligned_user_text,
            context=context,
            session=session,
        )
        if _stp_result is not None:
            self._log_path_taken(_PATH_SMART_PLANNER, "smart_planner_applied")
            return _stp_result
        # ─────────────────────────────────────────────────────────────────

        # ── Step 2: GPT Add-Item Planner (Phase 4) ────────────────────────
        # Enabled via COMPASS_GPT_ADD_ITEM_PLANNER_MODE=inline.
        # FIX: _apply_planner_result now routes len(validated_items) >= 2 through
        # apply_multi_item_plan() instead of returning None.
        planner_result = self._try_gpt_planner(
            user_text=slot_aligned_user_text,
            slots=list(slots),
            session=session,
        )
        if planner_result is not None and getattr(planner_result, "safe_to_apply", False):
            applied = self._apply_planner_result(planner_result, context)
            if applied is not None:
                self._log_planner_apply_outcome(
                    planner_result=planner_result,
                    applied=True,
                    block_reason=None,
                )
                self._log_path_taken(_PATH_GPT_ADD_ITEM_PLANNER, "planner_applied")
                return applied
            validated_plan = getattr(planner_result, "validated_plan", None)
            validated_items = getattr(validated_plan, "items", ()) or ()
            block_reason = "apply_helper_returned_none"
            self._log_planner_apply_outcome(
                planner_result=planner_result,
                applied=False,
                block_reason=block_reason,
            )
        # ────────────────────────────────────────────────────────────────────

        # ── Step 3: Local heuristic multi-item planner ────────────────────
        # Runs before the legacy slot parser.  Handles typos ("cicken"),
        # size-word boundaries, article boundaries, and variant descriptors.
        # Enabled unconditionally — falls through when < 2 items resolved.
        _planner_plan = plan_multi_item_order(
            slot_aligned_user_text,
            menu_store,
            state=getattr(context, "state", None),
            session_id=session_id,
        )
        if _planner_plan.is_compound and len(_planner_plan.items) >= 2:
            planner_segments: list[ParsedItemSegment] = [
                ParsedItemSegment(
                    raw_text=it.raw_span,
                    item_slot_value=it.item_name if it.item_name else None,
                    quantity=it.quantity if it.quantity > 1 else None,
                    slots=(),
                )
                for it in _planner_plan.items
            ]
            logger.info(
                "local_multi_item_planner.applied",
                extra={
                    "local_multi_item_planner_invoked": True,
                    "local_multi_item_planner_items": len(_planner_plan.items),
                    "local_multi_item_planner_unresolved": len(_planner_plan.unresolved_spans),
                    "final_slot_source": "local_multi_item_planner",
                    "final_slot_source_reason": _planner_plan.reason,
                },
            )
            self._log_path_taken(_PATH_LOCAL_MULTI_ITEM_PLANNER, _planner_plan.reason)
            return self.multi_item_coordinator.handle(
                context=context,
                segments=planner_segments,
                get_last_slots=self._get_last_slots,
            )
        else:
            logger.debug(
                "local_multi_item_planner.skipped",
                extra={
                    "local_multi_item_planner_invoked": True,
                    "local_multi_item_planner_items": len(_planner_plan.items),
                    "reason": _planner_plan.reason,
                },
            )
        # ─────────────────────────────────────────────────────────────────────

        # ── Step 4: Unsafe-slot guard ─────────────────────────────────────
        # Detects broken NLU slots that would cause wrong cart mutations if
        # fed to the legacy multi-item parser.  The guard does NOT return
        # immediately — it records the reason and lets the compound fallback
        # policy (step 6) decide whether to fall through to single-item
        # (e.g. "chicken sandwich with small coke") or show a prompt.
        broken_reason = slot_pairing_looks_broken(
            slots,
            slot_aligned_user_text,
            menu_store,
            local_confidence=local_confidence,
        )
        if broken_reason:
            logger.warning(
                "unsafe_slots_detected",
                extra={
                    "local_slots_flagged_as_unsafe": True,
                    "broken_reason": broken_reason,
                },
            )
        # ─────────────────────────────────────────────────────────────────

        # ── Step 5: Legacy slot-based multi-item parser ───────────────────
        # Only reached when slots appear SAFE (broken_reason is None).
        # The guard keeps broken slots away from parse_multi_item_utterance.
        if not broken_reason:
            multi_segments = parse_multi_item_utterance(
                slot_aligned_user_text,
                slots,
                menu_store=menu_store,
            )
            if len(multi_segments) >= 2:
                logger.debug(
                    "legacy_slot_parser.multi_item",
                    extra={
                        "final_slot_source": "legacy_slot_parser",
                        "segment_count": len(multi_segments),
                    },
                )
                self._log_path_taken(_PATH_LEGACY_SLOT_PARSER, "multi_segment")
                return self.multi_item_coordinator.handle(
                    context=context,
                    segments=multi_segments,
                    get_last_slots=self._get_last_slots,
                )
        # ─────────────────────────────────────────────────────────────────

        # ── Step 6: Compound fallback policy ─────────────────────────────
        # Only consulted when slots are broken.  Decides between:
        #   EXECUTE_VALID_PLAN / EXECUTE_PARTIAL_AND_CLARIFY
        #     → fall through to single-item (valid compound/modifier turns)
        #   FALLBACK_REPEAT_FIRST_ITEM
        #     → "couldn't separate clearly — what's the first item?"
        #   FALLBACK_ONE_AT_A_TIME
        #     → escalated "one at a time" after repeated failures
        if broken_reason:
            _planner_items = getattr(_planner_plan, "items", ()) or ()
            _planner_unresolved = getattr(_planner_plan, "unresolved_spans", ()) or ()
            _reprompt = int(
                (getattr(context, "reprompt_attempts", None) or {}).get(
                    "add_item_compound", 0
                )
            )
            compound_decision = decide_compound_fallback(
                transcript=slot_aligned_user_text,
                planner_result=None,  # GPT planner result handled upstream in step 2
                local_planner_result=_planner_plan,
                unsafe_slot_reason=broken_reason,
                valid_candidates_count=len(_planner_items),
                unresolved_spans=_planner_unresolved,
                reprompt_count=_reprompt,
            )
            logger.info(
                "compound_fallback_policy_decision",
                extra={
                    "compound_fallback_policy_action": compound_decision.value,
                    "unsafe_slot_reason": broken_reason,
                    "valid_candidates_count": len(_planner_items),
                    "unresolved_spans_count": len(_planner_unresolved),
                    "one_at_a_time_fallback_used": (
                        compound_decision == CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME
                    ),
                    "planner_attempted_before_fallback": True,
                    "final_compound_decision": compound_decision.value,
                },
            )

            if compound_decision == CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME:
                # Increment reprompt counter so next turn stays in this bucket
                _attempts = getattr(context, "reprompt_attempts", None)
                if isinstance(_attempts, dict):
                    _attempts["add_item_compound"] = _reprompt + 1
                self._log_path_taken(_PATH_FALLBACK_CLARIFY, "one_at_a_time")
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="multi_item_split_clarify",
                )

            if compound_decision == CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM:
                # Increment reprompt counter for potential future escalation
                _attempts = getattr(context, "reprompt_attempts", None)
                if isinstance(_attempts, dict):
                    _attempts["add_item_compound"] = _reprompt + 1
                self._log_path_taken(_PATH_FALLBACK_CLARIFY, "repeat_first_item")
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="compound_unclear_ask_first",
                )

            # EXECUTE_VALID_PLAN / EXECUTE_PARTIAL_AND_CLARIFY / ASK_ABOUT_UNRESOLVED:
            # Fall through to single-item path — the item+option structure is valid.
            logger.info(
                "compound_policy_fallthrough",
                extra={
                    "compound_fallback_policy_action": compound_decision.value,
                    "broken_reason_bypassed": broken_reason,
                    "final_slot_source": "compound_policy_single_item",
                },
            )
            self._log_path_taken(
                _PATH_SINGLE_ITEM,
                f"compound_policy:{compound_decision.value}",
            )
        # ─────────────────────────────────────────────────────────────────

        # ── Step 7: Single-item path ──────────────────────────────────────
        # Reached when:
        #   - Slots safe AND legacy multi-item parser found < 2 segments, OR
        #   - Slots broken AND compound policy chose to fall through
        #     (e.g. "burger with fries", low-confidence single-item, etc.)
        if not broken_reason:
            logger.debug(
                "single_item_path",
                extra={
                    "final_slot_source": "local_simple",
                    "confidence": local_confidence,
                },
            )
            self._log_path_taken(_PATH_SINGLE_ITEM, "single_item_path")
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

        context.reset_item_scope()
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

    def _log_path_taken(self, path: str, reason: str = "") -> None:
        """Emit add_item_handler_path_taken structured log event."""
        logger.info(
            "add_item_handler_path_taken",
            extra={
                "add_item_handler_path_taken": path,
                "add_item_handler_path_reason": reason,
            },
        )

    # ------------------------------------------------------------------
    # Phase 4 helpers
    # ------------------------------------------------------------------

    def _try_gpt_planner(
        self,
        *,
        user_text: str,
        slots: list[Any],
        session: Any,
    ) -> Any:
        """Run the GPT add-item planner and return a result, or None if disabled.

        Never raises — all exceptions return None so local flow is unaffected.

        Logs add_item_planner_skipped with reason when not invoked, so that
        the decision is always visible in structured logs.
        """
        planner = self._gpt_planner
        if planner is None:
            logger.debug(
                "add_item_planner_skipped",
                extra={
                    "add_item_planner_skipped": True,
                    "add_item_planner_skipped_reason": "planner_not_injected",
                },
            )
            return None
        try:
            local_slots_dicts = [
                {"n": getattr(sv, "name", ""), "v": str(getattr(sv, "value", ""))}
                for sv in slots
            ]
            # Cart names (compact — no prices)
            cart_item_names: list[str] = []
            cart = getattr(session, "cart", None) if session else None
            if cart is not None:
                try:
                    for ci in cart.get_items():
                        n = getattr(ci, "name", None) or str(ci)
                        if n:
                            cart_item_names.append(str(n))
                except Exception:
                    pass

            result = planner.run(
                user_text=user_text,
                local_slots=local_slots_dicts,
                cart_item_names=cart_item_names,
                menu_store=getattr(self.menu_repo, "store", None),
                menu_repo=self.menu_repo,
            )
            logger.info(
                "add_item_planner_result",
                extra={
                    "add_item_planner_mode": getattr(
                        getattr(planner, "_config", None), "add_item_planner_mode", "unknown"
                    ),
                    "add_item_planner_route_reason": getattr(result, "route_reason", ""),
                    "add_item_planner_called": getattr(result, "gpt_called", False),
                    "add_item_planner_decision": getattr(result, "decision", ""),
                    "add_item_planner_confidence": getattr(result, "confidence", None),
                    "add_item_planner_validator_passed": getattr(result, "validator_passed", False),
                    "add_item_planner_validator_reject_reason": getattr(result, "validator_reject_reason", None),
                    "add_item_planner_safe_to_apply": getattr(result, "safe_to_apply", False),
                    "add_item_planner_latency_ms": getattr(result, "latency_ms", None),
                },
            )
            return result
        except Exception as exc:
            logger.warning("add_item_planner_exception: %s", exc)
            return None

    def _apply_planner_result(
        self,
        planner_result: Any,
        context: ConversationContext,
    ) -> "HandlerResult | None":
        """Apply a safe planner result to the FSM flow.

        FIX: Multi-item plans (>1 validated item) now route through
        apply_multi_item_plan() instead of returning None.

        Returns a HandlerResult if applied, or None to fall through.
        """
        try:
            validated_plan = getattr(planner_result, "validated_plan", None)
            if validated_plan is None:
                return None
            validated_items = getattr(validated_plan, "items", ()) or ()

            if len(validated_items) == 0:
                return None

            # Multi-item plan — route through executor (FIX: was returning None)
            if len(validated_items) >= 2:
                return apply_multi_item_plan(
                    validated_plan,
                    context,
                    self.menu_repo,
                    self.multi_item_coordinator,
                    get_last_slots=self._get_last_slots,
                )

            # Single-item plan — existing path
            vi = validated_items[0]
            item_id: str = getattr(vi, "item_id", "") or ""
            if not item_id:
                return None

            # Look up the live menu item and delegate to existing resolution path.
            menu_store = getattr(self.menu_repo, "store", None)
            if menu_store is None:
                return None

            menu_item = getattr(menu_store, "items", {}).get(item_id)
            if menu_item is None:
                return None

            from app.menu.query_result import MenuQueryResult
            result = MenuQueryResult(
                items=[menu_item],
                raw_text=getattr(vi, "item_name", ""),
            )
            slots_as_sv: list[Any] = []
            # Build synthetic slot values from validated modifiers/sides
            for vm in getattr(vi, "modifiers", ()):
                from app.nlu.nlu_result import SlotValue
                slots_as_sv.append(SlotValue(name="MODIFIER", value=getattr(vm, "name", "")))
            for vs in getattr(vi, "sides", ()):
                from app.nlu.nlu_result import SlotValue
                slots_as_sv.append(SlotValue(name="SIDE", value=getattr(vs, "name", "")))

            context.reset_item_scope()
            context.pending_action = PendingAction.ADD_ITEM

            return self.item_resolution_handler.resolve_item_and_enter_flow(
                context=context,
                result=result,
                requested_item_text=getattr(vi, "item_name", ""),
                original_user_text=getattr(planner_result, "items", (("",),))[0][0]
                if getattr(planner_result, "items", None) else "",
                slots=slots_as_sv,
            )
        except Exception as exc:
            logger.warning("add_item_planner_apply_failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # SmartTurnPlanner helpers (compound add-item / correction)
    # ------------------------------------------------------------------

    def _try_smart_planner_add_item(
        self,
        *,
        user_text: str,
        context: ConversationContext,
        session: "Any | None",
    ) -> "HandlerResult | None":
        """Attempt SmartTurnPlanner for compound / correction utterances.

        Returns a HandlerResult if the plan is applied, or None to fall through.
        Logs smart_planner_skipped with reason when not invoked.
        Never raises.
        """
        try:
            from app.services.smart_turn_planner import _is_enabled as _stp_enabled
            if not _stp_enabled():
                logger.debug(
                    "smart_planner_skipped",
                    extra={
                        "smart_planner_skipped": True,
                        "smart_planner_skipped_reason": "feature_flag_off",
                    },
                )
                return None

            from app.services.smart_turn_policy import (
                should_use_smart_planner,
                determine_smart_task_mode,
                validate_smart_plan,
            )
            from app.services.smart_turn_planner import plan_smart_turn
            from app.services.smart_turn_context_builder import build_smart_turn_context

            local_confidence = float(context.last_intent_confidence or 0.0)
            state_name = ConversationState.IDLE.value

            should_use, trigger_reason = should_use_smart_planner(
                transcript=user_text,
                state=state_name,
                local_intent="ADD_ITEM",
                local_confidence=local_confidence,
            )
            if not should_use:
                logger.debug(
                    "smart_planner_skipped",
                    extra={
                        "smart_planner_skipped": True,
                        "smart_planner_skipped_reason": trigger_reason,
                    },
                )
                return None

            stp_ctx = build_smart_turn_context(
                transcript=user_text,
                state=state_name,
                local_intent="ADD_ITEM",
                local_confidence=local_confidence,
                context=context,
                session=session,
            )
            task_mode = determine_smart_task_mode(
                transcript=user_text,
                state=state_name,
                local_intent="ADD_ITEM",
                local_confidence=local_confidence,
            )

            plan = plan_smart_turn(
                transcript=user_text,
                state=state_name,
                local_intent="ADD_ITEM",
                local_confidence=local_confidence,
                menu_context=stp_ctx.menu_context,
                cart_snapshot=stp_ctx.cart_snapshot,
                last_cart_diff=stp_ctx.last_cart_diff,
                previous_turns=stp_ctx.previous_turns,
                trigger_reason=trigger_reason,
                task_mode=task_mode,
                allowed_options=stp_ctx.allowed_options,
                pending_item_name=stp_ctx.pending_item_name,
                pending_group_name=stp_ctx.pending_group_name,
                reprompt_count=stp_ctx.reprompt_count,
            )
            if plan is None:
                return None

            validation = validate_smart_plan(
                plan,
                menu_context=stp_ctx.menu_context,
                cart_snapshot=stp_ctx.cart_snapshot,
                state=state_name,
                local_intent="ADD_ITEM",
                trigger_reason=trigger_reason,
            )

            plan_items = getattr(plan, "items", ()) or ()
            logger.info(
                "smart_turn_planner_add_item",
                extra={
                    "smart_planner_invoked": True,
                    "smart_planner_task_mode": task_mode,
                    "smart_planner_trigger_reason": trigger_reason,
                    "smart_planner_decision": getattr(plan, "decision", ""),
                    "smart_planner_confidence": getattr(plan, "confidence", None),
                    "smart_planner_latency_ms": getattr(plan, "latency_ms", None),
                    "smart_planner_validation_result": validation.is_safe,
                    "smart_planner_fallback_reason": (
                        validation.block_reason if not validation.is_safe else None
                    ),
                    "smart_planner_context_keys": stp_ctx.context_keys,
                    "gpt_multi_item_planner_invoked": True,
                    "gpt_multi_item_planner_decision": getattr(plan, "decision", ""),
                    "gpt_multi_item_planner_items": len(plan_items),
                    "gpt_multi_item_planner_reason": trigger_reason,
                    "state_before": state_name,
                },
            )

            if not validation.is_safe:
                return None

            return self._apply_smart_plan_add_item(
                plan=plan,
                context=context,
                normalized_user_text=user_text,
            )
        except Exception as exc:
            logger.warning("smart_turn_planner_add_item_error: %s", exc)
            return None

    def _apply_smart_plan_add_item(
        self,
        *,
        plan: "Any",
        context: ConversationContext,
        normalized_user_text: str,
    ) -> "HandlerResult | None":
        """Apply a validated SmartTurnPlan through the existing item-resolution path.

        FIX: Handles both single-item and multi-item plans.
          - decision == "add_items" + len(items) >= 2 → apply_multi_item_plan()
          - decision == "add_items" + len(items) == 1 → existing single-item path
          - decision == "correction" → existing correction path

        Returns None to fall through if the menu lookup fails.
        Never raises.
        """
        try:
            decision = getattr(plan, "decision", "")

            # ── correction: re-route with corrected item name ─────────────
            if decision == "correction":
                corr = getattr(plan, "correction", None)
                if corr is None:
                    return None
                corrected_text = getattr(corr, "corrected_text", "").strip()
                if not corrected_text:
                    return None
                result = self.menu_repo.resolve_menu_query_normalized(corrected_text, limit=5)
                context.reset_item_scope()
                context.pending_action = PendingAction.ADD_ITEM
                logger.info(
                    "smart_turn_planner_correction_applied",
                    extra={
                        "original_text": getattr(corr, "original_text", ""),
                        "corrected_text": corrected_text,
                        "state_after": "item_resolution",
                    },
                )
                return self.item_resolution_handler.resolve_item_and_enter_flow(
                    context=context,
                    result=result,
                    requested_item_text=corrected_text,
                    original_user_text=corrected_text,
                    slots=[],
                )

            # ── add_items ──────────────────────────────────────────────────
            if decision == "add_items":
                items = getattr(plan, "items", ()) or ()
                if not items:
                    return None

                # FIX: Multi-item → apply all items via executor
                if len(items) >= 2:
                    logger.info(
                        "smart_turn_planner_multi_item_apply",
                        extra={
                            "item_count": len(items),
                            "item_names": [
                                getattr(it, "item_name", str(it)) for it in items
                            ],
                            "gpt_multi_item_planner_validation_errors": [],
                        },
                    )
                    return apply_multi_item_plan(
                        plan,
                        context,
                        self.menu_repo,
                        self.multi_item_coordinator,
                        get_last_slots=self._get_last_slots,
                    )

                # Single item — existing path
                first_item = items[0]
                item_name = getattr(first_item, "item_name", "").strip()
                if not item_name:
                    return None

                result = self.menu_repo.resolve_menu_query_normalized(item_name, limit=5)

                from app.nlu.nlu_result import SlotValue
                slots_as_sv: list[Any] = []
                for mod in getattr(first_item, "modifiers", ()):
                    slots_as_sv.append(SlotValue(name="MODIFIER", value=getattr(mod, "name", "")))
                for sid in getattr(first_item, "sides", ()):
                    slots_as_sv.append(SlotValue(name="SIDE", value=getattr(sid, "name", "")))

                context.reset_item_scope()
                context.pending_action = PendingAction.ADD_ITEM

                logger.info(
                    "smart_turn_planner_add_item_applied",
                    extra={
                        "item_name": item_name,
                        "modifier_count": len(getattr(first_item, "modifiers", ())),
                        "side_count": len(getattr(first_item, "sides", ())),
                        "state_after": "item_resolution",
                    },
                )
                return self.item_resolution_handler.resolve_item_and_enter_flow(
                    context=context,
                    result=result,
                    requested_item_text=item_name,
                    original_user_text=normalized_user_text,
                    slots=slots_as_sv,
                )

            return None
        except Exception as exc:
            logger.warning("smart_turn_planner_apply_add_item_error: %s", exc)
            return None

    def _log_planner_apply_outcome(
        self,
        *,
        planner_result: Any,
        applied: bool,
        block_reason: "str | None",
    ) -> None:
        """Emit a structured log event recording the final apply decision."""
        logger.info(
            "add_item_planner_apply_outcome",
            extra={
                "add_item_planner_applied": applied,
                "add_item_planner_apply_block_reason": block_reason,
                "add_item_planner_route_mode": getattr(planner_result, "route_mode", ""),
                "add_item_planner_confidence": getattr(planner_result, "confidence", None),
            },
        )
