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
from typing import Any, Sequence

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

        # ── Phase 4: GPT Add-Item Planner hook ────────────────────────────
        # Runs BEFORE the local multi-item parser.  In shadow mode the result
        # is logged only — live behavior is unchanged.  In inline mode a
        # safe_to_apply plan is applied for single-item utterances;
        # multi-item inline apply is deferred to a future PR.
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
                return applied
            # safe_to_apply was True but apply helper returned None — this
            # happens when the validated plan has >1 items (multi-item
            # inline apply is deferred) or when the menu lookup fails.
            validated_plan = getattr(planner_result, "validated_plan", None)
            validated_items = getattr(validated_plan, "items", ()) or ()
            block_reason = (
                "multi_item_deferred"
                if len(validated_items) != 1
                else "apply_helper_returned_none"
            )
            self._log_planner_apply_outcome(
                planner_result=planner_result,
                applied=False,
                block_reason=block_reason,
            )
        # ────────────────────────────────────────────────────────────────────

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
        """
        planner = self._gpt_planner
        if planner is None:
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
        """Apply a safe planner result to the FSM flow for single-item plans.

        For multi-item plans (>1 validated item), returns None so the local
        path handles it — multi-item inline apply is a future PR.

        Returns a HandlerResult if applied, or None to fall through.
        """
        try:
            validated_plan = getattr(planner_result, "validated_plan", None)
            if validated_plan is None:
                return None
            validated_items = getattr(validated_plan, "items", ()) or ()
            if len(validated_items) != 1:
                # Multi-item inline apply: defer to future PR, fall through.
                return None

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
            from app.core.pending_action import PendingAction
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

    def _log_planner_apply_outcome(
        self,
        *,
        planner_result: Any,
        applied: bool,
        block_reason: "str | None",
    ) -> None:
        """Emit a structured log event recording the final apply decision.

        Emitted only when safe_to_apply=True (i.e. when the apply gate passed
        and we attempted application).  Adds two fields not available at the
        earlier _try_gpt_planner() logging point:

        ``add_item_planner_applied``
            True  — the plan was applied and a HandlerResult was returned.
            False — the helper returned None (multi-item deferred, menu miss).

        ``add_item_planner_apply_block_reason``
            None when applied.
            "multi_item_deferred"       — validated plan has != 1 item.
            "apply_helper_returned_none" — unexpected None from helper.
        """
        logger.info(
            "add_item_planner_apply_outcome",
            extra={
                "add_item_planner_applied": applied,
                "add_item_planner_apply_block_reason": block_reason,
                "add_item_planner_route_mode": getattr(planner_result, "route_mode", ""),
                "add_item_planner_confidence": getattr(planner_result, "confidence", None),
            },
        )
