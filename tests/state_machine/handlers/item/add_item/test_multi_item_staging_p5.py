# tests/state_machine/handlers/item/add_item/test_multi_item_staging_p5.py
"""Priority 5: Multi-Item Structured Staging tests.

Coverage
--------
MS-01  Two items each with a related side — correct parent attachment, no cross-contamination
MS-02  Staged item drain bypasses AddItemHandler.handle; uses PrefillOrchestrator directly
MS-03  Second staged item becomes active after first item is completed (drain sequence)
MS-04  Four-item utterance: Fries Large / Onion Rings Small / Tuna Melt / Wings 6pc
MS-05  "two 6 piece wings": quantity=2, variant_label="6 piece" (not quantity=6)
MS-06  Partial success (1 resolved + 1 unresolved) → EXECUTE_PARTIAL_AND_CLARIFY
MS-07  Checkout blocked when staged_item_queue is non-empty
MS-08  One-at-a-time escalation only after reprompt_count ≥ 2; not on first failure
MS-09  format_limited_options: max 4 spoken options + overflow suffix when more exist
MS-10  format_limited_options: no overflow suffix when all options fit within max_spoken
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
from unittest.mock import MagicMock

import pytest

from app.core.item_queue_service import ItemQueueService
from app.responses.item.format_utils import format_limited_options
from app.services.compound_turn_policy import (
    CompoundFallbackDecision,
    decide_compound_fallback,
)
from app.services.order_lifecycle_guard import LifecycleCode, can_checkout
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.multi_item_plan_executor import (
    apply_multi_item_plan,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import (
    StagedItemPlan,
    StagedModifier,
    StagedSide,
)


# ---------------------------------------------------------------------------
# Shared test stubs
# ---------------------------------------------------------------------------


@dataclass
class _PlanItem:
    """Generic duck-typed plan item mirroring the executor's expected interface."""
    item_name: str = ""
    item_id: str = ""
    raw_span: str = ""
    quantity: int = 1
    size: str = ""
    size_name: str = ""
    variant: str = ""
    variant_name: str = ""
    modifiers: list = field(default_factory=list)
    sides: list = field(default_factory=list)


@dataclass
class _Side:
    """Duck-typed side with optional size attribute."""
    name: str
    size: str = ""
    variant: str = ""
    quantity: int = 1


@dataclass
class _Modifier:
    name: str
    operation: str = "add"


@dataclass
class _Plan:
    items: tuple = ()


class _FakeCoordinator:
    """Records calls to .handle() and returns a canned HandlerResult."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._result = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
        )

    def handle(
        self,
        *,
        context: Any,
        segments: list,
        get_last_slots: Callable,
        staged_items: Any = None,
    ) -> HandlerResult:
        self.calls.append({
            "segments": segments,
            "staged_items": staged_items,
        })
        return self._result


def _noop_slots(ctx: Any) -> Sequence:
    return ()


def _make_repo() -> MagicMock:
    return MagicMock()


def _make_context() -> ConversationContext:
    return ConversationContext()


def _make_session(ctx: ConversationContext) -> MagicMock:
    session = MagicMock()
    session.conversation_context = ctx
    session.session_id = "test_session"
    session.restaurant_id = "test_restaurant"
    return session


# ---------------------------------------------------------------------------
# MS-01  Two items each with a related side — correct attachment, no cross-contamination
# ---------------------------------------------------------------------------


class TestTwoItemsWithRelatedSides:
    """MS-01: sides attach to their parent item, not the other."""

    def test_side_attached_to_correct_staged_item(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="Chicken Burger", sides=[_Side("Coke", size="Small")]),
            _PlanItem(item_name="Hamburger", sides=[_Side("Fries", size="Large")]),
        ))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _make_context(), _make_repo(), coord,
            get_last_slots=_noop_slots,
        )
        assert result is not None
        staged = coord.calls[0]["staged_items"]
        assert len(staged) == 1
        hamburger = staged[0]
        assert hamburger.item_name == "Hamburger"
        assert len(hamburger.requested_sides) == 1
        assert hamburger.requested_sides[0].name == "Fries"
        assert hamburger.requested_sides[0].variant_label == "Large"

    def test_first_segment_side_not_in_staged_item(self) -> None:
        """Chicken Burger's Coke must NOT appear in the Hamburger staged plan."""
        plan = _Plan(items=(
            _PlanItem(item_name="Chicken Burger", sides=[_Side("Coke", size="Small")]),
            _PlanItem(item_name="Hamburger", sides=[_Side("Fries", size="Large")]),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _make_context(), _make_repo(), coord,
            get_last_slots=_noop_slots,
        )
        staged = coord.calls[0]["staged_items"]
        hamburger = staged[0]
        side_names = [s.name for s in hamburger.requested_sides]
        assert "Coke" not in side_names, (
            f"Expected Coke (Chicken Burger's side) NOT in Hamburger's staged sides, got {side_names}"
        )

    def test_side_size_preserved_in_staged_item(self) -> None:
        """Size/variant on a staged item's side is preserved correctly."""
        plan = _Plan(items=(
            _PlanItem(item_name="Burger"),
            _PlanItem(item_name="Wings", sides=[_Side("Coke", size="Small")]),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _make_context(), _make_repo(), coord,
            get_last_slots=_noop_slots,
        )
        staged = coord.calls[0]["staged_items"]
        wings = staged[0]
        assert wings.requested_sides[0].name == "Coke"
        assert wings.requested_sides[0].variant_label == "Small"

    def test_staged_item_with_no_sides(self) -> None:
        """A staged item with no sides has an empty requested_sides tuple."""
        plan = _Plan(items=(
            _PlanItem(item_name="Burger"),
            _PlanItem(item_name="Tuna Melt"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _make_context(), _make_repo(), coord,
            get_last_slots=_noop_slots,
        )
        staged = coord.calls[0]["staged_items"]
        tuna = staged[0]
        assert tuna.requested_sides == ()


# ---------------------------------------------------------------------------
# MS-02  Staged item drain bypasses AddItemHandler.handle
# ---------------------------------------------------------------------------


class TestStagedItemBypassesHandler:
    """MS-02: ItemQueueService drains staged plans via PrefillOrchestrator, not handle()."""

    def _make_drain_service(self, add_handler: Any) -> ItemQueueService:
        return ItemQueueService(
            handlers={"add_item_handler": add_handler},
            command_executor=MagicMock(),
        )

    def test_handle_not_called_for_staged_item(self) -> None:
        staged_plan = StagedItemPlan(
            item_id="item_01", item_name="Hamburger", quantity=1,
        )
        ctx = _make_context()
        ctx.staged_item_queue.append(staged_plan)

        mock_menu_item = MagicMock()  # truthy → resolution succeeds
        add_handler = MagicMock()
        add_handler.menu_repo.store.get_item.return_value = mock_menu_item
        add_handler.menu_repo.store.find_item_exact.return_value = None
        add_handler.prefill_orchestrator.enter_add_flow_for_item.return_value = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
        )

        service = self._make_drain_service(add_handler)
        current_result = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            response_payload={"item_name": "Chicken Burger", "quantity": 1},
        )
        service.try_drain(session=_make_session(ctx), current_result=current_result)

        # handle() must NOT be called for staged items
        add_handler.handle.assert_not_called()

    def test_prefill_orchestrator_called_with_staged_kwarg(self) -> None:
        """enter_add_flow_for_item must be called with staged=<the staged plan>."""
        staged_plan = StagedItemPlan(
            item_id="item_01", item_name="Hamburger", quantity=1,
        )
        ctx = _make_context()
        ctx.staged_item_queue.append(staged_plan)

        mock_menu_item = MagicMock()
        add_handler = MagicMock()
        add_handler.menu_repo.store.get_item.return_value = mock_menu_item
        add_handler.prefill_orchestrator.enter_add_flow_for_item.return_value = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
        )

        service = self._make_drain_service(add_handler)
        current_result = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
        )
        service.try_drain(session=_make_session(ctx), current_result=current_result)

        call_kwargs = add_handler.prefill_orchestrator.enter_add_flow_for_item.call_args[1]
        assert call_kwargs.get("staged") is staged_plan

    def test_no_drain_if_current_result_not_item_added(self) -> None:
        """try_drain returns None when current state is not item_added_successfully."""
        staged_plan = StagedItemPlan(item_id="i1", item_name="Burger")
        ctx = _make_context()
        ctx.staged_item_queue.append(staged_plan)

        add_handler = MagicMock()
        service = self._make_drain_service(add_handler)

        # response_key is NOT item_added_successfully
        current_result = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
        )
        result = service.try_drain(session=_make_session(ctx), current_result=current_result)
        assert result is None
        add_handler.handle.assert_not_called()
        add_handler.prefill_orchestrator.enter_add_flow_for_item.assert_not_called()

    def test_staged_queue_emptied_after_successful_drain(self) -> None:
        """After a staged item is drained, staged_item_queue should be empty."""
        staged_plan = StagedItemPlan(item_id="item_01", item_name="Hamburger", quantity=1)
        ctx = _make_context()
        ctx.staged_item_queue.append(staged_plan)

        add_handler = MagicMock()
        add_handler.menu_repo.store.get_item.return_value = MagicMock()
        add_handler.prefill_orchestrator.enter_add_flow_for_item.return_value = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
        )

        service = self._make_drain_service(add_handler)
        current_result = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
        )
        service.try_drain(session=_make_session(ctx), current_result=current_result)
        assert len(ctx.staged_item_queue) == 0


# ---------------------------------------------------------------------------
# MS-03  Second staged item active after first completes (drain sequence)
# ---------------------------------------------------------------------------


class TestStagedItemDrainSequence:
    """MS-03: drain result reflects the second item's state (not the first's)."""

    def test_drain_result_reflects_staged_item_state(self) -> None:
        staged_plan = StagedItemPlan(
            item_id="item_02", item_name="Hamburger", quantity=1,
            raw_span="hamburger",
        )
        ctx = _make_context()
        ctx.staged_item_queue.append(staged_plan)

        expected_result = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
            response_payload={"item_name": "Hamburger", "group_name": "bun"},
        )
        add_handler = MagicMock()
        add_handler.menu_repo.store.get_item.return_value = MagicMock()
        add_handler.prefill_orchestrator.enter_add_flow_for_item.return_value = expected_result

        service = ItemQueueService(
            handlers={"add_item_handler": add_handler},
            command_executor=MagicMock(),
        )
        current_result = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            response_payload={"item_name": "Chicken Burger", "quantity": 1},
        )
        drain_result = service.try_drain(
            session=_make_session(ctx), current_result=current_result
        )

        assert drain_result is not None
        assert drain_result.next_state == ConversationState.WAITING_FOR_MODIFIER
        assert drain_result.response_key == "ask_for_modifier"

    def test_drain_result_carries_queue_transition_flag(self) -> None:
        staged_plan = StagedItemPlan(item_id="item_02", item_name="Hamburger")
        ctx = _make_context()
        ctx.staged_item_queue.append(staged_plan)

        add_handler = MagicMock()
        add_handler.menu_repo.store.get_item.return_value = MagicMock()
        add_handler.prefill_orchestrator.enter_add_flow_for_item.return_value = HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIDE,
            response_key="ask_for_side",
            response_payload={"item_name": "Hamburger"},
        )

        service = ItemQueueService(
            handlers={"add_item_handler": add_handler},
            command_executor=MagicMock(),
        )
        drain_result = service.try_drain(
            session=_make_session(ctx),
            current_result=HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="item_added_successfully",
            ),
        )
        assert drain_result is not None
        assert drain_result.response_payload.get("queue_transition") is True


# ---------------------------------------------------------------------------
# MS-04  Four-item utterance: Fries Large / Onion Rings Small / Tuna Melt / Wings 6pc
# ---------------------------------------------------------------------------


class TestFourItemParsing:
    """MS-04: 4-item plan → 1 segment (first) + 3 staged items with correct attributes."""

    def _run_plan(self) -> tuple[list, list]:
        plan = _Plan(items=(
            _PlanItem(item_name="Fries", size="Large"),
            _PlanItem(item_name="Onion Rings", size="Small"),
            _PlanItem(item_name="Tuna Melt"),
            _PlanItem(item_name="Wings", variant_name="6 piece", quantity=1),
        ))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _make_context(), _make_repo(), coord,
            get_last_slots=_noop_slots,
        )
        assert result is not None
        return coord.calls[0]["segments"], coord.calls[0]["staged_items"]

    def test_first_item_becomes_segment(self) -> None:
        segments, _ = self._run_plan()
        assert len(segments) == 1
        assert segments[0].item_slot_value == "Fries"

    def test_remaining_three_items_staged(self) -> None:
        _, staged = self._run_plan()
        assert len(staged) == 3
        names = [s.item_name for s in staged]
        assert names == ["Onion Rings", "Tuna Melt", "Wings"]

    def test_onion_rings_size_small_preserved(self) -> None:
        _, staged = self._run_plan()
        onion_rings = staged[0]
        assert onion_rings.item_name == "Onion Rings"
        assert onion_rings.variant_label == "Small"

    def test_tuna_melt_has_no_variant(self) -> None:
        _, staged = self._run_plan()
        tuna_melt = staged[1]
        assert tuna_melt.item_name == "Tuna Melt"
        assert not tuna_melt.variant_label

    def test_wings_variant_is_6pc_not_item_size(self) -> None:
        _, staged = self._run_plan()
        wings = staged[2]
        assert wings.item_name == "Wings"
        assert wings.variant_label == "6 piece"
        assert wings.quantity == 1  # NOT 6


# ---------------------------------------------------------------------------
# MS-05  "two 6 piece wings" → quantity=2, variant_label="6 piece"
# ---------------------------------------------------------------------------


class TestTwoSixPieceWings:
    """MS-05: compound quantity+variant utterance — correct quantity, correct variant."""

    def test_quantity_two_variant_6_piece_preserved(self) -> None:
        """'two 6 piece wings' should produce quantity=2, variant='6 piece'."""
        plan = _Plan(items=(
            _PlanItem(item_name="Wings", variant_name="6 piece", quantity=2),
            _PlanItem(item_name="Burger"),  # second item needed to trigger multi path
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _make_context(), _make_repo(), coord,
            get_last_slots=_noop_slots,
        )
        segments = coord.calls[0]["segments"]
        wings_seg = segments[0]
        # quantity=2 in segment (not None, which represents 1)
        assert wings_seg.quantity == 2
        # VARIANT slot should be "6 piece"
        variant_slots = [sv for sv in wings_seg.slots if sv.name == "VARIANT"]
        assert len(variant_slots) == 1
        assert variant_slots[0].value == "6 piece"

    def test_quantity_not_six_for_6_piece(self) -> None:
        """'6 piece wings' in the second position must have quantity=1, not 6."""
        plan = _Plan(items=(
            _PlanItem(item_name="Burger"),
            _PlanItem(item_name="Wings", variant_name="6 piece", quantity=1),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _make_context(), _make_repo(), coord,
            get_last_slots=_noop_slots,
        )
        staged = coord.calls[0]["staged_items"]
        wings = staged[0]
        assert wings.item_name == "Wings"
        assert wings.variant_label == "6 piece"
        assert wings.quantity == 1  # quantity=1, NOT 6


# ---------------------------------------------------------------------------
# MS-06  Partial success: 1 resolved + 1 unresolved → EXECUTE_PARTIAL_AND_CLARIFY
# ---------------------------------------------------------------------------


class TestPartialSuccessCompoundPolicy:
    """MS-06: when at least one item resolves, the compound policy clarifies (not fails)."""

    def test_one_resolved_one_unresolved_returns_partial(self) -> None:
        result = decide_compound_fallback(
            transcript="hamburger dragon pasta fries",
            planner_result=None,
            local_planner_result=None,
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=1,   # hamburger resolved
            unresolved_spans=["dragon pasta"],
            reprompt_count=0,
        )
        assert result == CompoundFallbackDecision.EXECUTE_PARTIAL_AND_CLARIFY

    def test_all_resolved_returns_valid_plan(self) -> None:
        result = decide_compound_fallback(
            transcript="hamburger fries",
            planner_result=None,
            local_planner_result=None,
            unsafe_slot_reason=None,
            valid_candidates_count=2,
            unresolved_spans=[],
            reprompt_count=0,
        )
        assert result == CompoundFallbackDecision.EXECUTE_VALID_PLAN

    def test_zero_resolved_is_not_partial(self) -> None:
        result = decide_compound_fallback(
            transcript="garbled speech",
            planner_result=None,
            local_planner_result=None,
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=0,
            unresolved_spans=["garbled", "speech"],
            reprompt_count=1,
        )
        # With zero resolved and reprompt_count=1 → FALLBACK_REPEAT_FIRST_ITEM
        assert result == CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM

    def test_partial_is_not_one_at_a_time(self) -> None:
        result = decide_compound_fallback(
            transcript="hamburger dragon pasta",
            planner_result=None,
            local_planner_result=None,
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=1,
            unresolved_spans=["dragon pasta"],
            reprompt_count=0,
        )
        assert result != CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME


# ---------------------------------------------------------------------------
# MS-07  Checkout blocked when staged_item_queue is non-empty
# ---------------------------------------------------------------------------


class TestCheckoutBlockedWithStagedQueue:
    """MS-07: can_checkout blocks when staged items are still waiting."""

    def _make_cart(self, empty: bool = False) -> MagicMock:
        cart = MagicMock()
        cart.is_empty.return_value = empty
        return cart

    def test_checkout_blocked_with_one_staged_item(self) -> None:
        ctx = _make_context()
        ctx.staged_item_queue.append(
            StagedItemPlan(item_id="item_01", item_name="Hamburger")
        )
        result = can_checkout(cart=self._make_cart(empty=False), context=ctx)
        assert result.blocking is True
        assert result.code == LifecycleCode.CART_INCOMPLETE

    def test_checkout_blocked_with_multiple_staged_items(self) -> None:
        ctx = _make_context()
        ctx.staged_item_queue.append(StagedItemPlan(item_id="i1", item_name="Burger"))
        ctx.staged_item_queue.append(StagedItemPlan(item_id="i2", item_name="Fries"))
        result = can_checkout(cart=self._make_cart(empty=False), context=ctx)
        assert result.blocking is True
        assert result.code == LifecycleCode.CART_INCOMPLETE

    def test_checkout_response_mentions_remaining_items(self) -> None:
        ctx = _make_context()
        ctx.staged_item_queue.append(StagedItemPlan(item_id="i1", item_name="Burger"))
        result = can_checkout(cart=self._make_cart(empty=False), context=ctx)
        assert "remaining items" in result.response.lower() or "finish" in result.response.lower()

    def test_checkout_allowed_when_staged_queue_empty(self) -> None:
        ctx = _make_context()
        # No pending_add_item, no staged_item_queue, no pending_item_queue
        result = can_checkout(cart=self._make_cart(empty=False), context=ctx)
        assert not result.blocking

    def test_checkout_blocked_with_legacy_pending_queue_too(self) -> None:
        from app.state_machine.models.pending_item_models import QueuedItemRequest
        ctx = _make_context()
        ctx.pending_item_queue.append(
            QueuedItemRequest(raw_text="fries", item_slot_value="Fries")
        )
        result = can_checkout(cart=self._make_cart(empty=False), context=ctx)
        assert result.blocking is True
        assert result.code == LifecycleCode.CART_INCOMPLETE


# ---------------------------------------------------------------------------
# MS-08  One-at-a-time escalation only after reprompt_count ≥ 2
# ---------------------------------------------------------------------------


class TestOneAtATimeEscalation:
    """MS-08: FALLBACK_ONE_AT_A_TIME is a last resort — not used on first failure."""

    def _call(self, reprompt_count: int) -> CompoundFallbackDecision:
        # Transcript must not contain any _ITEM_OPTION_MARKERS (" with ", " no ",
        # " extra ", " add ", " light ", etc.) to avoid Rule 6 firing.
        return decide_compound_fallback(
            transcript="garbled unclear speech",
            planner_result=None,
            local_planner_result=None,
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=0,
            unresolved_spans=[],
            reprompt_count=reprompt_count,
        )

    def test_first_encounter_not_one_at_a_time(self) -> None:
        result = self._call(reprompt_count=0)
        assert result != CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME

    def test_second_encounter_asks_first_item(self) -> None:
        result = self._call(reprompt_count=1)
        assert result == CompoundFallbackDecision.FALLBACK_REPEAT_FIRST_ITEM

    def test_third_encounter_escalates(self) -> None:
        result = self._call(reprompt_count=2)
        assert result == CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME

    def test_fourth_encounter_still_escalates(self) -> None:
        result = self._call(reprompt_count=5)
        assert result == CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME

    def test_valid_item_never_one_at_a_time(self) -> None:
        """When at least 1 valid item exists, one-at-a-time is never used."""
        result = decide_compound_fallback(
            transcript="hamburger dragon pasta",
            planner_result=None,
            local_planner_result=None,
            unsafe_slot_reason="multi_item_slots",
            valid_candidates_count=1,
            unresolved_spans=["dragon pasta"],
            reprompt_count=5,  # even with high reprompt_count
        )
        assert result != CompoundFallbackDecision.FALLBACK_ONE_AT_A_TIME


# ---------------------------------------------------------------------------
# MS-09/MS-10  format_limited_options helper
# ---------------------------------------------------------------------------


class TestFormatLimitedOptions:
    """MS-09 & MS-10: format_limited_options speaks at most max_spoken options."""

    def test_ms09_six_options_default_speaks_four(self) -> None:
        """MS-09: with 6 options and default max_spoken=4, only 4 are spoken."""
        options = [
            "Plain Bun", "Sesame Bun", "Potato Bun", "Brioche Bun",
            "Pretzel Bun", "Whole Wheat Bun",
        ]
        result = format_limited_options(options)
        assert "Plain Bun" in result
        assert "Sesame Bun" in result
        assert "Potato Bun" in result
        assert "Brioche Bun" in result
        assert "Pretzel Bun" not in result
        assert "Whole Wheat Bun" not in result

    def test_ms09_overflow_suffix_included_when_more_exist(self) -> None:
        """MS-09: overflow suffix is included when options exceed max_spoken."""
        options = ["A", "B", "C", "D", "E"]
        result = format_limited_options(options, max_spoken=4)
        assert "options" in result.lower(), (
            f"Expected overflow suffix containing 'options', got: {result!r}"
        )

    def test_ms10_no_overflow_when_all_options_fit(self) -> None:
        """MS-10: no overflow suffix when total options ≤ max_spoken."""
        options = ["Small", "Medium", "Large"]
        result = format_limited_options(options, max_spoken=4)
        assert "options" not in result.lower(), (
            f"Expected no overflow suffix for 3 options (max 4), got: {result!r}"
        )

    def test_ms10_exactly_at_limit_no_overflow(self) -> None:
        """MS-10: exactly max_spoken options → no overflow suffix."""
        options = ["A", "B", "C", "D"]
        result = format_limited_options(options, max_spoken=4)
        assert "A" in result
        assert "D" in result
        assert "options" not in result.lower()

    def test_custom_max_spoken(self) -> None:
        """max_spoken parameter is respected."""
        options = ["A", "B", "C"]
        result_2 = format_limited_options(options, max_spoken=2)
        assert "A" in result_2
        assert "B" in result_2
        assert "C" not in result_2
        assert "options" in result_2.lower()

    def test_custom_overflow_suffix(self) -> None:
        """Custom overflow_suffix is appended when more options exist."""
        options = ["X", "Y", "Z", "W", "V"]
        result = format_limited_options(
            options, max_spoken=3, overflow_suffix="say more to continue"
        )
        assert "say more to continue" in result

    def test_empty_options_returns_empty(self) -> None:
        result = format_limited_options([])
        assert result == ""

    def test_single_option_no_overflow(self) -> None:
        result = format_limited_options(["Only Option"])
        assert result == "Only Option"
        assert "options" not in result.lower()

    def test_two_options_joined_with_or(self) -> None:
        result = format_limited_options(["Coke", "Sprite"])
        assert result == "Coke or Sprite"

    def test_overflow_suffix_is_sentence(self) -> None:
        """Default overflow suffix produces a complete readable string."""
        options = ["A", "B", "C", "D", "E"]
        result = format_limited_options(options)
        # Should end with the overflow suffix
        assert result.endswith("Say options to hear more")

    def test_list_options_path_can_use_higher_limit(self) -> None:
        """MS-10: when list_options is requested, caller uses higher max_spoken."""
        all_options = ["Plain", "Sesame", "Potato", "Brioche", "Pretzel", "Wheat"]
        # Simulate list_options behavior: max_spoken=6 (or more)
        result = format_limited_options(all_options, max_spoken=6)
        # All 6 options should be present
        for opt in all_options:
            assert opt in result, f"Expected {opt!r} in list_options result, got: {result!r}"
        # No overflow suffix since all fit
        assert "options" not in result.lower()

    def test_format_returns_natural_list(self) -> None:
        """Three options produce 'A, B, or C' format."""
        result = format_limited_options(["A", "B", "C"], max_spoken=4)
        assert result == "A, B, or C"
