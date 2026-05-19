# tests/state_machine/handlers/item/add_item/test_multi_item_plan_executor.py
"""Unit tests for apply_multi_item_plan (multi_item_plan_executor module).

Test categories
---------------
PE-01  plan with < 2 items → None
PE-02  plan with 0 items → None
PE-03  plan with 2 items → coordinator called, 1 segment + 1 staged
PE-04  plan with 5 items → coordinator called with 1 segment + 4 staged
PE-05  item with no item_name and no raw_span → dropped; if < 2 survive → None
PE-06  item_name falls back to raw_span when item_name missing/empty
PE-07  quantity clamped: qty=0 → 1 (first segment)
PE-08  quantity clamped: qty=200 → 99 (first segment)
PE-09  quantity=1 stored as None in segment (coordinator treats None as 1)
PE-10  quantity=2 stored as 2 in segment
PE-11  size/size_name → staged variant_label for items[1..N]
PE-12  variant/variant_name → staged variant_label for items[1..N]
PE-13  modifiers → staged requested_modifiers for items[1..N]
PE-14  sides → staged requested_sides for items[1..N]
PE-15  exception inside coordinator.handle → apply_multi_item_plan returns None
PE-16  duck-typing: SmartTurnItem-like object (has 'size' attribute) works
PE-17  duck-typing: GptValidatedItem-like object (has 'size_name') works
PE-18  coordinator receives correct context and get_last_slots callable
PE-19  ITEM synthetic slot value equals item_name for first segment; staged items have item_name
PE-20  plan attribute 'items' missing → None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
from unittest.mock import MagicMock, call

import pytest

from app.state_machine.handlers.item.add_item.multi_item_plan_executor import (
    apply_multi_item_plan,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


@dataclass
class _PlanItem:
    """Generic duck-typed plan item."""
    item_name: str = ""
    raw_span: str = ""
    quantity: int = 1
    size: str = ""
    size_name: str = ""
    variant: str = ""
    variant_name: str = ""
    modifiers: list = field(default_factory=list)
    sides: list = field(default_factory=list)


@dataclass
class _Modifier:
    name: str


@dataclass
class _Side:
    name: str


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
        self.calls.append(
            {
                "context": context,
                "segments": segments,
                "get_last_slots": get_last_slots,
                "staged_items": staged_items,
            }
        )
        return self._result


class _FakeContext:
    last_slots: tuple = ()


def _noop_get_slots(ctx: Any) -> Sequence:
    return ctx.last_slots


def _make_repo() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# PE-01/02  plan with < 2 items → None
# ---------------------------------------------------------------------------


class TestFewerThanTwoItems:
    def test_one_item_returns_none(self) -> None:
        plan = _Plan(items=(_PlanItem(item_name="burger"),))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is None
        assert coord.calls == []

    def test_zero_items_returns_none(self) -> None:
        plan = _Plan(items=())
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is None
        assert coord.calls == []

    def test_plan_with_no_items_attribute_returns_none(self) -> None:
        """PE-20: plan missing 'items' attribute."""
        plan = object()  # no 'items'
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is None

    def test_items_none_returns_none(self) -> None:
        plan = _Plan(items=None)  # type: ignore[arg-type]
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is None


# ---------------------------------------------------------------------------
# PE-03/04  Happy path: 2+ items → coordinator called
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_two_items_calls_coordinator(self) -> None:
        """PE-03: 2 items → 1 segment (items[0]) + 1 staged item (items[1])."""
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="fries"),
        ))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is not None
        assert result.response_key == "item_added_successfully"
        assert len(coord.calls) == 1
        # items[0] → single segment passed to coordinator
        segments = coord.calls[0]["segments"]
        assert len(segments) == 1
        assert segments[0].item_slot_value == "burger"
        # items[1] → staged_items list
        staged = coord.calls[0]["staged_items"]
        assert staged is not None and len(staged) == 1
        assert staged[0].item_name == "fries"

    def test_five_items_produces_one_segment_and_four_staged(self) -> None:
        """PE-04: 5 items → 1 segment (items[0]) + 4 staged items (items[1..4])."""
        plan = _Plan(items=(
            _PlanItem(item_name="grilled chicken sandwich"),
            _PlanItem(item_name="large fries"),
            _PlanItem(item_name="small onion rings"),
            _PlanItem(item_name="tuna melt"),
            _PlanItem(item_name="chicken wings"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        assert len(segments) == 1
        assert segments[0].item_slot_value == "grilled chicken sandwich"
        staged = coord.calls[0]["staged_items"]
        assert staged is not None and len(staged) == 4
        staged_names = [s.item_name for s in staged]
        assert staged_names == ["large fries", "small onion rings", "tuna melt", "chicken wings"]

    def test_returns_coordinator_result_directly(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="fries"),
        ))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is coord._result


# ---------------------------------------------------------------------------
# PE-05/06  Items with missing names
# ---------------------------------------------------------------------------


class TestMissingItemNames:
    def test_item_with_no_name_dropped(self) -> None:
        """If dropping unnamed items leaves < 2, returns None."""
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="", raw_span=""),  # no name → dropped
        ))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is None

    def test_raw_span_used_as_fallback_name(self) -> None:
        """PE-06: item_name empty but raw_span present → uses raw_span.
        items[0] raw_span → first segment; items[1] → staged."""
        plan = _Plan(items=(
            _PlanItem(item_name="", raw_span="tuna melt"),
            _PlanItem(item_name="burger"),
        ))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is not None
        segments = coord.calls[0]["segments"]
        # items[0] becomes the first (and only) segment
        assert segments[0].item_slot_value == "tuna melt"
        # items[1] goes into staged_items
        staged = coord.calls[0]["staged_items"]
        assert staged is not None and staged[0].item_name == "burger"

    def test_two_valid_and_one_unnamed_still_routes(self) -> None:
        """With 2 valid items after dropping unnamed, should still route.
        items[0]=burger → segment; items[1]=fries → staged; items[2]='' → dropped."""
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="fries"),
            _PlanItem(item_name="", raw_span=""),  # dropped
        ))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is not None
        # 1 segment (items[0]) + 1 staged (items[1]); items[2] dropped
        assert len(coord.calls[0]["segments"]) == 1
        assert len(coord.calls[0]["staged_items"]) == 1


# ---------------------------------------------------------------------------
# PE-07/08/09/10  Quantity handling
# ---------------------------------------------------------------------------


class TestQuantity:
    def test_quantity_zero_clamped_to_one(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="burger", quantity=0),
            _PlanItem(item_name="fries", quantity=1),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        seg_burger = coord.calls[0]["segments"][0]
        # qty=1 → stored as None
        assert seg_burger.quantity is None

    def test_quantity_200_clamped_to_99(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="burger", quantity=200),
            _PlanItem(item_name="fries", quantity=1),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        seg_burger = coord.calls[0]["segments"][0]
        assert seg_burger.quantity == 99

    def test_quantity_one_stored_as_none(self) -> None:
        """qty=1: first segment has quantity=None (coordinator treats None as 1);
        staged item has quantity=1 (int)."""
        plan = _Plan(items=(
            _PlanItem(item_name="burger", quantity=1),
            _PlanItem(item_name="fries", quantity=1),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        # First segment → quantity=None
        assert coord.calls[0]["segments"][0].quantity is None
        # Staged item → quantity=1 (stored as int, not None)
        assert coord.calls[0]["staged_items"][0].quantity == 1

    def test_quantity_two_stored_as_two(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="burger", quantity=2),
            _PlanItem(item_name="fries", quantity=1),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        seg_burger = coord.calls[0]["segments"][0]
        assert seg_burger.quantity == 2

    def test_quantity_99_not_clamped(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="burger", quantity=99),
            _PlanItem(item_name="fries"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        seg_burger = coord.calls[0]["segments"][0]
        assert seg_burger.quantity == 99


# ---------------------------------------------------------------------------
# PE-11/12  Size / variant slots
# ---------------------------------------------------------------------------


class TestSizeVariantSlots:
    def _get_segment_slot_values(self, segments: list, name: str) -> list[str]:
        results = []
        for seg in segments:
            for sv in seg.slots:
                if sv.name == name:
                    results.append(sv.value)
        return results

    def test_size_attribute_on_first_item_produces_variant_slot(self) -> None:
        """PE-11: items[0].size → VARIANT SlotValue in first segment."""
        plan = _Plan(items=(
            _PlanItem(item_name="fries", size="large"),
            _PlanItem(item_name="burger"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        variant_values = self._get_segment_slot_values(segments, "VARIANT")
        assert "large" in variant_values

    def test_size_attribute_on_staged_item_preserved_as_variant_label(self) -> None:
        """PE-11: items[1].size → staged item's variant_label."""
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="fries", size="large"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        staged = coord.calls[0]["staged_items"]
        assert staged[0].variant_label == "large"

    def test_size_name_attribute_on_first_item_produces_variant_slot(self) -> None:
        """PE-11: items[0].size_name → VARIANT SlotValue in first segment."""
        plan = _Plan(items=(
            _PlanItem(item_name="fries", size_name="small"),
            _PlanItem(item_name="burger"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        variant_values = self._get_segment_slot_values(segments, "VARIANT")
        assert "small" in variant_values

    def test_variant_attribute_on_first_item_produces_variant_slot(self) -> None:
        """PE-12: items[0].variant → VARIANT SlotValue in first segment."""
        plan = _Plan(items=(
            _PlanItem(item_name="wings", variant="6 piece"),
            _PlanItem(item_name="burger"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        variant_values = self._get_segment_slot_values(segments, "VARIANT")
        assert "6 piece" in variant_values

    def test_variant_name_attribute_on_first_item_produces_variant_slot(self) -> None:
        """PE-12: items[0].variant_name → VARIANT SlotValue in first segment."""
        plan = _Plan(items=(
            _PlanItem(item_name="wings", variant_name="12 piece"),
            _PlanItem(item_name="burger"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        variant_values = self._get_segment_slot_values(segments, "VARIANT")
        assert "12 piece" in variant_values


# ---------------------------------------------------------------------------
# PE-13/14  Modifier and side slots
# ---------------------------------------------------------------------------


class TestModifierSideSlots:
    def _get_slot_values(self, segments: list, name: str) -> list[str]:
        results = []
        for seg in segments:
            for sv in seg.slots:
                if sv.name == name:
                    results.append(sv.value)
        return results

    def test_modifiers_produce_modifier_slots(self) -> None:
        """PE-13: modifiers list → MODIFIER SlotValues."""
        plan = _Plan(items=(
            _PlanItem(
                item_name="burger",
                modifiers=[_Modifier("extra cheese"), _Modifier("no onions")],
            ),
            _PlanItem(item_name="fries"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        modifier_values = self._get_slot_values(segments, "MODIFIER")
        assert "extra cheese" in modifier_values
        assert "no onions" in modifier_values

    def test_sides_produce_side_slots(self) -> None:
        """PE-14: sides list → SIDE SlotValues."""
        plan = _Plan(items=(
            _PlanItem(
                item_name="burger",
                sides=[_Side("fries"), _Side("coke")],
            ),
            _PlanItem(item_name="wings"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        side_values = self._get_slot_values(segments, "SIDE")
        assert "fries" in side_values
        assert "coke" in side_values

    def test_no_modifiers_no_side_slots(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="fries"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        for seg in segments:
            for sv in seg.slots:
                assert sv.name not in {"MODIFIER", "SIDE"}


# ---------------------------------------------------------------------------
# PE-15  Exception inside coordinator → None returned
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    def test_coordinator_exception_returns_none(self) -> None:
        """PE-15: if coordinator.handle raises, apply_multi_item_plan returns None."""
        class _BrokenCoordinator:
            def handle(self, **kwargs):
                raise RuntimeError("coordinator offline")

        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="fries"),
        ))
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), _BrokenCoordinator(),  # type: ignore[arg-type]
            get_last_slots=_noop_get_slots,
        )
        assert result is None

    def test_plan_item_attribute_error_handled(self) -> None:
        """Item that raises on attribute access → treated as no-name → dropped."""
        class _BadItem:
            @property
            def item_name(self):
                raise AttributeError("boom")

            raw_span = ""
            quantity = 1

        plan = _Plan(items=(_BadItem(), _PlanItem(item_name="fries")))  # type: ignore[arg-type]
        coord = _FakeCoordinator()
        # Should not crash — bad item dropped, only 1 valid → None
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is None


# ---------------------------------------------------------------------------
# PE-16/17  Duck-typing
# ---------------------------------------------------------------------------


class TestDuckTyping:
    def test_smart_turn_item_like_object(self) -> None:
        """PE-16: SmartTurnItem-like — has item_name, quantity, size, modifiers, sides.
        items[0] → segment, items[1] → staged."""
        @dataclass
        class _SmartItem:
            item_name: str
            quantity: int = 1
            size: str = ""
            variant: str = ""
            modifiers: list = field(default_factory=list)
            sides: list = field(default_factory=list)

        plan = _Plan(items=(
            _SmartItem(item_name="fries", size="large"),
            _SmartItem(item_name="burger", quantity=2),
        ))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is not None
        segments = coord.calls[0]["segments"]
        # items[0]=fries → single segment
        assert len(segments) == 1
        assert segments[0].item_slot_value == "fries"
        # items[1]=burger → staged
        staged = coord.calls[0]["staged_items"]
        assert staged is not None and len(staged) == 1
        assert staged[0].item_name == "burger"
        assert staged[0].quantity == 2

    def test_gpt_validated_item_like_object(self) -> None:
        """PE-17: GptValidatedItem-like — has item_name, size_name, modifiers.
        items[0] → segment, items[1] → staged with variant_label."""
        @dataclass
        class _GptItem:
            item_name: str
            item_id: str = ""
            quantity: int = 1
            size_name: str = ""
            variant_name: str = ""
            modifiers: list = field(default_factory=list)
            sides: list = field(default_factory=list)

        plan = _Plan(items=(
            _GptItem(item_name="tuna melt", size_name="regular"),
            _GptItem(item_name="chicken wings", variant_name="6 piece"),
        ))
        coord = _FakeCoordinator()
        result = apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert result is not None
        # items[0]=tuna melt → segment
        segments = coord.calls[0]["segments"]
        assert segments[0].item_slot_value == "tuna melt"
        # items[1]=chicken wings → staged with variant_label from variant_name
        staged = coord.calls[0]["staged_items"]
        assert staged is not None
        assert staged[0].item_name == "chicken wings"
        assert staged[0].variant_label == "6 piece"


# ---------------------------------------------------------------------------
# PE-18  Coordinator receives correct context / get_last_slots
# ---------------------------------------------------------------------------


class TestContextPassing:
    def test_coordinator_receives_context(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="fries"),
        ))
        ctx = _FakeContext()
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, ctx, _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert coord.calls[0]["context"] is ctx

    def test_coordinator_receives_get_last_slots_callable(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="fries"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        assert coord.calls[0]["get_last_slots"] is _noop_get_slots


# ---------------------------------------------------------------------------
# PE-19  ITEM synthetic slot value matches item_name
# ---------------------------------------------------------------------------


class TestItemSlotValue:
    def test_item_slot_value_equals_item_name(self) -> None:
        """items[0] → first segment with correct item_slot_value.
        items[1] → staged item with correct item_name."""
        plan = _Plan(items=(
            _PlanItem(item_name="grilled chicken sandwich"),
            _PlanItem(item_name="large fries"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        # items[0] → first segment
        assert segments[0].item_slot_value == "grilled chicken sandwich"
        # items[1] → staged item
        staged = coord.calls[0]["staged_items"]
        assert staged[0].item_name == "large fries"

    def test_each_segment_has_item_sv(self) -> None:
        """First segment should have exactly 1 ITEM slot.
        Staged items carry item_name directly (no slots needed)."""
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="fries"),
            _PlanItem(item_name="coke"),
        ))
        coord = _FakeCoordinator()
        apply_multi_item_plan(
            plan, _FakeContext(), _make_repo(), coord,
            get_last_slots=_noop_get_slots,
        )
        segments = coord.calls[0]["segments"]
        # Only 1 segment (items[0])
        assert len(segments) == 1
        item_slots = [sv for sv in segments[0].slots if sv.name == "ITEM"]
        assert len(item_slots) == 1, (
            f"Expected 1 ITEM slot, got {len(item_slots)} for segment {segments[0].item_slot_value!r}"
        )
        # items[1] and items[2] → staged, each with item_name
        staged = coord.calls[0]["staged_items"]
        assert staged is not None and len(staged) == 2
        assert staged[0].item_name == "fries"
        assert staged[1].item_name == "coke"
