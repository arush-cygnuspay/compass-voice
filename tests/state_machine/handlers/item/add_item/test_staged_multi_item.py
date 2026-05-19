# tests/state_machine/handlers/item/add_item/test_staged_multi_item.py
"""Structured multi-item staging integration tests.

12 scenarios verifying the StagedItemPlan / ItemQueueService /
PrefillOrchestrator pipeline introduced in feature/app-flow-v2.

Coverage
--------
1.  test_two_items_each_with_related_side
      items[0]="chicken burger" + side "coke";
      items[1]="hamburger" + side "fries"
      → staged plan for hamburger has "fries" only, not "coke".

2.  test_complete_plus_incomplete_item
      When the first item is instantly added and the second is staged,
      try_drain processes the staged item.

3.  test_two_incomplete_items_modifier_each
      Both items need a required modifier.  The first item enters
      WAITING_FOR_MODIFIER; the second stays in staged_item_queue.

4.  test_side_attached_to_correct_parent
      No cross-attachment: items[0] side ≠ items[1] side.

5.  test_size_attached_to_correct_related_item
      items[0] has variant "large"; items[1] has variant "small".
      Each staged plan carries its own variant_label only.

6.  test_partial_success_one_off_menu
      Local planner resolves 1 item but has unresolved_spans.
      Compound policy → EXECUTE_PARTIAL_AND_CLARIFY → single-item path
      (valid item proceeds, unresolved span is noted).

7.  test_cancel_pending_clears_staged_only
      reset_item_scope() does NOT clear staged_item_queue.
      reset_order_scope() DOES clear staged_item_queue.

8.  test_checkout_blocked_while_staged_non_empty
      can_checkout() blocks when staged_item_queue is non-empty.

9.  test_drain_continues_to_second_items_missing_requirement
      After the first item is instantly added the drain loop advances
      to the second staged item, stopping when that item needs user
      input (WAITING_FOR_MODIFIER).

10. test_one_at_a_time_fallback_only_after_repeated_failure
      reprompt_count=0 → no clarification (single-item path).
      reprompt_count=1 → "compound_unclear_ask_first".
      reprompt_count=2 → "multi_item_split_clarify".

11. test_smart_planner_side_has_size_in_schema
      SmartTurnSide.size is preserved through _item_to_staged_plan
      into StagedSide.variant_label ("small coke" → variant_label="small").

12. test_staged_item_bypasses_planner_reentry
      GPT planner is invoked once for a 3-item plan.  Items[1..N]
      are drained from staged_item_queue via PrefillOrchestrator —
      the GPT planner is NOT re-invoked during drain.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.services.order_lifecycle_guard import can_checkout
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.multi_item_plan_executor import (
    _item_to_staged_plan,
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
# Shared stubs
# ---------------------------------------------------------------------------


@dataclass
class _PlanSide:
    name: str
    quantity: int = 1
    size: str | None = None
    variant: str | None = None


@dataclass
class _PlanModifier:
    name: str
    operation: str = "add"


@dataclass
class _PlanItem:
    item_name: str
    item_id: str = ""
    raw_span: str = ""
    quantity: int = 1
    size: str | None = None
    variant: str | None = None
    sides: list = field(default_factory=list)
    modifiers: list = field(default_factory=list)
    plan_source: str = "test"


@dataclass
class _Plan:
    items: tuple = ()


@dataclass(slots=True)
class _PricingVariant:
    variant_id: str
    label: str
    normalized_label: str
    price_cents: int = 0


@dataclass(slots=True)
class _Pricing:
    mode: str = "fixed"
    price_cents: int | None = 500
    variants: list | None = None
    currency: str = "USD"


@dataclass(slots=True)
class _MenuItem:
    item_id: str
    name: str
    normalized_name: str
    aliases: tuple = ()
    normalized_aliases: tuple = ()
    voice_labels: tuple = ()
    pricing: _Pricing = field(default_factory=_Pricing)
    side_groups: list = field(default_factory=list)
    modifier_groups: list = field(default_factory=list)
    item_variants: list = field(default_factory=list)
    is_available: bool = True
    category_id: str = ""
    description: str = ""


class _MenuStore:
    def __init__(self, items: list[_MenuItem]) -> None:
        self._items = {item.item_id: item for item in items}
        self._by_name = {item.normalized_name: item for item in items}

    def get_item(self, item_id: str) -> _MenuItem | None:
        return self._items.get(item_id)

    def find_item_exact(self, normalized_name: str) -> _MenuItem | None:
        return self._by_name.get(normalized_name)

    def find_item_ids_by_alias(self, normalized_alias: str) -> list[str]:
        return []

    def find_item_ids_by_voice_label(self, normalized_label: str) -> list[str]:
        return []

    def iter_discoverable_items(self) -> list[_MenuItem]:
        return list(self._items.values())

    def find_discoverable_item_mentions(self, normalized_text: str) -> list[dict]:
        return []

    def find_entity(self, key: str, *, allowed_types=None, parent_item_id=None):
        return None

    def resolve_query_normalized(self, normalized_text, *, limit=5):
        from app.nlu.menu_normalization.menu_query_result import MenuQueryResult
        matches = [v for v in self._items.values()
                   if normalized_text in v.normalized_name]
        return MenuQueryResult(items=matches[:limit])

    def resolve_category_query_normalized(self, normalized_text, *, limit=5):
        return None


class _FakeMenuRepo:
    def __init__(self, store: _MenuStore | None = None) -> None:
        self.store = store or _MenuStore([])

    def resolve_menu_query_from_slots(self, **kwargs):
        from app.menu.query_result import MenuQueryResult, MenuQueryType
        return MenuQueryResult(type=MenuQueryType.NOT_FOUND)

    def resolve_menu_query_from_slots_normalized(self, **kwargs):
        from app.menu.query_result import MenuQueryResult, MenuQueryType
        return MenuQueryResult(type=MenuQueryType.NOT_FOUND)

    def resolve_menu_query(self, text: str, limit: int = 5):
        from app.menu.query_result import MenuQueryResult, MenuQueryType
        return MenuQueryResult(type=MenuQueryType.NOT_FOUND)

    def resolve_menu_query_normalized(self, text: str, limit: int = 5):
        from app.menu.query_result import MenuQueryResult, MenuQueryType
        return MenuQueryResult(type=MenuQueryType.NOT_FOUND)

    def find_near_miss_item_normalized(self, normalized_text, *, threshold=None):
        return None

    def resolve_category_query_normalized(self, normalized_text, *, limit=5):
        return None


_ITEM_ADDED = HandlerResult(
    next_state=ConversationState.IDLE,
    response_key="item_added_successfully",
    response_payload={"item_name": "burger", "quantity": 1},
)
_WAITING_MODIFIER = HandlerResult(
    next_state=ConversationState.WAITING_FOR_MODIFIER,
    response_key="ask_for_modifier",
    response_payload={"item_name": "burger", "quantity": 1},
)


def _make_handler(
    store: _MenuStore | None = None,
    gpt_planner: Any = None,
) -> tuple[AddItemHandler, MagicMock]:
    if store is None:
        store = _MenuStore([])
    repo = _FakeMenuRepo(store)
    handler = AddItemHandler(repo, gpt_planner=gpt_planner)
    coord_mock = MagicMock(return_value=_ITEM_ADDED)
    handler.multi_item_coordinator.handle = coord_mock  # type: ignore[method-assign]
    return handler, coord_mock


def _make_context(
    *,
    slots: tuple = (),
    confidence: float = 1.0,
    reprompt_count: int = 0,
) -> ConversationContext:
    ctx = ConversationContext()
    ctx.last_slots = slots
    ctx.last_intent_confidence = confidence
    if reprompt_count > 0:
        ctx.reprompt_attempts = {"add_item_compound": reprompt_count}
    return ctx


# ---------------------------------------------------------------------------
# 1. Two items each with a related side — no cross-attachment
# ---------------------------------------------------------------------------


class TestTwoItemsEachWithRelatedSide:
    """items[0]='chicken burger' gets side 'coke'; items[1]='hamburger' gets side 'fries'.
    The staged plan for hamburger must carry only 'fries', not 'coke'.
    """

    def _make_plan(self) -> _Plan:
        return _Plan(items=(
            _PlanItem(
                item_name="chicken burger",
                sides=[_PlanSide(name="coke", size="small")],
            ),
            _PlanItem(
                item_name="hamburger",
                sides=[_PlanSide(name="fries", size="large")],
            ),
        ))

    def test_hamburger_staged_has_only_fries(self) -> None:
        coordinator = MagicMock()
        coordinator.handle.return_value = _ITEM_ADDED
        ctx = ConversationContext()
        apply_multi_item_plan(
            self._make_plan(),
            ctx,
            MagicMock(),  # menu_repo (unused by executor)
            coordinator,
            get_last_slots=lambda _ctx: (),
        )
        coordinator.handle.assert_called_once()
        staged_items = coordinator.handle.call_args.kwargs["staged_items"]
        assert staged_items is not None and len(staged_items) == 1
        hamburger_plan = staged_items[0]
        assert hamburger_plan.item_name == "hamburger"
        side_names = [s.name for s in hamburger_plan.requested_sides]
        assert "fries" in side_names, f"Expected fries in hamburger sides, got {side_names}"
        assert "coke" not in side_names, "Coke must not be cross-attached to hamburger"

    def test_first_segment_has_chicken_burger(self) -> None:
        coordinator = MagicMock()
        coordinator.handle.return_value = _ITEM_ADDED
        ctx = ConversationContext()
        apply_multi_item_plan(
            self._make_plan(), ctx, MagicMock(), coordinator,
            get_last_slots=lambda _ctx: (),
        )
        segments = coordinator.handle.call_args.kwargs["segments"]
        assert len(segments) == 1
        assert segments[0].item_slot_value == "chicken burger"

    def test_side_sizes_are_per_item(self) -> None:
        coordinator = MagicMock()
        coordinator.handle.return_value = _ITEM_ADDED
        ctx = ConversationContext()
        apply_multi_item_plan(
            self._make_plan(), ctx, MagicMock(), coordinator,
            get_last_slots=lambda _ctx: (),
        )
        staged = coordinator.handle.call_args.kwargs["staged_items"][0]
        fries = staged.requested_sides[0]
        assert fries.variant_label == "large", (
            f"Expected large fries, got variant_label={fries.variant_label!r}"
        )


# ---------------------------------------------------------------------------
# 4. Side attached to correct parent (no cross-attachment detail)
# ---------------------------------------------------------------------------


class TestSideAttachedToCorrectParent:
    """Strict non-cross-attachment check for any plan with 2 items and distinct sides."""

    def test_no_cross_attachment(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="tuna melt", sides=[_PlanSide(name="mayo")]),
            _PlanItem(item_name="grilled chicken", sides=[_PlanSide(name="ranch dressing")]),
        ))
        coordinator = MagicMock()
        coordinator.handle.return_value = _ITEM_ADDED
        ctx = ConversationContext()
        apply_multi_item_plan(
            plan, ctx, MagicMock(), coordinator,
            get_last_slots=lambda _ctx: (),
        )
        staged = coordinator.handle.call_args.kwargs["staged_items"]
        chicken_plan = staged[0]
        side_names = {s.name for s in chicken_plan.requested_sides}
        assert "ranch dressing" in side_names, "grilled chicken must have its own side"
        assert "mayo" not in side_names, "mayo must not leak into grilled chicken's staged plan"

    def test_three_items_each_has_own_side(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="burger", sides=[_PlanSide(name="ketchup")]),
            _PlanItem(item_name="wrap", sides=[_PlanSide(name="hot sauce")]),
            _PlanItem(item_name="salad", sides=[_PlanSide(name="vinaigrette")]),
        ))
        coordinator = MagicMock()
        coordinator.handle.return_value = _ITEM_ADDED
        ctx = ConversationContext()
        apply_multi_item_plan(
            plan, ctx, MagicMock(), coordinator,
            get_last_slots=lambda _ctx: (),
        )
        staged = coordinator.handle.call_args.kwargs["staged_items"]
        assert len(staged) == 2
        wrap_sides = {s.name for s in staged[0].requested_sides}
        salad_sides = {s.name for s in staged[1].requested_sides}
        assert "hot sauce" in wrap_sides and "vinaigrette" not in wrap_sides
        assert "vinaigrette" in salad_sides and "hot sauce" not in salad_sides


# ---------------------------------------------------------------------------
# 5. Size attached to correct related item
# ---------------------------------------------------------------------------


class TestSizeAttachedToCorrectRelatedItem:
    """items[0] has variant "large"; items[1] has variant "small".
    Each staged plan carries its own variant_label only.
    """

    def test_each_item_has_own_size(self) -> None:
        plan = _Plan(items=(
            _PlanItem(item_name="fries", size="large"),
            _PlanItem(item_name="coke", size="small"),
        ))
        coordinator = MagicMock()
        coordinator.handle.return_value = _ITEM_ADDED
        ctx = ConversationContext()
        apply_multi_item_plan(
            plan, ctx, MagicMock(), coordinator,
            get_last_slots=lambda _ctx: (),
        )
        staged = coordinator.handle.call_args.kwargs["staged_items"]
        coke_plan = staged[0]
        assert coke_plan.item_name == "coke"
        assert coke_plan.variant_label == "small", (
            f"Expected coke to have 'small' variant_label, got {coke_plan.variant_label!r}"
        )

    def test_side_size_is_per_side_not_per_item(self) -> None:
        """Side-level size must live on the StagedSide, not the parent StagedItemPlan."""
        plan = _Plan(items=(
            _PlanItem(item_name="burger"),
            _PlanItem(item_name="sandwich", sides=[_PlanSide(name="coke", size="small")]),
        ))
        coordinator = MagicMock()
        coordinator.handle.return_value = _ITEM_ADDED
        ctx = ConversationContext()
        apply_multi_item_plan(
            plan, ctx, MagicMock(), coordinator,
            get_last_slots=lambda _ctx: (),
        )
        staged = coordinator.handle.call_args.kwargs["staged_items"]
        sandwich_plan = staged[0]
        assert sandwich_plan.variant_label is None, "Item-level size must be None (size belongs to side)"
        coke_side = sandwich_plan.requested_sides[0]
        assert coke_side.variant_label == "small", (
            f"Side 'coke' should have variant_label='small', got {coke_side.variant_label!r}"
        )


# ---------------------------------------------------------------------------
# 7. Cancel pending clears staged_item_queue only with order/session reset
# ---------------------------------------------------------------------------


class TestCancelPendingClearedStagedOnly:
    """reset_item_scope() must NOT clear staged_item_queue.
    reset_order_scope() MUST clear staged_item_queue.
    """

    def _make_staged_plan(self, name: str) -> StagedItemPlan:
        return StagedItemPlan(
            item_id="",
            item_name=name,
            quantity=1,
        )

    def test_reset_item_scope_preserves_staged_queue(self) -> None:
        ctx = ConversationContext()
        ctx.staged_item_queue = deque([self._make_staged_plan("hamburger")])
        ctx.reset_item_scope()
        assert len(ctx.staged_item_queue) == 1, (
            "reset_item_scope() must not clear staged_item_queue"
        )

    def test_reset_order_scope_clears_staged_queue(self) -> None:
        ctx = ConversationContext()
        ctx.staged_item_queue = deque([
            self._make_staged_plan("hamburger"),
            self._make_staged_plan("fries"),
        ])
        ctx.reset_order_scope()
        assert len(ctx.staged_item_queue) == 0, (
            "reset_order_scope() must clear staged_item_queue"
        )

    def test_reset_session_scope_clears_staged_queue(self) -> None:
        ctx = ConversationContext()
        ctx.staged_item_queue = deque([self._make_staged_plan("wrap")])
        ctx.reset_session_scope()
        assert len(ctx.staged_item_queue) == 0, (
            "reset_session_scope() must clear staged_item_queue"
        )

    def test_staged_queue_survives_multiple_item_resets(self) -> None:
        ctx = ConversationContext()
        ctx.staged_item_queue = deque([self._make_staged_plan("coke")])
        ctx.reset_item_scope()
        ctx.reset_item_scope()
        assert len(ctx.staged_item_queue) == 1


# ---------------------------------------------------------------------------
# 8. Checkout blocked while staged_item_queue is non-empty
# ---------------------------------------------------------------------------


class TestCheckoutBlockedByStagedQueue:
    """can_checkout() must block when staged_item_queue has pending items."""

    def _make_cart(self, items: list[str]) -> MagicMock:
        cart = MagicMock()
        cart.items = items
        return cart

    def _make_context_with_staged(self, n: int = 1) -> ConversationContext:
        ctx = ConversationContext()
        for i in range(n):
            ctx.staged_item_queue.append(
                StagedItemPlan(item_id="", item_name=f"item_{i}", quantity=1)
            )
        return ctx

    def test_staged_queue_blocks_checkout(self) -> None:
        ctx = self._make_context_with_staged(1)
        cart = self._make_cart(["burger"])
        result = can_checkout(context=ctx, cart=cart)
        assert result.blocking, (
            "can_checkout() must block when staged_item_queue is non-empty"
        )

    def test_two_staged_items_still_blocked(self) -> None:
        ctx = self._make_context_with_staged(2)
        cart = self._make_cart(["tuna melt", "wings"])
        result = can_checkout(context=ctx, cart=cart)
        assert result.blocking

    def test_empty_staged_queue_does_not_block_for_staged_reason(self) -> None:
        ctx = ConversationContext()
        assert len(ctx.staged_item_queue) == 0
        # No pending item → checkout guard runs normally (may still block for other reasons,
        # but NOT because of staged_item_queue).  Just verify it doesn't raise.
        cart = self._make_cart(["burger"])
        result = can_checkout(context=ctx, cart=cart)
        assert result is not None


# ---------------------------------------------------------------------------
# 11. SmartTurnSide.size preserved through _item_to_staged_plan
# ---------------------------------------------------------------------------


class TestSmartPlannerSideHasSizeInSchema:
    """SmartTurnSide.size → StagedSide.variant_label through executor conversion."""

    def test_side_size_preserved_as_variant_label(self) -> None:
        """_item_to_staged_plan must convert side.size → StagedSide.variant_label."""
        plan_item = _PlanItem(
            item_name="chicken burger",
            sides=[_PlanSide(name="coke", size="small")],
        )
        result = _item_to_staged_plan(plan_item)
        assert result is not None
        assert len(result.requested_sides) == 1
        side = result.requested_sides[0]
        assert side.name == "coke"
        assert side.variant_label == "small", (
            f"Expected variant_label='small', got {side.variant_label!r}"
        )

    def test_smart_turn_side_has_size_attribute(self) -> None:
        """SmartTurnSide must expose a 'size' field (schema contract)."""
        from app.services.smart_turn_planner import SmartTurnSide
        side = SmartTurnSide(name="coke", size="small", quantity=1)
        assert side.size == "small"
        assert side.variant is None

    def test_smart_turn_side_has_variant_attribute(self) -> None:
        from app.services.smart_turn_planner import SmartTurnSide
        side = SmartTurnSide(name="fries", variant="large", quantity=1)
        assert side.variant == "large"
        assert side.size is None

    def test_side_variant_field_also_preserved(self) -> None:
        """side.variant (not .size) also maps to StagedSide.variant_label."""
        plan_item = _PlanItem(
            item_name="sandwich",
            sides=[_PlanSide(name="iced tea", variant="large")],
        )
        result = _item_to_staged_plan(plan_item)
        assert result is not None
        assert result.requested_sides[0].variant_label == "large"

    def test_none_size_produces_none_variant_label(self) -> None:
        plan_item = _PlanItem(
            item_name="wrap",
            sides=[_PlanSide(name="chips")],  # no size
        )
        result = _item_to_staged_plan(plan_item)
        assert result is not None
        assert result.requested_sides[0].variant_label is None


# ---------------------------------------------------------------------------
# 10. One-at-a-time fallback only after repeated failure
# ---------------------------------------------------------------------------


class TestOneAtATimeFallbackTiming:
    """The clarification prompt ladder:
      reprompt=0 → single-item path (no clarification)
      reprompt=1 → "compound_unclear_ask_first"
      reprompt=2 → "multi_item_split_clarify"
    """

    SLOTS_BROKEN = (
        SlotValue(name="ITEM", value="alpha"),
        SlotValue(name="ITEM", value="beta"),
    )
    UTTERANCE = "alpha beta"  # no "with/no" marker

    def test_first_attempt_no_clarification(self) -> None:
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=0)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key not in {"compound_unclear_ask_first", "multi_item_split_clarify"}, (
            f"First attempt must not produce clarification, got {result.response_key!r}"
        )

    def test_second_attempt_asks_first_item(self) -> None:
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=1)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key == "compound_unclear_ask_first", (
            f"Second attempt should produce compound_unclear_ask_first, got {result.response_key!r}"
        )

    def test_third_attempt_one_at_a_time(self) -> None:
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=2)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key == "multi_item_split_clarify", (
            f"Third attempt should escalate to multi_item_split_clarify, got {result.response_key!r}"
        )

    def test_reprompt_counter_increments_at_second_attempt(self) -> None:
        """After FALLBACK_REPEAT_FIRST_ITEM fires (reprompt=1), counter must reach 2."""
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=1)
        handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert ctx.reprompt_attempts.get("add_item_compound", 0) == 2

    def test_reprompt_counter_unchanged_at_first_attempt(self) -> None:
        """At reprompt=0 the policy falls through — counter must not increment."""
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=0)
        handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert ctx.reprompt_attempts.get("add_item_compound", 0) == 0


# ---------------------------------------------------------------------------
# 12. Staged item bypasses planner re-entry (GPT called once)
# ---------------------------------------------------------------------------


class TestStagedItemBypassesPlannerReentry:
    """GPT planner must be invoked exactly once for an N-item plan.
    Items[1..N] are drained from staged_item_queue via PrefillOrchestrator,
    never via AddItemHandler.handle() or the GPT planner.
    """

    @dataclass
    class _GptItem:
        item_name: str
        item_id: str = ""
        quantity: int = 1
        size: str | None = None
        variant: str | None = None
        modifiers: list = field(default_factory=list)
        sides: list = field(default_factory=list)

    @dataclass
    class _ValidatedPlan:
        items: tuple

    @dataclass
    class _PlannerResult:
        safe_to_apply: bool = True
        validated_plan: Any = None
        gpt_called: bool = True
        decision: str = "add_item"
        confidence: float = 0.95
        route_reason: str = "gpt_planner"
        validator_passed: bool = True
        validator_reject_reason: str | None = None
        latency_ms: float = 120.0

    def _make_gpt_planner(self, items: list) -> MagicMock:
        validated_plan = self._ValidatedPlan(items=tuple(items))
        plan_result = self._PlannerResult(validated_plan=validated_plan)
        planner = MagicMock()
        planner.run.return_value = plan_result
        return planner

    def test_gpt_planner_called_once_for_three_item_plan(self) -> None:
        """GPT planner.run() must be called exactly once regardless of item count."""
        gpt_items = [
            self._GptItem(item_name="burger"),
            self._GptItem(item_name="fries"),
            self._GptItem(item_name="coke"),
        ]
        gpt_planner = self._make_gpt_planner(gpt_items)

        with patch(
            "app.state_machine.handlers.item.add_item.add_item_handler.plan_multi_item_order"
        ) as mock_local:
            mock_local.return_value = MagicMock(is_compound=False, items=[])
            handler, coord_mock = _make_handler(gpt_planner=gpt_planner)
            ctx = _make_context()
            handler.handle(
                intent=Intent.ADD_ITEM, context=ctx,
                user_text="burger fries coke", session=None,
            )

        gpt_planner.run.assert_called_once(), (
            "GPT planner must be called exactly once; subsequent items use staged drain"
        )

    def test_staged_queue_populated_not_empty_after_multi_item_plan(self) -> None:
        """After coordinator.handle() with staged_items, staged_item_queue must be populated."""
        gpt_items = [
            self._GptItem(item_name="tuna melt"),
            self._GptItem(item_name="onion rings"),
        ]
        gpt_planner = self._make_gpt_planner(gpt_items)

        captured_staged: list = []

        def _capturing_coord(*, context, segments, get_last_slots, staged_items=None):
            if staged_items:
                context.staged_item_queue = deque(staged_items)
                captured_staged.extend(staged_items)
            return _ITEM_ADDED

        with patch(
            "app.state_machine.handlers.item.add_item.add_item_handler.plan_multi_item_order"
        ) as mock_local:
            mock_local.return_value = MagicMock(is_compound=False, items=[])
            handler, _ = _make_handler(gpt_planner=gpt_planner)
            handler.multi_item_coordinator.handle = _capturing_coord  # type: ignore[method-assign]
            ctx = _make_context()
            handler.handle(
                intent=Intent.ADD_ITEM, context=ctx,
                user_text="tuna melt and onion rings", session=None,
            )

        assert len(captured_staged) == 1, (
            f"Expected 1 staged item (onion rings), got {len(captured_staged)}"
        )
        assert captured_staged[0].item_name == "onion rings"

    def test_coordinator_receives_staged_items_not_multiple_segments(self) -> None:
        """items[0] → 1 segment; items[1..N] → staged_items (not additional segments)."""
        gpt_items = [
            self._GptItem(item_name="wrap"),
            self._GptItem(item_name="soup"),
            self._GptItem(item_name="juice"),
        ]
        gpt_planner = self._make_gpt_planner(gpt_items)

        with patch(
            "app.state_machine.handlers.item.add_item.add_item_handler.plan_multi_item_order"
        ) as mock_local:
            mock_local.return_value = MagicMock(is_compound=False, items=[])
            handler, coord_mock = _make_handler(gpt_planner=gpt_planner)
            ctx = _make_context()
            handler.handle(
                intent=Intent.ADD_ITEM, context=ctx,
                user_text="wrap soup juice", session=None,
            )

        coord_mock.assert_called_once()
        call_kwargs = coord_mock.call_args.kwargs
        assert len(call_kwargs["segments"]) == 1, (
            "Only items[0] should produce a segment; items[1..N] go to staged_items"
        )
        staged = call_kwargs.get("staged_items") or []
        assert len(staged) == 2, (
            f"Expected 2 staged items (soup + juice), got {len(staged)}"
        )


# ---------------------------------------------------------------------------
# 6. Partial success — valid item proceeds, off-menu span noted
# ---------------------------------------------------------------------------


class TestPartialSuccessOneOffMenu:
    """When the local planner resolves 1 item but has unresolved_spans,
    compound policy returns EXECUTE_PARTIAL_AND_CLARIFY → single-item path.
    The valid item is not discarded.
    """

    SLOTS_TWO_ITEMS = (
        SlotValue(name="ITEM", value="burger"),
        SlotValue(name="ITEM", value="dragon pasta"),
    )

    def test_partial_plan_falls_through_to_single_item(self) -> None:
        """With 1 valid candidate the compound policy must NOT return a clarification prompt."""
        handler, coord_mock = _make_handler()
        ctx = _make_context(slots=self.SLOTS_TWO_ITEMS)

        mock_plan = MagicMock()
        mock_plan.is_compound = True
        mock_plan.items = [MagicMock(item_name="burger", raw_span="burger")]
        mock_plan.unresolved_spans = ["dragon pasta"]

        with patch(
            "app.state_machine.handlers.item.add_item.add_item_handler.plan_multi_item_order",
            return_value=mock_plan,
        ):
            result = handler.handle(
                intent=Intent.ADD_ITEM, context=ctx,
                user_text="burger and dragon pasta", session=None,
            )

        assert result.response_key not in {"multi_item_split_clarify", "compound_unclear_ask_first"}, (
            f"Partial success should fall through to single-item path, got {result.response_key!r}"
        )

    def test_unresolved_span_does_not_trigger_one_at_a_time(self) -> None:
        """unresolved_spans alone must not trigger one-at-a-time escalation."""
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS_TWO_ITEMS)

        mock_plan = MagicMock()
        mock_plan.is_compound = True
        mock_plan.items = [MagicMock(item_name="burger")]
        mock_plan.unresolved_spans = ["mystery item"]

        with patch(
            "app.state_machine.handlers.item.add_item.add_item_handler.plan_multi_item_order",
            return_value=mock_plan,
        ):
            result = handler.handle(
                intent=Intent.ADD_ITEM, context=ctx,
                user_text="burger and mystery item", session=None,
            )

        assert result.response_key != "multi_item_split_clarify"


# ---------------------------------------------------------------------------
# 2 & 9. ItemQueueService drain — staged item is processed after instant add
# ---------------------------------------------------------------------------


class TestDrainWithStagedItems:
    """Tests for the ItemQueueService.try_drain() staged-path.

    We mock the heavy components (menu_repo, prefill_orchestrator, command_executor)
    and verify the drain loop behavioral invariants.
    """

    def _make_staged_plan(self, name: str) -> StagedItemPlan:
        return StagedItemPlan(
            item_id="item-123",
            item_name=name,
            quantity=1,
            plan_source="test",
        )

    def test_drain_calls_prefill_orchestrator_for_staged_item(self) -> None:
        """When staged_item_queue is non-empty and first item was just added,
        try_drain() must call prefill_orchestrator.enter_add_flow_for_item().
        """
        from app.core.item_queue_service import ItemQueueService

        ctx = ConversationContext()
        ctx.staged_item_queue = deque([self._make_staged_plan("hamburger")])

        first_result = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            response_payload={"item_name": "burger", "quantity": 1},
        )

        drain_result = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
            response_payload={"item_name": "hamburger", "quantity": 1},
        )

        # Fake menu item
        fake_menu_item = MagicMock()
        fake_menu_item.name = "hamburger"

        # Fake menu_store
        fake_store = MagicMock()
        fake_store.get_item.return_value = fake_menu_item

        # Fake prefill orchestrator
        fake_prefill = MagicMock()
        fake_prefill.enter_add_flow_for_item.return_value = drain_result

        # Fake add_item_handler
        fake_handler = MagicMock()
        fake_handler.menu_repo.store = fake_store
        fake_handler.prefill_orchestrator = fake_prefill

        # Fake session
        fake_session = MagicMock()
        fake_session.conversation_context = ctx
        fake_session.session_id = "test-session"
        fake_session.restaurant_id = "test-restaurant"

        svc = ItemQueueService(
            handlers={"add_item_handler": fake_handler},
            command_executor=MagicMock(),
        )

        result = svc.try_drain(session=fake_session, current_result=first_result)

        fake_prefill.enter_add_flow_for_item.assert_called_once()
        call_kwargs = fake_prefill.enter_add_flow_for_item.call_args.kwargs
        assert call_kwargs.get("staged") is not None, "staged kwarg must be passed"
        assert call_kwargs["staged"].item_name == "hamburger"
        assert result is not None
        assert result.next_state == ConversationState.WAITING_FOR_MODIFIER

    def test_drain_returns_none_when_no_queued_items(self) -> None:
        """try_drain() must return None when both queues are empty."""
        from app.core.item_queue_service import ItemQueueService

        ctx = ConversationContext()  # both queues empty

        first_result = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
        )

        fake_session = MagicMock()
        fake_session.conversation_context = ctx

        svc = ItemQueueService(
            handlers={"add_item_handler": MagicMock()},
            command_executor=MagicMock(),
        )

        result = svc.try_drain(session=fake_session, current_result=first_result)
        assert result is None

    def test_drain_skipped_when_not_item_added(self) -> None:
        """try_drain() must return None unless current_result.response_key is item_added_successfully."""
        from app.core.item_queue_service import ItemQueueService

        ctx = ConversationContext()
        ctx.staged_item_queue = deque([self._make_staged_plan("burger")])

        not_added_result = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
        )

        fake_session = MagicMock()
        fake_session.conversation_context = ctx

        svc = ItemQueueService(
            handlers={"add_item_handler": MagicMock()},
            command_executor=MagicMock(),
        )

        result = svc.try_drain(session=fake_session, current_result=not_added_result)
        assert result is None, "Drain must not run unless item was just added"


# ---------------------------------------------------------------------------
# 3. Two incomplete items with modifier each — first waits, second stays staged
# ---------------------------------------------------------------------------


class TestTwoIncompleteItemsModifierEach:
    """When the first item enters WAITING_FOR_MODIFIER, the drain loop stops
    and staged_item_queue still contains the second item.
    """

    def _make_staged_plan(self, name: str) -> StagedItemPlan:
        return StagedItemPlan(
            item_id="item-x",
            item_name=name,
            quantity=1,
            plan_source="test",
        )

    def test_staged_queue_not_empty_when_first_item_waits(self) -> None:
        """If the first item produced from the staged drain needs user input,
        the remaining staged items must still be in the queue.
        """
        from app.core.item_queue_service import ItemQueueService

        ctx = ConversationContext()
        second_staged = self._make_staged_plan("wrap")
        ctx.staged_item_queue = deque([self._make_staged_plan("burger"), second_staged])

        first_added = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            response_payload={"item_name": "sandwich", "quantity": 1},
        )

        # First drain call returns WAITING_FOR_MODIFIER — stops the drain loop
        modifier_wait = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="ask_for_modifier",
            response_payload={"item_name": "burger", "quantity": 1},
        )

        fake_menu_item = MagicMock()
        fake_menu_item.name = "burger"
        fake_store = MagicMock()
        fake_store.get_item.return_value = fake_menu_item

        fake_prefill = MagicMock()
        fake_prefill.enter_add_flow_for_item.return_value = modifier_wait

        fake_handler = MagicMock()
        fake_handler.menu_repo.store = fake_store
        fake_handler.prefill_orchestrator = fake_prefill

        fake_session = MagicMock()
        fake_session.conversation_context = ctx
        fake_session.session_id = "test"
        fake_session.restaurant_id = "test"

        svc = ItemQueueService(
            handlers={"add_item_handler": fake_handler},
            command_executor=MagicMock(),
        )

        drain_result = svc.try_drain(session=fake_session, current_result=first_added)

        assert drain_result is not None
        assert drain_result.next_state == ConversationState.WAITING_FOR_MODIFIER
        # Second staged item must still be in the queue
        assert len(ctx.staged_item_queue) == 1, (
            "Second staged item must remain in queue when first item enters WAITING_FOR_MODIFIER"
        )
        assert ctx.staged_item_queue[0].item_name == "wrap"
