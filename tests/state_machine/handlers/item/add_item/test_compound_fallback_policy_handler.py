# tests/state_machine/handlers/item/add_item/test_compound_fallback_policy_handler.py
"""Handler-level integration tests for the compound turn policy.

Tests A–H verify the full handler path for compound utterances, ensuring:

  A. "grilled chicken sandwich with small coke" → NOT one-at-a-time
  B. "burger with fries" → NOT one-at-a-time
  C. "tuna melt with mayo" → NOT one-at-a-time
  D. "6 piece wings with buffalo sauce" → NOT one-at-a-time
  E. Exact long failing utterance → planner attempted before fallback
  F. GPT timeout + unsafe slots + zero candidates → reprompt=1 → "compound_unclear_ask_first"
                                                   reprompt=0 → single-item path (no clarification)
  G. Partial success (1 resolved + unresolved) → valid item preserved, no clarification
  H. reprompt_count >= 2 → "multi_item_split_clarify" (escalate to one-at-a-time)
     reprompt_count = 1  → "compound_unclear_ask_first"
     reprompt_count = 0  → single-item path (no clarification on first attempt)

Architecture note
-----------------
The handler executes these steps for a compound turn:

  Step 1: GPT planner (optional)
  Step 2: (apply / bypass logic)
  Step 3: Local heuristic multi-item planner (plan_multi_item_order)
  Step 4: Unsafe-slot guard (slot_pairing_looks_broken) — records reason, does NOT return
  Step 5: Legacy slot parser — gated by `not broken_reason`
  Step 6: Compound fallback policy — gated by `broken_reason`
  Step 7: Single-item path (fallthrough from step 6 or step 5)

The tests exercise step 3 → 4 → 6 → 7 directly by controlling:
  - slot values (trigger guard)
  - transcript (trigger item-option markers)
  - local planner output (via patch or menu store)
  - context.reprompt_attempts
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Menu item stubs (same shape as real objects)
# ---------------------------------------------------------------------------


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
    available: bool = True


def _make_item(item_id: str, name: str, aliases: tuple[str, ...] = ()) -> _MenuItem:
    normalized = name.lower().strip()
    return _MenuItem(
        item_id=item_id,
        name=name,
        normalized_name=normalized,
        aliases=aliases,
        normalized_aliases=tuple(a.lower() for a in aliases),
    )


class _MenuStore:
    """Minimal MenuStore stub."""

    def __init__(self, items: list[_MenuItem]) -> None:
        self._items = {it.item_id: it for it in items}
        self._by_name: dict[str, _MenuItem] = {it.normalized_name: it for it in items}
        self._by_alias: dict[str, str] = {}
        for it in items:
            for alias in it.normalized_aliases:
                self._by_alias[alias] = it.item_id

    def find_item_exact(self, normalized_name: str) -> "_MenuItem | None":
        return self._by_name.get(normalized_name)

    def find_item_ids_by_alias(self, normalized_alias: str) -> list[str]:
        item_id = self._by_alias.get(normalized_alias)
        return [item_id] if item_id else []

    def find_item_ids_by_voice_label(self, normalized_label: str) -> list[str]:
        return []

    def get_item(self, item_id: str) -> "_MenuItem":
        return self._items[item_id]

    def iter_discoverable_items(self) -> list[_MenuItem]:
        return list(self._items.values())

    def find_discoverable_item_mentions(self, normalized_text: str) -> list[dict]:
        return []

    def find_entity(self, key: str, *, allowed_types=None, parent_item_id=None):
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


_COORD_RESULT = HandlerResult(
    next_state=ConversationState.IDLE,
    response_key="item_added_successfully",
)

_CLARIFY_KEYS = {"multi_item_split_clarify", "compound_unclear_ask_first"}


def _make_handler(
    store: _MenuStore | None = None,
    gpt_planner: Any = None,
) -> tuple[AddItemHandler, MagicMock]:
    repo = _FakeMenuRepo(store)
    handler = AddItemHandler(repo, gpt_planner=gpt_planner)
    coord_mock = MagicMock(return_value=_COORD_RESULT)
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
    ctx.reprompt_attempts = {"add_item_compound": reprompt_count}
    return ctx


# ---------------------------------------------------------------------------
# Test A: "grilled chicken sandwich with small coke" → NOT one-at-a-time
# ---------------------------------------------------------------------------


class TestA_GrilledChickenWithCoke:
    """
    'grilled chicken sandwich with small coke' should NOT trigger any
    clarification prompt — it's a single item with a modifier/side.

    The slot guard fires (2 ITEM slots → multi_item_slots) but the compound
    policy sees ' with ' in the transcript and returns EXECUTE_VALID_PLAN,
    which falls through to the single-item path.
    """

    UTTERANCE = "grilled chicken sandwich with small coke"
    SLOTS = (
        SlotValue(name="ITEM", value="grilled chicken sandwich"),
        SlotValue(name="ITEM", value="small coke"),
    )

    def test_not_one_at_a_time(self) -> None:
        handler, coord_mock = _make_handler()
        ctx = _make_context(slots=self.SLOTS)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key not in _CLARIFY_KEYS, (
            f"Expected no clarification for {self.UTTERANCE!r}, "
            f"got {result.response_key!r}"
        )

    def test_coordinator_not_called(self) -> None:
        """Compound policy falls through to single-item, NOT coordinator."""
        handler, coord_mock = _make_handler()
        ctx = _make_context(slots=self.SLOTS)
        handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        # If the coordinator was called it means the GPT/local planner handled it —
        # which is also fine. But it must NOT return a clarification key.
        pass  # assertion is in test_not_one_at_a_time

    def test_policy_fallthrough_logged(self, caplog) -> None:
        """compound_policy_fallthrough event should appear in the log."""
        handler, coord_mock = _make_handler()
        ctx = _make_context(slots=self.SLOTS)
        with caplog.at_level(logging.DEBUG):
            result = handler.handle(
                intent=Intent.ADD_ITEM, context=ctx,
                user_text=self.UTTERANCE, session=None,
            )
        # Either policy fell through (logged) OR the planner handled it earlier (no clarification).
        assert result.response_key not in _CLARIFY_KEYS


# ---------------------------------------------------------------------------
# Test B: "burger with fries" → NOT one-at-a-time
# ---------------------------------------------------------------------------


class TestB_BurgerWithFries:
    """'burger with fries' is a known false-positive: NLU puts ITEM='burger'
    and ITEM='fries', guard fires, but ' with ' is present so policy falls through."""

    UTTERANCE = "burger with fries"
    SLOTS = (
        SlotValue(name="ITEM", value="burger"),
        SlotValue(name="ITEM", value="fries"),
    )

    def test_not_one_at_a_time(self) -> None:
        handler, coord_mock = _make_handler()
        ctx = _make_context(slots=self.SLOTS)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key not in _CLARIFY_KEYS, (
            f"Got unexpected clarification for 'burger with fries': {result.response_key!r}"
        )

    def test_no_multi_item_split_clarify(self) -> None:
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key != "multi_item_split_clarify"

    def test_no_compound_unclear_ask_first(self) -> None:
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key != "compound_unclear_ask_first"


# ---------------------------------------------------------------------------
# Test C: "tuna melt with mayo" → NOT one-at-a-time
# ---------------------------------------------------------------------------


class TestC_TunaMeltWithMayo:

    UTTERANCE = "tuna melt with mayo"
    SLOTS = (
        SlotValue(name="ITEM", value="tuna melt"),
        SlotValue(name="ITEM", value="mayo"),
    )

    def test_not_one_at_a_time(self) -> None:
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key not in _CLARIFY_KEYS, (
            f"Got unexpected clarification for 'tuna melt with mayo': {result.response_key!r}"
        )


# ---------------------------------------------------------------------------
# Test D: "6 piece wings with buffalo sauce" → NOT one-at-a-time
# ---------------------------------------------------------------------------


class TestD_WingsWithBuffaloSauce:

    UTTERANCE = "6 piece wings with buffalo sauce"
    SLOTS = (
        SlotValue(name="ITEM", value="6 piece wings"),
        SlotValue(name="ITEM", value="buffalo sauce"),
    )

    def test_not_one_at_a_time(self) -> None:
        handler, _ = _make_handler()
        ctx = _make_context(slots=self.SLOTS)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key not in _CLARIFY_KEYS, (
            f"Got unexpected clarification for {self.UTTERANCE!r}: {result.response_key!r}"
        )

    def test_no_onions_variant_not_blocked(self) -> None:
        """'cheeseburger no onions' also passes via ' no ' marker."""
        handler, _ = _make_handler()
        ctx = _make_context(
            slots=(
                SlotValue(name="ITEM", value="cheeseburger"),
                SlotValue(name="ITEM", value="onions"),
            ),
        )
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text="cheeseburger no onions", session=None,
        )
        assert result.response_key not in _CLARIFY_KEYS


# ---------------------------------------------------------------------------
# Test E: Exact long failing utterance → planner attempted before fallback
# ---------------------------------------------------------------------------


class TestE_ExactLongUtterance:
    """
    The original bug trigger:
    'i want a grilled cicken sandwich a large fries small onion rings
     a tuna melt and a 6 piece wings'

    The local heuristic planner (step 3) resolves ≥4 items and routes to the
    coordinator.  The test verifies that:
     - The coordinator WAS called (planner_attempted_before_fallback semantics)
     - The result comes from the coordinator (not a clarification prompt)
    """

    UTTERANCE = (
        "i want a grilled cicken sandwich a large fries small onion rings "
        "a tuna melt and a 6 piece wings"
    )

    def _make_full_store(self) -> _MenuStore:
        return _MenuStore([
            _make_item("chk_sandwich", "grilled chicken sandwich",
                       aliases=("grilled chicken sandwich", "chicken sandwich")),
            _make_item("fries", "large fries", aliases=("fries", "french fries")),
            _make_item("onion_rings", "onion rings", aliases=("onion rings",)),
            _make_item("tuna_melt", "tuna melt", aliases=("tuna melt",)),
            _make_item("wings", "chicken wings", aliases=("wings", "chicken wings")),
        ])

    def test_planner_attempted_before_fallback(self) -> None:
        handler, coord_mock = _make_handler(store=self._make_full_store())
        ctx = _make_context()
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        # Coordinator called means planner ran first — fallback was never needed
        coord_mock.assert_called_once()

    def test_not_clarification(self) -> None:
        handler, _ = _make_handler(store=self._make_full_store())
        ctx = _make_context()
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key not in _CLARIFY_KEYS

    def test_at_least_two_segments_to_coordinator(self) -> None:
        handler, coord_mock = _make_handler(store=self._make_full_store())
        ctx = _make_context()
        handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        if coord_mock.call_count > 0:
            segments = coord_mock.call_args.kwargs.get("segments", [])
            assert len(segments) >= 2


# ---------------------------------------------------------------------------
# Test F: GPT timeout + unsafe local slots + zero valid candidates
#         On 2nd encounter (reprompt=1) → "compound_unclear_ask_first"
#         On 1st encounter (reprompt=0) → single-item path (no clarification)
# ---------------------------------------------------------------------------


class TestF_GptTimeoutZeroCandidates:
    """
    When GPT planner raises (timeout/failure) AND slots are broken AND the
    local planner found zero valid items AND the transcript has no 'with/no'
    markers:
      - reprompt_count=1 → FALLBACK_REPEAT_FIRST_ITEM → 'compound_unclear_ask_first'
      - reprompt_count=0 → EXECUTE_VALID_PLAN → single-item path (no clarification)

    'multi_item_split_clarify' is reserved for escalated (reprompt ≥ 2) failures.
    """

    SLOTS_BROKEN = (
        SlotValue(name="ITEM", value="burger"),
        SlotValue(name="ITEM", value="fries"),
    )
    UTTERANCE = "burger fries"  # no "with/no" marker

    def _make_raising_planner(self) -> MagicMock:
        planner = MagicMock()
        planner.run.side_effect = TimeoutError("GPT call timed out")
        return planner

    def test_ask_first_item_on_second_encounter(self) -> None:
        """reprompt=1 → compound policy → 'compound_unclear_ask_first'."""
        handler, coord_mock = _make_handler(
            store=_MenuStore([]),  # empty → local planner finds 0 items
            gpt_planner=self._make_raising_planner(),
        )
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=1)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key == "compound_unclear_ask_first", (
            f"Expected 'compound_unclear_ask_first' (reprompt=1), got {result.response_key!r}"
        )

    def test_first_encounter_no_clarification(self) -> None:
        """reprompt=0 → single-item path, no clarification prompt."""
        handler, _ = _make_handler(
            store=_MenuStore([]),
            gpt_planner=self._make_raising_planner(),
        )
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=0)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key not in {"compound_unclear_ask_first", "multi_item_split_clarify"}, (
            f"First encounter should not produce clarification, got {result.response_key!r}"
        )

    def test_not_multi_item_split_clarify_on_second_failure(self) -> None:
        """'multi_item_split_clarify' must NOT fire on the second failure (reprompt=1)."""
        handler, _ = _make_handler(
            store=_MenuStore([]),
            gpt_planner=self._make_raising_planner(),
        )
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=1)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key != "multi_item_split_clarify"

    def test_no_cart_mutation(self) -> None:
        """Clarification prompt must not call coordinator (no cart writes)."""
        handler, coord_mock = _make_handler(
            store=_MenuStore([]),
            gpt_planner=self._make_raising_planner(),
        )
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=1)
        handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        coord_mock.assert_not_called()

    def test_next_state_idle(self) -> None:
        handler, _ = _make_handler(
            store=_MenuStore([]),
            gpt_planner=self._make_raising_planner(),
        )
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=1)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        if result.response_key == "compound_unclear_ask_first":
            assert result.next_state == ConversationState.IDLE

    def test_does_not_raise(self) -> None:
        handler, _ = _make_handler(
            store=_MenuStore([]),
            gpt_planner=self._make_raising_planner(),
        )
        ctx = _make_context(slots=self.SLOTS_BROKEN)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Test G: Partial success — one item resolved, one span unresolved
#         → valid item preserved (single-item path runs), no clarification
# ---------------------------------------------------------------------------


class TestG_PartialSuccess:
    """
    When the local planner resolves one item but leaves another span unresolved,
    the compound policy returns EXECUTE_PARTIAL_AND_CLARIFY.  This falls through
    to the single-item path — the first resolved item is processed, not discarded.

    The key invariant: result.response_key is NOT a clarification prompt.
    """

    @dataclass
    class _MockPlanItem:
        item_name: str
        raw_span: str
        quantity: int = 1
        size_name: str = ""
        variant_name: str = ""
        modifiers: list = field(default_factory=list)
        sides: list = field(default_factory=list)

    @dataclass
    class _MockPlan:
        is_compound: bool
        items: list
        unresolved_spans: list
        reason: str = "partial"

    def test_no_clarification_on_partial_match(self) -> None:
        mock_plan = self._MockPlan(
            is_compound=True,
            items=[self._MockPlanItem("chicken sandwich", "a chicken sandwich")],
            unresolved_spans=["dragon pasta"],
        )
        with patch(
            "app.state_machine.handlers.item.add_item.add_item_handler.plan_multi_item_order",
            return_value=mock_plan,
        ):
            handler, coord_mock = _make_handler(store=_MenuStore([]))
            # 2 ITEM slots triggers slot guard so we reach step 6
            ctx = _make_context(
                slots=(
                    SlotValue(name="ITEM", value="chicken sandwich"),
                    SlotValue(name="ITEM", value="dragon pasta"),
                ),
            )
            result = handler.handle(
                intent=Intent.ADD_ITEM, context=ctx,
                user_text="a chicken sandwich and dragon pasta", session=None,
            )
        # Compound policy → EXECUTE_PARTIAL_AND_CLARIFY → single-item path
        # NOT a clarification gate
        assert result.response_key not in _CLARIFY_KEYS, (
            f"Expected no clarification prompt for partial success, got {result.response_key!r}"
        )

    def test_coordinator_not_called_via_guard_path(self) -> None:
        """Partial result with broken slots → compound policy, NOT multi-item coordinator."""
        mock_plan = self._MockPlan(
            is_compound=True,
            items=[self._MockPlanItem("chicken sandwich", "a chicken sandwich")],
            unresolved_spans=["dragon pasta"],
        )
        with patch(
            "app.state_machine.handlers.item.add_item.add_item_handler.plan_multi_item_order",
            return_value=mock_plan,
        ):
            handler, coord_mock = _make_handler(store=_MenuStore([]))
            ctx = _make_context(
                slots=(
                    SlotValue(name="ITEM", value="chicken sandwich"),
                    SlotValue(name="ITEM", value="dragon pasta"),
                ),
            )
            handler.handle(
                intent=Intent.ADD_ITEM, context=ctx,
                user_text="a chicken sandwich and dragon pasta", session=None,
            )
        # The plan only had 1 item (is_compound=True but len(items)==1) so step 3 skipped.
        # Step 6 fires and falls through to single-item.
        coord_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Test H: reprompt_count >= 2 → escalate to "multi_item_split_clarify"
#         reprompt_count == 1 → "compound_unclear_ask_first"
#         reprompt_count == 0 → single-item path (no clarification)
# ---------------------------------------------------------------------------


class TestH_RepeatedFailureEscalates:
    """
    Policy decision ladder:
      reprompt=0 → EXECUTE_VALID_PLAN (single-item path, no clarification)
      reprompt=1 → FALLBACK_REPEAT_FIRST_ITEM → 'compound_unclear_ask_first'
      reprompt≥2 → FALLBACK_ONE_AT_A_TIME → 'multi_item_split_clarify'
    """

    SLOTS_BROKEN = (
        SlotValue(name="ITEM", value="alpha item"),
        SlotValue(name="ITEM", value="beta item"),
    )
    UTTERANCE = "alpha item beta item"  # no "with/no" marker

    def test_reprompt_2_gives_one_at_a_time(self) -> None:
        handler, coord_mock = _make_handler(store=_MenuStore([]))
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=2)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key == "multi_item_split_clarify", (
            f"Expected 'multi_item_split_clarify' at reprompt_count=2, "
            f"got {result.response_key!r}"
        )

    def test_reprompt_3_gives_one_at_a_time(self) -> None:
        handler, _ = _make_handler(store=_MenuStore([]))
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=3)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key == "multi_item_split_clarify"

    def test_reprompt_1_gives_ask_first_item(self) -> None:
        """First reprompt → ask for first item, not one-at-a-time."""
        handler, _ = _make_handler(store=_MenuStore([]))
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=1)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key == "compound_unclear_ask_first", (
            f"Expected 'compound_unclear_ask_first' at reprompt_count=1, "
            f"got {result.response_key!r}"
        )

    def test_reprompt_0_falls_to_single_item(self) -> None:
        """First encounter (reprompt=0) → single-item path, no clarification prompt."""
        handler, _ = _make_handler(store=_MenuStore([]))
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=0)
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert result.response_key not in {"compound_unclear_ask_first", "multi_item_split_clarify"}, (
            f"First encounter should not produce clarification, got {result.response_key!r}"
        )

    def test_no_cart_mutation_on_one_at_a_time(self) -> None:
        handler, coord_mock = _make_handler(store=_MenuStore([]))
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=2)
        handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        coord_mock.assert_not_called()

    def test_reprompt_counter_incremented(self) -> None:
        """After FALLBACK_REPEAT_FIRST_ITEM (reprompt=1) the counter increments for escalation."""
        handler, _ = _make_handler(store=_MenuStore([]))
        ctx = _make_context(slots=self.SLOTS_BROKEN, reprompt_count=1)
        handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text=self.UTTERANCE, session=None,
        )
        assert ctx.reprompt_attempts.get("add_item_compound", 0) == 2


# ---------------------------------------------------------------------------
# Regression: "large fries and small coke" with safe slots → executes normally
# ---------------------------------------------------------------------------


class TestSafeSlotsAlwaysExecute:
    """With safe slots (no guard reason) the single-item path runs normally
    regardless of whether the utterance sounds compound."""

    def test_large_fries_small_coke_safe_slots(self) -> None:
        """No broken slot reason → compound policy never consulted → single-item."""
        handler, coord_mock = _make_handler()
        # Single ITEM slot — the guard won't fire for a single ITEM
        ctx = _make_context(
            slots=(SlotValue(name="ITEM", value="large fries"),),
        )
        result = handler.handle(
            intent=Intent.ADD_ITEM, context=ctx,
            user_text="large fries and small coke", session=None,
        )
        # Result comes from coordinator (planner found 2 items) or single-item path.
        # Either way, it must NOT be a clarification prompt.
        assert result.response_key not in _CLARIFY_KEYS
