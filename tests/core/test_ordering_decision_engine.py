# tests/core/test_ordering_decision_engine.py
"""Unit tests for OrderingDecisionEngine.

Coverage:
1.  Exact ITEM match → ADD_ITEM (exact_match)
2.  CATEGORY_SINGLE_ITEM → ADD_ITEM (category_single_item)
3.  CATEGORY → ASK_ITEM_CONFIRMATION (category_detected) with candidates
4.  AMBIGUOUS items → ASK_ITEM_CONFIRMATION (multiple_matches) with candidates
5.  AMBIGUOUS categories → matched_category_names populated
6.  NOT_FOUND, no near-miss → SUGGEST_ALTERNATIVES (LOW)
7.  NOT_FOUND, near-miss MEDIUM → ASK_ITEM_CONFIRMATION (near_miss_suggestion), tier=MEDIUM
8.  NOT_FOUND, near-miss HIGH  → ASK_ITEM_CONFIRMATION (near_miss_suggestion), tier=HIGH
9.  NOT_FOUND, attempt 1 → attempt=1 in decision
10. NOT_FOUND, attempt 2 → suggestions rotated (item[1:] picked)
11. NOT_FOUND, attempt 3 → ESCALATE_TO_AGENT
12. Suggestions capped at 4 (no overflow)
13. Post-add idle checkout intent with non-empty cart → CHECKOUT
14. Post-add idle checkout intent with empty cart   → UNCLEAR
15. Non-checkout intent from IDLE                   → None
16. Cancel intent from IDLE                         → None (CANCEL_ORDER not in checkout set)
17. Explicit quantity beats affirm intent (via WaitingForQuantityHandler — documented decision)
18. Normal add with no quantity → quantity field not set (engine does not default to 1)
"""
from __future__ import annotations

import pytest

from app.contracts.ordering_decision import OrderingAction, OrderingDecision
from app.core.ordering_decision_engine import OrderingDecisionEngine
from app.menu.models import MenuItem, Pricing
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.query_service import NearMissResult
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.conversation_context import ConversationContext


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_item(item_id: str, name: str) -> MenuItem:
    return MenuItem(
        item_id=item_id,
        name=name,
        normalized_name=normalize_text(name),
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=1000),
        side_groups=[],
        modifier_groups=[],
        available=True,
    )


class _StubMenuRepo:
    """Minimal stub — near_miss controlled per test."""

    def __init__(
        self,
        *,
        near_miss_item: MenuItem | None = None,
        near_miss_score: float = 5.5,
    ):
        self._near_miss_item = near_miss_item
        self._near_miss_score = near_miss_score

    def find_near_miss_item_normalized(
        self, normalized_text: str, *, threshold=None
    ) -> NearMissResult | None:
        if self._near_miss_item is None:
            return None
        return NearMissResult(item=self._near_miss_item, score=self._near_miss_score)

    def resolve_category_query_normalized(self, normalized_text: str, *, limit=5):
        return None


def _engine(near_miss_item=None, near_miss_score=5.5) -> OrderingDecisionEngine:
    return OrderingDecisionEngine(
        _StubMenuRepo(near_miss_item=near_miss_item, near_miss_score=near_miss_score)
    )


def _ctx() -> ConversationContext:
    return ConversationContext()


def _not_found_result(suggestions=None):
    items = [_make_item(f"s{i}", name) for i, name in enumerate(suggestions or [])]
    return MenuQueryResult(
        type=MenuQueryType.NOT_FOUND,
        suggested_items=items,
        suggested_categories=[],
    )


# ── 1. Exact ITEM match ────────────────────────────────────────────────────────

def test_exact_item_match_returns_add_item() -> None:
    item = _make_item("burger_1", "Classic Burger")
    result = MenuQueryResult(type=MenuQueryType.ITEM, item=item)
    decision = _engine().decide_from_menu_result(
        result=result,
        requested_item_text="classic burger",
        context=_ctx(),
        slots=[],
    )
    assert decision.action == OrderingAction.ADD_ITEM
    assert decision.reason == "exact_match"
    assert decision.item_id == "burger_1"
    assert decision.item_name == "Classic Burger"


# ── 2. Category with single item ──────────────────────────────────────────────

def test_category_single_item_returns_add_item() -> None:
    item = _make_item("burger_1", "Classic Burger")
    result = MenuQueryResult(type=MenuQueryType.CATEGORY_SINGLE_ITEM, items=[item])
    decision = _engine().decide_from_menu_result(
        result=result,
        requested_item_text="burger",
        context=_ctx(),
        slots=[],
    )
    assert decision.action == OrderingAction.ADD_ITEM
    assert decision.reason == "category_single_item"
    assert decision.item_id == "burger_1"


# ── 3. Category with multiple items → disambiguation ─────────────────────────

def test_category_returns_ask_confirmation_with_candidates() -> None:
    items = [_make_item("b1", "Classic Burger"), _make_item("b2", "Spicy Burger")]
    result = MenuQueryResult(
        type=MenuQueryType.CATEGORY,
        items=items,
        category_id="cat_burgers",
        category_name="Burgers",
    )
    decision = _engine().decide_from_menu_result(
        result=result,
        requested_item_text="burger",
        context=_ctx(),
        slots=[],
    )
    assert decision.action == OrderingAction.ASK_ITEM_CONFIRMATION
    assert decision.reason == "category_detected"
    assert decision.category_id == "cat_burgers"
    assert decision.category_name == "Burgers"
    assert len(decision.candidates) == 2
    assert decision.candidates[0].item_id == "b1"
    assert decision.candidates[1].item_id == "b2"
    assert decision.requires_confirmation is True


# ── 4. Ambiguous items ────────────────────────────────────────────────────────

def test_ambiguous_items_returns_ask_confirmation() -> None:
    items = [_make_item("b1", "Classic Burger"), _make_item("b2", "BBQ Burger")]
    result = MenuQueryResult(
        type=MenuQueryType.ITEM_AMBIGUOUS,
        matched_items=items,
    )
    decision = _engine().decide_from_menu_result(
        result=result,
        requested_item_text="burger",
        context=_ctx(),
        slots=[],
    )
    assert decision.action == OrderingAction.ASK_ITEM_CONFIRMATION
    assert decision.reason == "multiple_matches"
    assert len(decision.candidates) == 2
    candidate_ids = {c.item_id for c in decision.candidates}
    assert "b1" in candidate_ids and "b2" in candidate_ids


# ── 5. Ambiguous with category names ─────────────────────────────────────────

def test_ambiguous_category_names_populated() -> None:
    result = MenuQueryResult(
        type=MenuQueryType.CATEGORY_AMBIGUOUS,
        matched_items=[],
        matched_categories=[
            {"name": "Burgers", "category_id": "c1"},
            {"name": "Chicken", "category_id": "c2"},
        ],
    )
    decision = _engine().decide_from_menu_result(
        result=result, requested_item_text="burger", context=_ctx(), slots=[]
    )
    assert "Burgers" in decision.matched_category_names
    assert "Chicken" in decision.matched_category_names


# ── 6. NOT_FOUND, no near-miss → LOW / SUGGEST_ALTERNATIVES ──────────────────

def test_not_found_no_near_miss_returns_suggest_alternatives() -> None:
    result = _not_found_result(["Classic Burger", "Spicy Burger"])
    decision = _engine().decide_from_menu_result(
        result=result, requested_item_text="veggie pizza", context=_ctx(), slots=[]
    )
    assert decision.action == OrderingAction.SUGGEST_ALTERNATIVES
    assert decision.tier == "LOW"
    assert decision.reason == "no_match"
    assert "Classic Burger" in decision.suggestions


# ── 7. NOT_FOUND, MEDIUM near-miss ────────────────────────────────────────────

def test_near_miss_medium_routes_to_confirmation() -> None:
    near_miss = _make_item("b2", "Spicy Burger")
    decision = _engine(near_miss_item=near_miss, near_miss_score=5.5).decide_from_menu_result(
        result=_not_found_result(),
        requested_item_text="spicy chicken burger",
        context=_ctx(),
        slots=[],
    )
    assert decision.action == OrderingAction.ASK_ITEM_CONFIRMATION
    assert decision.reason == "near_miss_suggestion"
    assert decision.tier == "MEDIUM"
    assert decision.item_id == "b2"
    assert decision.item_name == "Spicy Burger"
    assert decision.requires_confirmation is True


# ── 8. NOT_FOUND, HIGH near-miss ──────────────────────────────────────────────

def test_near_miss_high_routes_to_confirmation() -> None:
    near_miss = _make_item("b2", "Spicy Burger")
    decision = _engine(near_miss_item=near_miss, near_miss_score=7.0).decide_from_menu_result(
        result=_not_found_result(),
        requested_item_text="spicy burger",
        context=_ctx(),
        slots=[],
    )
    assert decision.action == OrderingAction.ASK_ITEM_CONFIRMATION
    assert decision.tier == "HIGH"


# ── 9. NOT_FOUND attempt tracking ────────────────────────────────────────────

def test_first_not_found_has_attempt_1() -> None:
    ctx = _ctx()
    decision = _engine().decide_from_menu_result(
        result=_not_found_result(), requested_item_text="sushi", context=ctx, slots=[]
    )
    assert decision.attempt == 1


def test_second_not_found_has_attempt_2() -> None:
    ctx = _ctx()
    # Simulate first bump having been applied by executor
    ctx.bump_not_found(normalize_text("sushi"))
    decision = _engine().decide_from_menu_result(
        result=_not_found_result(), requested_item_text="sushi", context=ctx, slots=[]
    )
    assert decision.attempt == 2


# ── 10. Attempt-2 suggestion rotation ────────────────────────────────────────

def test_attempt_2_shifts_suggestions() -> None:
    ctx = _ctx()
    ctx.bump_not_found(normalize_text("veggie pizza"))  # simulate prior bump

    many = ["Pizza A", "Pizza B", "Pizza C", "Pizza D"]
    decision = _engine().decide_from_menu_result(
        result=_not_found_result(many), requested_item_text="veggie pizza", context=ctx, slots=[]
    )
    assert decision.attempt == 2
    # Attempt 2 shifts by 1 → "Pizza A" drops, "Pizza B" is now first
    assert "Pizza A" not in decision.suggestions
    assert "Pizza B" in decision.suggestions


# ── 11. Third failure → ESCALATE_TO_AGENT ────────────────────────────────────

def test_third_not_found_escalates() -> None:
    ctx = _ctx()
    ctx.bump_not_found(normalize_text("veggie pizza"))
    ctx.bump_not_found(normalize_text("veggie pizza"))

    decision = _engine().decide_from_menu_result(
        result=_not_found_result(), requested_item_text="veggie pizza", context=ctx, slots=[]
    )
    assert decision.action == OrderingAction.ESCALATE_TO_AGENT
    assert decision.reason == "repeated_not_found"
    assert decision.attempt == 3


# ── 12. Suggestions capped at 4 ──────────────────────────────────────────────

def test_suggestions_capped_at_four() -> None:
    many = ["A", "B", "C", "D", "E", "F"]
    decision = _engine().decide_from_menu_result(
        result=_not_found_result(many), requested_item_text="sushi", context=_ctx(), slots=[]
    )
    assert len(decision.suggestions) <= 4


# ── 13. Idle checkout with non-empty cart ────────────────────────────────────

@pytest.mark.parametrize("intent", [
    Intent.DENY,
    Intent.END_ADDING,
    Intent.FINISH_ORDER,
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
])
def test_idle_checkout_intent_non_empty_cart_returns_checkout(intent) -> None:
    decision = _engine().decide_from_idle_completion(
        intent=intent,
        cart_is_empty=False,
    )
    assert decision is not None
    assert decision.action == OrderingAction.CHECKOUT
    assert decision.reason == "cart_non_empty_done_signal"


# ── 14. Idle checkout with empty cart ────────────────────────────────────────

def test_idle_checkout_intent_empty_cart_returns_unclear() -> None:
    decision = _engine().decide_from_idle_completion(
        intent=Intent.FINISH_ORDER,
        cart_is_empty=True,
    )
    assert decision is not None
    assert decision.action == OrderingAction.UNCLEAR
    assert decision.handler_result_hint == "idle_nothing_to_checkout"


# ── 15. Non-checkout IDLE intent returns None ─────────────────────────────────

def test_non_checkout_intent_returns_none() -> None:
    decision = _engine().decide_from_idle_completion(
        intent=Intent.ADD_ITEM,
        cart_is_empty=False,
    )
    assert decision is None


# ── 16. CANCEL intent from IDLE is not checkout ───────────────────────────────

def test_cancel_intent_not_treated_as_checkout() -> None:
    decision = _engine().decide_from_idle_completion(
        intent=Intent.CANCEL_ORDER,
        cart_is_empty=False,
    )
    # CANCEL_ORDER must NOT route through checkout path
    assert decision is None


# ── 17. Engine purity: context not mutated by decide_from_menu_result ─────────

def test_engine_does_not_mutate_context() -> None:
    ctx = _ctx()
    initial_count = ctx.not_found_count(normalize_text("sushi"))

    _engine().decide_from_menu_result(
        result=_not_found_result(),
        requested_item_text="sushi",
        context=ctx,
        slots=[],
    )
    # Engine must NOT have bumped the counter — that's the executor's job.
    assert ctx.not_found_count(normalize_text("sushi")) == initial_count


# ── 18. Quantity field not set for plain ADD_ITEM (no default) ─────────────────

def test_add_item_decision_has_no_quantity_default() -> None:
    item = _make_item("b1", "Classic Burger")
    result = MenuQueryResult(type=MenuQueryType.ITEM, item=item)
    decision = _engine().decide_from_menu_result(
        result=result, requested_item_text="burger", context=_ctx(), slots=[]
    )
    assert decision.action == OrderingAction.ADD_ITEM
    # Quantity is not the engine's responsibility — PrefillOrchestrator owns it.
    assert decision.quantity is None
