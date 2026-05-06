# tests/state_machine/handlers/item/test_item_not_found_flow.py
"""
Confidence-tiered item_not_found flow tests.

Coverage:
1. Low-confidence NOT_FOUND → IDLE + short unavailability message + alternatives
2. Near-miss NOT_FOUND → CONFIRMING_ITEM + "Did you mean X?"
3. Near-miss AFFIRM → enter add flow (attempt counter reset)
4. Near-miss DENY → IDLE + alternatives (attempt bumped)
5. Three failed attempts → escalation ("I don't have that item...")
6. Attempt counter survives reset_item_scope (not cleared until reset_order_scope)
7. Response text: max 2 sentences, no "and N more", correct wording
8. item_not_found backward-compat: legacy suggested_item_names payload still renders
"""
from __future__ import annotations

from app.menu.models import MenuItem, Pricing
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.query_service import NearMissResult
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.responses.item_responses import (
    item_not_found,
    item_not_found_near_miss,
    item_not_found_escalation,
)
from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
from app.state_machine.handlers.item.add_item.item_resolution_handler import ItemResolutionHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ── Stubs ─────────────────────────────────────────────────────────────────────

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
    """Minimal repo stub — near_miss controlled per test."""

    def __init__(
        self,
        *,
        near_miss_item: MenuItem | None = None,
        near_miss_score: float = 5.5,
    ):
        self._near_miss_item = near_miss_item
        self._near_miss_score = near_miss_score
        self._items = {
            "burger_1": _make_item("burger_1", "Classic Burger"),
            "burger_2": _make_item("burger_2", "Spicy Burger"),
        }

    def get_item(self, item_id: str) -> MenuItem:
        return self._items[item_id]

    def find_near_miss_item_normalized(
        self, normalized_text: str, *, threshold=None
    ) -> NearMissResult | None:
        if self._near_miss_item is None:
            return None
        return NearMissResult(item=self._near_miss_item, score=self._near_miss_score)

    def resolve_menu_query(self, text, *, limit=5):
        return MenuQueryResult(type=MenuQueryType.NOT_FOUND)

    def resolve_menu_query_from_slots(self, *, user_text, slots, fallback_to_text=True, limit=5):
        return MenuQueryResult(type=MenuQueryType.NOT_FOUND)

    def resolve_item_within_candidates_normalized(self, normalized_text, candidate_item_ids):
        return None

    def resolve_menu_query_normalized(self, normalized_text, *, limit=5):
        return MenuQueryResult(type=MenuQueryType.NOT_FOUND)

    def resolve_category_query_normalized(self, normalized_text, *, limit=5):
        return None


class _StubPrefillOrchestrator:
    def enter_add_flow_for_item(self, *, context, item, user_text, slots):
        from app.state_machine.handler_result import HandlerResult
        context.current_item_id = item.item_id
        context.current_item_name = item.name
        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            response_payload={"item_name": item.name, "quantity": 1},
            reset_context=True,
        )


class _Session:
    conversation_state = ConversationState.CONFIRMING_ITEM


def _not_found_result(suggestions: list[str] | None = None) -> MenuQueryResult:
    items = [_make_item(f"sug_{i}", name) for i, name in enumerate(suggestions or [])]
    return MenuQueryResult(
        type=MenuQueryType.NOT_FOUND,
        suggested_items=items,
        suggested_categories=[],
    )


def _resolve(handler: ItemResolutionHandler, context: ConversationContext,
             query: str, result: MenuQueryResult) -> object:
    return handler.resolve_item_and_enter_flow(
        context=context,
        result=result,
        requested_item_text=query,
        original_user_text=query,
        slots=[],
    )


def _confirm(handler: ConfirmingHandler, context: ConversationContext,
             user_text: str, intent=Intent.UNKNOWN) -> object:
    return handler.handle(
        intent=intent,
        context=context,
        user_text=user_text,
        session=_Session(),
    )


# ── 1. Low-confidence NOT_FOUND → IDLE + alternatives ─────────────────────────

def test_low_confidence_not_found_goes_to_idle() -> None:
    repo = _StubMenuRepo(near_miss_item=None)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    result = _resolve(handler, ctx, "veggie pizza", _not_found_result(["Hawaiian Pizza"]))

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_not_found"


def test_low_confidence_response_has_alternatives() -> None:
    repo = _StubMenuRepo(near_miss_item=None)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    result = _resolve(handler, ctx, "veggie pizza",
                      _not_found_result(["Hawaiian Pizza", "All Meat Pizza"]))

    payload = result.response_payload or {}
    assert "veggie pizza" == payload.get("query")
    suggestions = payload.get("suggestions") or []
    assert "Hawaiian Pizza" in suggestions


def test_low_confidence_response_text_is_concise() -> None:
    text = item_not_found(None, None, {
        "query": "veggie pizza",
        "suggestions": ["Hawaiian Pizza", "All Meat Pizza", "BBQ Chicken Pizza"],
    })
    assert "veggie pizza" in text
    assert "Which one would you like?" in text
    # No "and N more" overflow text
    assert "more" not in text.lower()


def test_low_confidence_no_suggestions_short_response() -> None:
    text = item_not_found(None, None, {"query": "sushi"})
    assert "sushi" in text
    assert len(text.split(".")) <= 3  # at most 2 sentences


# ── 2. Near-miss → CONFIRMING_ITEM ────────────────────────────────────────────

def test_near_miss_routes_to_confirming_item() -> None:
    near_miss = _make_item("burger_2", "Spicy Burger")
    repo = _StubMenuRepo(near_miss_item=near_miss)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    result = _resolve(handler, ctx, "spicy chicken burger",
                      _not_found_result(["Spicy Burger"]))

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert result.response_key == "item_not_found_near_miss"


def test_near_miss_response_text() -> None:
    text = item_not_found_near_miss(None, None, {"item_name": "Spicy Burger"})
    assert text == "Did you mean Spicy Burger?"


def test_near_miss_stores_confirmation_context() -> None:
    near_miss = _make_item("burger_2", "Spicy Burger")
    repo = _StubMenuRepo(near_miss_item=near_miss)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    _resolve(handler, ctx, "spicy chicken burger", _not_found_result())

    conf = ctx.awaiting_confirmation_for
    assert conf is not None
    assert conf["reason"] == "near_miss_suggestion"
    assert conf["value_id"] == "burger_2"


# ── 3. Near-miss AFFIRM → enter add flow ──────────────────────────────────────

def test_near_miss_affirm_enters_add_flow() -> None:
    near_miss = _make_item("burger_2", "Spicy Burger")
    repo = _StubMenuRepo(near_miss_item=near_miss)
    resolution_handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    confirm_handler = ConfirmingHandler(repo)
    ctx = ConversationContext()

    _resolve(resolution_handler, ctx, "spicy chicken burger", _not_found_result())
    result = _confirm(confirm_handler, ctx, "yes", intent=Intent.CONFIRM)

    # User confirmed — we must have left the near-miss prompt (exact next state
    # depends on item's quantity/sides/modifier config, not tested here).
    assert result.next_state != ConversationState.CONFIRMING_ITEM
    assert result.response_key != "item_not_found_near_miss"


def test_near_miss_affirm_resets_not_found_counter() -> None:
    near_miss = _make_item("burger_2", "Spicy Burger")
    repo = _StubMenuRepo(near_miss_item=near_miss)
    resolution_handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    confirm_handler = ConfirmingHandler(repo)
    ctx = ConversationContext()

    _resolve(resolution_handler, ctx, "spicy chicken burger", _not_found_result())
    _confirm(confirm_handler, ctx, "yes", intent=Intent.CONFIRM)

    assert ctx.not_found_count(normalize_text("spicy chicken burger")) == 0


# ── 4. Near-miss DENY → IDLE + alternatives ───────────────────────────────────

def test_near_miss_deny_returns_to_idle() -> None:
    near_miss = _make_item("burger_2", "Spicy Burger")
    repo = _StubMenuRepo(near_miss_item=near_miss)
    resolution_handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    confirm_handler = ConfirmingHandler(repo)
    ctx = ConversationContext()

    _resolve(resolution_handler, ctx, "spicy chicken burger",
             _not_found_result(["Spicy Burger", "Classic Burger"]))
    result = _confirm(confirm_handler, ctx, "no", intent=Intent.UNKNOWN)

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_not_found"


def test_near_miss_deny_bumps_attempt_counter() -> None:
    near_miss = _make_item("burger_2", "Spicy Burger")
    repo = _StubMenuRepo(near_miss_item=near_miss)
    resolution_handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    confirm_handler = ConfirmingHandler(repo)
    ctx = ConversationContext()

    _resolve(resolution_handler, ctx, "spicy chicken burger", _not_found_result())
    # After resolution: count=1.  After DENY: count=2.
    _confirm(confirm_handler, ctx, "no", intent=Intent.UNKNOWN)

    assert ctx.not_found_count(normalize_text("spicy chicken burger")) == 2


# ── 5. Three failures → escalation ────────────────────────────────────────────

def test_third_attempt_escalates() -> None:
    repo = _StubMenuRepo(near_miss_item=None)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    _resolve(handler, ctx, "veggie pizza", _not_found_result())
    _resolve(handler, ctx, "veggie pizza", _not_found_result())
    result = _resolve(handler, ctx, "veggie pizza", _not_found_result())

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_not_found_escalation"


def test_escalation_response_text() -> None:
    text = item_not_found_escalation(None, None, {})
    assert "I don't have that item" in text
    assert "connect you" in text
    assert len(text.split(".")) <= 3  # max 2 sentences


def test_near_miss_deny_on_second_attempt_escalates_on_third() -> None:
    near_miss = _make_item("burger_2", "Spicy Burger")
    repo = _StubMenuRepo(near_miss_item=near_miss)
    resolution_handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    confirm_handler = ConfirmingHandler(repo)
    ctx = ConversationContext()

    # Attempt 1: near_miss → CONFIRMING_ITEM
    _resolve(resolution_handler, ctx, "spicy chicken burger", _not_found_result())
    # DENY → attempt 2 → IDLE
    _confirm(confirm_handler, ctx, "no", intent=Intent.UNKNOWN)

    # Attempt 3: item_resolution_handler → escalate
    repo2 = _StubMenuRepo(near_miss_item=None)
    resolution_handler2 = ItemResolutionHandler(repo2, _StubPrefillOrchestrator())
    result = _resolve(resolution_handler2, ctx, "spicy chicken burger", _not_found_result())

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_not_found_escalation"


# ── 6. Attempt counter survives reset_item_scope ──────────────────────────────

def test_attempt_counter_survives_reset_item_scope() -> None:
    repo = _StubMenuRepo(near_miss_item=None)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    _resolve(handler, ctx, "veggie pizza", _not_found_result())
    # reset_item_scope is called inside resolve on low-confidence path
    # Counter should still be 1
    assert ctx.not_found_count(normalize_text("veggie pizza")) == 1


def test_attempt_counter_cleared_by_reset_order_scope() -> None:
    ctx = ConversationContext()
    ctx.bump_not_found("veggie pizza")
    ctx.bump_not_found("veggie pizza")

    ctx.reset_order_scope()

    assert ctx.not_found_count("veggie pizza") == 0


# ── 7. Attempt 2 suggestion rotation ──────────────────────────────────────────

def test_attempt_2_rotates_suggestions() -> None:
    repo = _StubMenuRepo(near_miss_item=None)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    many = ["Pizza A", "Pizza B", "Pizza C", "Pizza D"]
    result1 = _resolve(handler, ctx, "veggie pizza", _not_found_result(many))
    result2 = _resolve(handler, ctx, "veggie pizza", _not_found_result(many))

    sugg1 = set(result1.response_payload.get("suggestions") or [])
    sugg2 = set(result2.response_payload.get("suggestions") or [])
    # Second attempt shifts by 1, so top item from attempt 1 not in attempt 2
    assert "Pizza A" not in sugg2 or sugg1 != sugg2


# ── 8. Backward-compat: legacy payload format ─────────────────────────────────

def test_legacy_suggested_item_names_still_rendered() -> None:
    text = item_not_found(None, None, {
        "query": "sushi",
        "suggested_item_names": ["Classic Burger", "Spicy Burger"],
    })
    assert "Classic Burger" in text or "Spicy Burger" in text
    assert "sushi" in text.lower()


def test_legacy_suggested_category_names_rendered() -> None:
    text = item_not_found(None, None, {
        "query": "sushi",
        "suggested_category_names": ["Burgers", "Drinks"],
    })
    assert "Burgers" in text or "Drinks" in text


# ── 9. Response length / format constraints ───────────────────────────────────

def test_item_not_found_no_and_n_more() -> None:
    text = item_not_found(None, None, {
        "query": "sushi",
        "suggestions": ["A", "B", "C", "D", "E"],  # 5 items — capped at 4
    })
    assert "more" not in text.lower()


def test_near_miss_response_is_one_sentence() -> None:
    text = item_not_found_near_miss(None, None, {"item_name": "Classic Burger"})
    # Single question
    assert text.count("?") == 1
    assert "." not in text  # no period in a yes/no question


# ── 10. Confidence tier ───────────────────────────────────────────────────────

def test_near_miss_result_medium_tier() -> None:
    near_miss = _make_item("burger_2", "Spicy Burger")
    repo = _StubMenuRepo(near_miss_item=near_miss, near_miss_score=5.5)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    result = _resolve(handler, ctx, "spicy chicken burger", _not_found_result())

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert ctx.awaiting_confirmation_for["tier"] == "MEDIUM"
    assert result.response_payload.get("tier") == "MEDIUM"


def test_near_miss_result_high_tier() -> None:
    near_miss = _make_item("burger_2", "Spicy Burger")
    repo = _StubMenuRepo(near_miss_item=near_miss, near_miss_score=7.0)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    result = _resolve(handler, ctx, "spicy burger", _not_found_result())

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert ctx.awaiting_confirmation_for["tier"] == "HIGH"
    assert result.response_payload.get("tier") == "HIGH"


def test_near_miss_result_tier_boundary() -> None:
    from app.menu.scorer import HIGH_MATCH_THRESHOLD, NEAR_MISS_THRESHOLD
    near_miss = _make_item("burger_1", "Classic Burger")

    # Exactly at HIGH_MATCH_THRESHOLD → HIGH
    r_high = NearMissResult(item=near_miss, score=HIGH_MATCH_THRESHOLD)
    assert r_high.tier == "HIGH"

    # Just below HIGH_MATCH_THRESHOLD → MEDIUM
    r_medium = NearMissResult(item=near_miss, score=HIGH_MATCH_THRESHOLD - 0.1)
    assert r_medium.tier == "MEDIUM"

    # At NEAR_MISS_THRESHOLD (minimum) → MEDIUM
    r_floor = NearMissResult(item=near_miss, score=NEAR_MISS_THRESHOLD)
    assert r_floor.tier == "MEDIUM"


# ── 11. LOW confidence does not say "Did you mean X?" ────────────────────────

def test_low_confidence_does_not_say_did_you_mean() -> None:
    text = item_not_found(None, None, {
        "query": "veggie pizza",
        "suggestions": ["Hawaiian Pizza", "BBQ Chicken Pizza"],
    })
    assert "Did you mean" not in text


# ── 12. Customizable item alone → "can be customized" response ───────────────

def test_customizable_only_item_triggers_customizable_response() -> None:
    text = item_not_found(None, None, {
        "query": "make my own burger",
        "suggestions": ["Build Your Own Burger"],
    })
    assert "can be customized" in text
    assert "Build Your Own Burger" in text
    assert "Which one would you like?" not in text


def test_byo_abbreviation_triggers_customizable_response() -> None:
    text = item_not_found(None, None, {
        "query": "byo pizza",
        "suggestions": ["BYO Pizza"],
    })
    assert "can be customized" in text


# ── 13. Mixed customizable + regular → standard list format ─────────────────

def test_mixed_customizable_and_regular_uses_standard_format() -> None:
    text = item_not_found(None, None, {
        "query": "custom pizza",
        "suggestions": ["Build Your Own Pizza", "Hawaiian Pizza", "BBQ Pizza"],
    })
    assert "Which one would you like?" in text
    assert "can be customized" not in text


# ── 14. Escalation includes "to the restaurant" ──────────────────────────────

def test_escalation_mentions_restaurant() -> None:
    text = item_not_found_escalation(None, None, {})
    assert "restaurant" in text.lower()


# ── 15. No response exceeds 2 sentences ──────────────────────────────────────

def test_item_not_found_max_two_sentences_with_suggestions() -> None:
    text = item_not_found(None, None, {
        "query": "sushi",
        "suggestions": ["Classic Burger", "Spicy Burger", "BBQ Burger"],
    })
    sentence_count = text.count(".") + text.count("?") + text.count("!")
    assert sentence_count <= 2


def test_item_not_found_max_two_sentences_no_suggestions() -> None:
    text = item_not_found(None, None, {"query": "unicorn salad"})
    sentence_count = text.count(".") + text.count("?") + text.count("!")
    assert sentence_count <= 2


def test_escalation_max_two_sentences() -> None:
    text = item_not_found_escalation(None, None, {})
    sentence_count = text.count(".") + text.count("?") + text.count("!")
    assert sentence_count <= 2


# ── 16. Same-category preference via stub ─────────────────────────────────────

class _StubMenuRepoWithCategory(_StubMenuRepo):
    """Extends the base stub so resolve_category_query_normalized returns pizza items."""

    def resolve_category_query_normalized(self, normalized_text, *, limit=5):
        if "pizza" in normalized_text:
            from app.menu.query_result import MenuQueryResult, MenuQueryType
            items = [
                _make_item("pz1", "Hawaiian Pizza"),
                _make_item("pz2", "BBQ Chicken Pizza"),
                _make_item("pz3", "Pepperoni Pizza"),
            ]
            return MenuQueryResult(
                type=MenuQueryType.CATEGORY,
                items=items,
                category_id="cat_pizza",
                category_name="Pizzas",
            )
        return None


def test_category_suggestions_preferred_over_general() -> None:
    repo = _StubMenuRepoWithCategory(near_miss_item=None)
    handler = ItemResolutionHandler(repo, _StubPrefillOrchestrator())
    ctx = ConversationContext()

    # General suggestions do NOT include pizza items; category lookup should win.
    result = _resolve(handler, ctx, "veggie pizza",
                      _not_found_result(["Classic Burger", "Spicy Burger"]))

    suggestions = result.response_payload.get("suggestions") or []
    assert any("Pizza" in s for s in suggestions), (
        f"Expected pizza category items in suggestions, got: {suggestions}"
    )
