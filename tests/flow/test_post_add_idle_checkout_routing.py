# tests/flow/test_post_add_idle_checkout_routing.py
"""
Regression tests: checkout/done intents are allowed in IDLE state after item add.

Covers the scenario where a caller says "checkout", "that's it", or "no" after
the bot asks "Would you like anything else?" (last_response_key=item_added_successfully).

Spec:
1.  IDLE + non-empty cart + DENY intent           → confirm_order_summary (not intent_not_allowed)
2.  IDLE + non-empty cart + CHECKOUT intent        → confirm_order_summary
3.  IDLE + non-empty cart + CONFIRM_ORDER intent   → confirm_order_summary
4.  IDLE + non-empty cart + text "that's it" / CHECKOUT intent → confirm_order_summary
5.  IDLE + non-empty cart + ADD_ITEM intent        → continues add-item flow (not checkout)
6.  IDLE + non-empty cart + CANCEL_ORDER intent    → cancel path (not checkout)
7.  IDLE + empty cart + CHECKOUT intent            → idle_nothing_to_checkout
8.  IDLE + low-confidence "checkout" text (UNKNOWN) after item add → confirm_order_summary
"""
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_test_harness import (
    ScriptedTurn,
    StubCheckoutService,
    StubSmsService,
    build_engine,
    build_menu_repo,
    new_session,
    seed_cart_item,
    simulate_turn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _idle_session_with_item(*, last_key: str = "item_added_successfully"):
    session = new_session(
        state=ConversationState.IDLE,
        order_type="pickup",
    )
    session.last_response_key = last_key
    seed_cart_item(session, item_id="burger")
    return session


def _idle_session_empty(*, last_key: str = "item_added_successfully"):
    session = new_session(
        state=ConversationState.IDLE,
        order_type="pickup",
    )
    session.last_response_key = last_key
    return session


def _build_engine():
    return build_engine(
        menu_repo=build_menu_repo(),
        sms_service=StubSmsService(configured=True),
        checkout_service=StubCheckoutService(),
    )


# ---------------------------------------------------------------------------
# 1. DENY → order summary
# ---------------------------------------------------------------------------

def test_idle_deny_with_cart_routes_to_order_summary() -> None:
    engine = _build_engine()
    session = _idle_session_with_item()

    result = simulate_turn(engine, session, ScriptedTurn("no", intent=Intent.DENY))

    assert result.response_key == "confirm_order_summary", (
        f"Expected confirm_order_summary, got {result.response_key}"
    )
    assert session.conversation_state == ConversationState.CONFIRMING_ORDER


# ---------------------------------------------------------------------------
# 2. CHECKOUT → order summary
# ---------------------------------------------------------------------------

def test_idle_checkout_intent_with_cart_routes_to_order_summary() -> None:
    engine = _build_engine()
    session = _idle_session_with_item()

    result = simulate_turn(engine, session, ScriptedTurn("checkout", intent=Intent.CHECKOUT))

    assert result.response_key == "confirm_order_summary", (
        f"Expected confirm_order_summary, got {result.response_key}"
    )
    assert session.conversation_state == ConversationState.CONFIRMING_ORDER


# ---------------------------------------------------------------------------
# 3. CONFIRM_ORDER → order summary
# ---------------------------------------------------------------------------

def test_idle_confirm_order_intent_with_cart_routes_to_order_summary() -> None:
    engine = _build_engine()
    session = _idle_session_with_item()

    result = simulate_turn(engine, session, ScriptedTurn("that's it", intent=Intent.CONFIRM_ORDER))

    assert result.response_key == "confirm_order_summary", (
        f"Expected confirm_order_summary, got {result.response_key}"
    )
    assert session.conversation_state == ConversationState.CONFIRMING_ORDER


# ---------------------------------------------------------------------------
# 4. "That's it" phrase (with checkout intent) → order summary
# ---------------------------------------------------------------------------

def test_idle_thats_it_with_checkout_intent_routes_to_order_summary() -> None:
    engine = _build_engine()
    session = _idle_session_with_item()

    result = simulate_turn(engine, session, ScriptedTurn("that's it", intent=Intent.CHECKOUT))

    assert result.response_key == "confirm_order_summary", (
        f"Expected confirm_order_summary, got {result.response_key}"
    )
    assert session.conversation_state == ConversationState.CONFIRMING_ORDER


# ---------------------------------------------------------------------------
# 5. ADD_ITEM intent does not route to checkout
# ---------------------------------------------------------------------------

def test_idle_add_item_intent_does_not_route_to_checkout() -> None:
    engine = _build_engine()
    session = _idle_session_with_item()

    result = simulate_turn(engine, session, ScriptedTurn("add a coke", intent=Intent.ADD_ITEM))

    assert result.response_key != "confirm_order_summary"
    assert session.conversation_state != ConversationState.CONFIRMING_ORDER


# ---------------------------------------------------------------------------
# 6. CANCEL_ORDER does not route to checkout
# ---------------------------------------------------------------------------

def test_idle_cancel_order_does_not_route_to_checkout() -> None:
    engine = _build_engine()
    session = _idle_session_with_item()

    result = simulate_turn(engine, session, ScriptedTurn("cancel", intent=Intent.CANCEL_ORDER))

    assert result.response_key != "confirm_order_summary"
    assert session.conversation_state != ConversationState.CONFIRMING_ORDER


# ---------------------------------------------------------------------------
# 7. Empty cart + CHECKOUT → idle_nothing_to_checkout
# ---------------------------------------------------------------------------

def test_idle_checkout_intent_with_empty_cart_returns_nothing_to_checkout() -> None:
    engine = _build_engine()
    session = _idle_session_empty()

    result = simulate_turn(engine, session, ScriptedTurn("checkout", intent=Intent.CHECKOUT))

    assert result.response_key == "idle_nothing_to_checkout", (
        f"Expected idle_nothing_to_checkout, got {result.response_key}"
    )


# ---------------------------------------------------------------------------
# 8. Low-confidence "checkout" text (UNKNOWN) after item add → order summary
# ---------------------------------------------------------------------------

def test_idle_unknown_intent_checkout_text_after_add_routes_to_order_summary() -> None:
    """
    Simulates NLU returning low-confidence checkout intent → downgraded to UNKNOWN.
    The phrase 'checkout' is now in DONE_WORDS so the UNKNOWN shortcut fires.
    """
    engine = _build_engine()
    session = _idle_session_with_item(last_key="item_added_successfully")

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn("checkout", intent=Intent.UNKNOWN, confidence=0.99),
    )

    assert result.response_key == "confirm_order_summary", (
        f"Expected confirm_order_summary, got {result.response_key!r}. "
        "'checkout' must be in DONE_WORDS so UNKNOWN after item add routes to order summary."
    )
    assert session.conversation_state == ConversationState.CONFIRMING_ORDER


def test_idle_unknown_intent_check_out_text_after_add_routes_to_order_summary() -> None:
    """Same as above but for the two-word variant 'check out'."""
    engine = _build_engine()
    session = _idle_session_with_item(last_key="item_added_successfully")

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn("check out", intent=Intent.UNKNOWN, confidence=0.99),
    )

    assert result.response_key == "confirm_order_summary", (
        f"Expected confirm_order_summary, got {result.response_key!r}."
    )
    assert session.conversation_state == ConversationState.CONFIRMING_ORDER


def test_idle_unknown_intent_thats_it_text_after_add_routes_to_order_summary() -> None:
    """'That's it' as UNKNOWN intent after item add → DONE_WORDS phrase match → order summary."""
    engine = _build_engine()
    session = _idle_session_with_item(last_key="item_added_successfully")

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn("that's it", intent=Intent.UNKNOWN, confidence=0.99),
    )

    assert result.response_key == "confirm_order_summary", (
        f"Expected confirm_order_summary, got {result.response_key!r}."
    )
    assert session.conversation_state == ConversationState.CONFIRMING_ORDER
