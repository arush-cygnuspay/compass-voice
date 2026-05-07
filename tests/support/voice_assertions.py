# tests/support/voice_assertions.py
"""Reusable, harness-agnostic assertion helpers for system-level voice tests.

Goal: keep regression tests short and intent-focused. All helpers accept
either a SimulatedTurn (from voice_test_harness.simulate_turn) or a TurnOutput
(from engine.process_turn) and pull what they need.
"""
from __future__ import annotations

from typing import Iterable

from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_test_harness import (
    SimulatedTurn,
    response_text,
)


# ─── response-key / state helpers ────────────────────────────────────────────

def assert_response_key(turn, *expected: str) -> None:
    """Pass if turn.response_key matches at least one of the expected keys."""
    actual = getattr(turn, "response_key", None)
    assert actual in expected, (
        f"expected response_key in {expected!r}, got {actual!r} "
        f"(text={response_text(turn)!r})"
    )


def assert_state(session: Session, *expected: ConversationState) -> None:
    actual = session.conversation_state
    assert actual in expected, (
        f"expected state in {[e.value for e in expected]!r}, got {actual.value!r}"
    )


def assert_response_mentions(turn, *phrases: str) -> None:
    text = response_text(turn).lower()
    missing = [p for p in phrases if p.lower() not in text]
    assert not missing, f"response missing phrases {missing!r}; got {text!r}"


def assert_response_not_mentions(turn, *phrases: str) -> None:
    text = response_text(turn).lower()
    leaked = [p for p in phrases if p.lower() in text]
    assert not leaked, f"response leaked phrases {leaked!r}; got {text!r}"


# ─── cart helpers ────────────────────────────────────────────────────────────

def cart_items(session: Session) -> list:
    """Return the underlying CartItem list — defensive against API drift."""
    cart = session.cart
    for attr in ("items", "_items", "lines"):
        v = getattr(cart, attr, None)
        if isinstance(v, list):
            return v
    if hasattr(cart, "to_dict"):
        d = cart.to_dict()
        if isinstance(d, dict) and isinstance(d.get("items"), list):
            return d["items"]
    return []


def assert_cart_size(session: Session, expected: int) -> None:
    items = cart_items(session)
    assert len(items) == expected, f"expected {expected} cart items, got {len(items)}: {items!r}"


def assert_cart_contains(session: Session, item_substring: str) -> None:
    items = cart_items(session)
    haystack = " | ".join(_describe_item(it) for it in items).lower()
    assert item_substring.lower() in haystack, (
        f"cart does not contain {item_substring!r}; have: {haystack!r}"
    )


def assert_no_duplicate_items(session: Session) -> None:
    items = cart_items(session)
    keys = [(_get(it, "item_id"), _get(it, "quantity"), tuple(sorted((_get(it, "modifiers") or {}).items()))) for it in items]
    # Same identity tuple appearing twice = duplicate add (idempotency violation)
    seen = {}
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    assert not dupes, f"cart has duplicate add-item events: {dupes!r}"


# ─── reprompt / counter helpers ──────────────────────────────────────────────

def assert_reprompt_capped(session: Session, field: str, max_attempts: int) -> None:
    count = int(session.reprompt_count_by_field.get(field, 0) or 0)
    assert count <= max_attempts, (
        f"reprompt_count_by_field[{field!r}]={count} exceeded cap {max_attempts}"
    )


def assert_unknown_intent_not_looping(turns: Iterable, *, max_same_key: int = 3) -> None:
    """Fail if the SAME response_key fires more than max_same_key times in a row."""
    last = None
    streak = 0
    for t in turns:
        k = getattr(t, "response_key", None)
        if k == last:
            streak += 1
        else:
            last = k
            streak = 1
        assert streak <= max_same_key, (
            f"same response_key {k!r} fired {streak} times in a row — escalation policy missing"
        )


# ─── private utilities ───────────────────────────────────────────────────────

def _get(obj, attr, default=None):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def _describe_item(it) -> str:
    name = _get(it, "name") or _get(it, "item_name") or _get(it, "item_id") or "?"
    qty = _get(it, "quantity") or 1
    return f"{qty}x {name}"
