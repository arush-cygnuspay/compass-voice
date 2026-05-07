# tests/state_machine/policy/test_idle_checkout_anti_regression.py
"""Anti-regression tests: coercion must NOT fire in non-IDLE states.

Phase 7 of the idle-checkout coercion spec.

Validates that WAITING_FOR_SIDE, WAITING_FOR_MODIFIER, WAITING_FOR_SIZE,
WAITING_FOR_SIDE_SIZE, WAITING_FOR_QUANTITY, CONFIRMING_ORDER, and other
non-IDLE states are completely unaffected by IdleCheckoutCoercionPolicy.
These states have their own handlers and must not see their UNKNOWN intents
silently promoted to CHECKOUT.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.policy.idle_checkout_coercion import coerce_idle_to_checkout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nlu(text: str) -> SimpleNamespace:
    return SimpleNamespace(normalized_text=text, intent_confidence=0.0)


def _cart(empty: bool = False):
    c = SimpleNamespace()
    c.is_empty = lambda: empty
    return c


def _coerce(text: str, state: ConversationState, intent: Intent = Intent.UNKNOWN):
    return coerce_idle_to_checkout(
        state=state,
        intent_result=IntentResult(intent=intent, raw_text=text),
        nlu=_nlu(text),
        cart=_cart(empty=False),
    )


# ---------------------------------------------------------------------------
# Non-IDLE waiting states — complete checkout phrase catalog
# ---------------------------------------------------------------------------

_CHECKOUT_PHRASES = [
    "done",
    "finished",
    "thats it",
    "thats all",
    "nothing else",
    "no more",
    "checkout",
    "check out",
    "payment",
    "pay",
    "send payment link",
    "place the order",
    "finalize",
    "complete my order",
]

_NON_IDLE_STATES = [
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
    ConversationState.CONFIRMING_ORDER,
    ConversationState.WAITING_FOR_PAYMENT,
    ConversationState.CONFIRMING_ITEM,
    ConversationState.MODIFYING_ITEM,
    ConversationState.GREETING,
    ConversationState.COMPLETED,
]


class TestNonIdleStatesNeverCoerced:
    @pytest.mark.parametrize("state", _NON_IDLE_STATES)
    @pytest.mark.parametrize("phrase", _CHECKOUT_PHRASES)
    def test_phrase_not_coerced_in_non_idle_state(self, state, phrase):
        result = _coerce(phrase, state=state)
        assert result.coercion_reason is None, (
            f"'{phrase}' in state '{state.value}' must not be coerced but got "
            f"reason={result.coercion_reason}"
        )
        assert result.intent_result.intent == Intent.UNKNOWN, (
            f"Intent must remain UNKNOWN in state '{state.value}', "
            f"got {result.intent_result.intent}"
        )


# ---------------------------------------------------------------------------
# Non-UNKNOWN intents in IDLE are not coerced (Rule 4)
# ---------------------------------------------------------------------------

class TestNonUnknownIntentsInIdleNotCoerced:
    @pytest.mark.parametrize("intent,phrase", [
        (Intent.ADD_ITEM, "done"),
        (Intent.REMOVE_ITEM, "checkout"),
        (Intent.MODIFY_ITEM, "payment"),
        (Intent.ASK_ITEM_INFO, "finalize"),
        (Intent.AFFIRM, "done"),
        (Intent.CANCEL, "checkout"),
        (Intent.GREETING, "thats all"),
    ])
    def test_resolved_intent_not_overridden(self, intent, phrase):
        result = _coerce(phrase, state=ConversationState.IDLE, intent=intent)
        assert result.coercion_reason is None
        assert result.intent_result.intent == intent


# ---------------------------------------------------------------------------
# WAITING_FOR_SIDE — "done" must NOT become CHECKOUT (handler manages it)
# ---------------------------------------------------------------------------

class TestWaitingForSideAntiRegression:
    def test_done_in_waiting_for_side_not_checkout(self):
        result = _coerce("done", state=ConversationState.WAITING_FOR_SIDE)
        assert result.intent_result.intent == Intent.UNKNOWN

    def test_nothing_else_in_waiting_for_side_not_checkout(self):
        result = _coerce("nothing else", state=ConversationState.WAITING_FOR_SIDE)
        assert result.intent_result.intent == Intent.UNKNOWN

    def test_finished_in_waiting_for_modifier_not_checkout(self):
        result = _coerce("finished", state=ConversationState.WAITING_FOR_MODIFIER)
        assert result.intent_result.intent == Intent.UNKNOWN


# ---------------------------------------------------------------------------
# Empty cart in IDLE — Rule 2 prevention
# ---------------------------------------------------------------------------

class TestEmptyCartInIdle:
    @pytest.mark.parametrize("phrase", _CHECKOUT_PHRASES)
    def test_empty_cart_never_coerced(self, phrase):
        result = coerce_idle_to_checkout(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.UNKNOWN, raw_text=phrase),
            nlu=_nlu(phrase),
            cart=_cart(empty=True),
        )
        assert result.coercion_reason is None
        assert result.intent_result.intent == Intent.UNKNOWN
