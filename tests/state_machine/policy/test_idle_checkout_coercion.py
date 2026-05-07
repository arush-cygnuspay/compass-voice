# tests/state_machine/policy/test_idle_checkout_coercion.py
"""Unit tests for IdleCheckoutCoercionPolicy (coerce_idle_to_checkout).

Validates all 6 coercion rules:
  Rule 1 — only fires in IDLE state
  Rule 2 — skips empty cart
  Rule 3 — skips already-handled checkout-family intents
  Rule 4 — only coerces UNKNOWN intent
  Rule 5 — classifier-based coercion (exact/payment phrases)
  Rule 6 — semantic done-like fallback

Also validates:
  - coercion_reason is populated correctly
  - no coercion → original IntentResult returned unchanged
  - non-IDLE states (WAITING_FOR_SIDE etc.) are never touched
  - bare yes/no are not coerced
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

def _nlu(text: str, confidence: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(normalized_text=text, intent_confidence=confidence)


def _intent(intent: Intent = Intent.UNKNOWN, text: str = "") -> IntentResult:
    return IntentResult(intent=intent, raw_text=text)


class _Cart:
    def __init__(self, empty: bool = False):
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


def _coerce(
    text: str,
    *,
    state: ConversationState = ConversationState.IDLE,
    intent: Intent = Intent.UNKNOWN,
    cart_empty: bool = False,
    confidence: float = 0.0,
):
    return coerce_idle_to_checkout(
        state=state,
        intent_result=_intent(intent, text),
        nlu=_nlu(text, confidence),
        cart=_Cart(empty=cart_empty),
    )


# ---------------------------------------------------------------------------
# Rule 1 — Only IDLE state
# ---------------------------------------------------------------------------

class TestRule1OnlyIdleState:
    @pytest.mark.parametrize("state", [
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_SIZE,
        ConversationState.WAITING_FOR_SIDE_SIZE,
        ConversationState.WAITING_FOR_QUANTITY,
        ConversationState.CONFIRMING_ORDER,
        ConversationState.WAITING_FOR_PAYMENT,
        ConversationState.CONFIRMING_ITEM,
        ConversationState.GREETING,
    ])
    def test_non_idle_not_coerced(self, state):
        result = _coerce("done", state=state)
        assert result.coercion_reason is None
        assert result.intent_result.intent == Intent.UNKNOWN

    def test_idle_state_can_coerce(self):
        result = _coerce("done", state=ConversationState.IDLE)
        assert result.intent_result.intent == Intent.CHECKOUT


# ---------------------------------------------------------------------------
# Rule 2 — Empty cart → no coercion
# ---------------------------------------------------------------------------

class TestRule2EmptyCart:
    def test_empty_cart_checkout_phrase_not_coerced(self):
        result = _coerce("done", cart_empty=True)
        assert result.coercion_reason is None
        assert result.intent_result.intent == Intent.UNKNOWN

    def test_empty_cart_payment_phrase_not_coerced(self):
        result = _coerce("payment", cart_empty=True)
        assert result.coercion_reason is None

    def test_non_empty_cart_checkout_phrase_coerced(self):
        result = _coerce("done", cart_empty=False)
        assert result.intent_result.intent == Intent.CHECKOUT


# ---------------------------------------------------------------------------
# Rule 3 — Already-handled checkout-family intents are skipped
# ---------------------------------------------------------------------------

class TestRule3AlreadyHandledIntents:
    @pytest.mark.parametrize("intent", [
        Intent.CHECKOUT,
        Intent.CONFIRM_ORDER,
        Intent.FINISH_ORDER,
        Intent.END_ADDING,
        Intent.PAYMENT_REQUEST,
        Intent.REVIEW_ORDER,
        Intent.START_ORDER,
        Intent.DENY,
    ])
    def test_checkout_family_not_double_coerced(self, intent):
        result = _coerce("checkout", intent=intent)
        assert result.coercion_reason is None
        assert result.intent_result.intent == intent


# ---------------------------------------------------------------------------
# Rule 4 — Only coerce UNKNOWN
# ---------------------------------------------------------------------------

class TestRule4OnlyUnknownIntent:
    @pytest.mark.parametrize("intent", [
        Intent.ADD_ITEM,
        Intent.REMOVE_ITEM,
        Intent.MODIFY_ITEM,
        Intent.ASK_ITEM_INFO,
        Intent.AFFIRM,
        Intent.CANCEL,
        Intent.GREETING,
    ])
    def test_non_unknown_intent_not_coerced(self, intent):
        result = _coerce("done", intent=intent)
        assert result.coercion_reason is None
        assert result.intent_result.intent == intent


# ---------------------------------------------------------------------------
# Rule 5 — Classifier-based coercion (checkout / finalization / payment)
# ---------------------------------------------------------------------------

class TestRule5ClassifierPhrases:
    @pytest.mark.parametrize("phrase", [
        "checkout",
        "check out",
        "done",
        "finished",
        "thats it",
        "that is it",
        "thats all",
        "that is all",
        "nothing else",
        "no more",
        "im done",
        "i am done",
        "complete my order",
        "finalize",
        "finalize my order",
        "place the order",
        "place my order",
    ])
    def test_finalization_phrase_coerced(self, phrase):
        result = _coerce(phrase)
        assert result.intent_result.intent == Intent.CHECKOUT
        assert result.coercion_reason is not None
        assert "idle_checkout_classifier" in result.coercion_reason

    @pytest.mark.parametrize("phrase", [
        "payment",
        "pay",
        "pay now",
        "continue to payment",
        "proceed to payment",
        "send payment link",
        "send the payment link",
        "text me payment link",
        "text me the payment link",
        "text me the link",
    ])
    def test_payment_phrase_coerced(self, phrase):
        result = _coerce(phrase)
        assert result.intent_result.intent == Intent.CHECKOUT
        assert result.coercion_reason is not None
        assert "idle_checkout_classifier" in result.coercion_reason

    def test_coercion_reason_contains_classifier_reason(self):
        result = _coerce("checkout")
        assert result.coercion_reason is not None
        assert "exact_idle_checkout_phrase" in result.coercion_reason


# ---------------------------------------------------------------------------
# Rule 6 — Semantic done-like fallback
# ---------------------------------------------------------------------------

class TestRule6SemanticFallback:
    def test_lets_checkout_coerced_via_semantic_or_classifier(self):
        # "lets checkout" is in semantic DONE_WORDS and/or classifier
        result = _coerce("lets checkout")
        assert result.intent_result.intent == Intent.CHECKOUT

    def test_go_ahead_and_checkout_coerced(self):
        result = _coerce("go ahead and checkout")
        assert result.intent_result.intent == Intent.CHECKOUT


# ---------------------------------------------------------------------------
# Bare yes/no — must NOT be coerced
# ---------------------------------------------------------------------------

class TestBareYesNoNotCoerced:
    @pytest.mark.parametrize("phrase", [
        "yes",
        "yeah",
        "yep",
        "no",
        "nope",
        "nah",
    ])
    def test_bare_yes_no_not_coerced(self, phrase):
        # These have their own intents (AFFIRM/DENY); even if they slip through
        # as UNKNOWN, they must not trigger checkout.
        result = _coerce(phrase)
        assert result.intent_result.intent == Intent.UNKNOWN, (
            f"'{phrase}' must not be coerced to CHECKOUT"
        )


# ---------------------------------------------------------------------------
# Non-checkout unknown text — no coercion
# ---------------------------------------------------------------------------

class TestNonCheckoutTextNotCoerced:
    @pytest.mark.parametrize("phrase", [
        "burger",
        "add a coke",
        "what do you have",
        "how much is a burger",
        "i want fries",
    ])
    def test_ordering_text_not_coerced(self, phrase):
        result = _coerce(phrase)
        assert result.coercion_reason is None
        assert result.intent_result.intent == Intent.UNKNOWN

    def test_empty_text_not_coerced(self):
        result = _coerce("")
        assert result.coercion_reason is None

    def test_whitespace_only_not_coerced(self):
        result = _coerce("   ")
        assert result.coercion_reason is None


# ---------------------------------------------------------------------------
# raw_text is preserved in coerced IntentResult
# ---------------------------------------------------------------------------

class TestRawTextPreserved:
    def test_raw_text_carried_through(self):
        ir = IntentResult(intent=Intent.UNKNOWN, raw_text="that is all")
        result = coerce_idle_to_checkout(
            state=ConversationState.IDLE,
            intent_result=ir,
            nlu=_nlu("thats all"),
            cart=_Cart(empty=False),
        )
        assert result.intent_result.raw_text == "that is all"
        assert result.intent_result.intent == Intent.CHECKOUT
