# tests/nlu/test_idle_checkout_classifier.py
"""Unit tests for ControlPhraseClassifier IDLE-state checkout classification.

Validates that:
- All target checkout/done/payment phrases produce action="checkout".
- Bare yes/no do NOT produce action="checkout".
- Prefixed phrases ("please checkout", "okay payment") are caught.
- Non-checkout idle phrases return action="none".
- Non-IDLE states are unaffected by the new IDLE logic.
"""
import pytest

from app.nlu.control_phrase_classifier import DEFAULT_CLASSIFIER, ControlPhraseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(text: str, state: str = "idle") -> ControlPhraseResult:
    return DEFAULT_CLASSIFIER.classify(text, state=state)


def _is_checkout(text: str, state: str = "idle") -> bool:
    return _classify(text, state=state).action == "checkout"


# ---------------------------------------------------------------------------
# Core checkout / done phrases
# ---------------------------------------------------------------------------

class TestIdleCheckoutExactPhrases:
    @pytest.mark.parametrize("phrase", [
        "checkout",
        "check out",
        "done",
        "finished",
        "im done",
        "i am done",
        "im finished",
        "i am finished",
        "thats it",
        "that is it",
        "thats all",
        "that is all",
        "nothing else",
        "no more",
        "all good",
        "im good",
        "i am good",
        "were good",
        "we are good",
    ])
    def test_done_phrase_is_checkout(self, phrase):
        assert _is_checkout(phrase)

    @pytest.mark.parametrize("phrase", [
        "complete my order",
        "complete the order",
        "finish my order",
        "finish the order",
        "finalize",
        "finalize my order",
        "finalize the order",
        "place the order",
        "place my order",
        "lets checkout",
        "lets check out",
        "go ahead and checkout",
        "go ahead and check out",
        "ready to checkout",
        "ready to check out",
    ])
    def test_finalization_phrase_is_checkout(self, phrase):
        assert _is_checkout(phrase)


# ---------------------------------------------------------------------------
# Payment-specific phrases — must route to confirm_order_summary first
# ---------------------------------------------------------------------------

class TestIdlePaymentPhrases:
    @pytest.mark.parametrize("phrase", [
        "payment",
        "pay",
        "pay now",
        "continue to payment",
        "proceed to payment",
        "send payment link",
        "send the payment link",
        "send me the payment link",
        "text me payment link",
        "text me the payment link",
        "text me the link",
        "text me a payment link",
    ])
    def test_payment_phrase_is_checkout(self, phrase):
        assert _is_checkout(phrase)


# ---------------------------------------------------------------------------
# Prefixed phrases
# ---------------------------------------------------------------------------

class TestIdlePrefixedPhrases:
    @pytest.mark.parametrize("phrase", [
        "please checkout",
        "okay done",
        "yeah checkout",
        "yes checkout",
        "alright checkout",
        "ok payment",
        "okay payment",
        "please finalize",
        "please place the order",
    ])
    def test_prefixed_phrase_is_checkout(self, phrase):
        assert _is_checkout(phrase)


# ---------------------------------------------------------------------------
# Case / normalization — classifier lowercases and strips punctuation
# ---------------------------------------------------------------------------

class TestIdleNormalization:
    def test_apostrophe_stripped_thats_it(self):
        # "That's it." → normalize_text → "thats it"
        assert _is_checkout("That's it.")

    def test_apostrophe_stripped_im_done(self):
        assert _is_checkout("I'm done.")

    def test_uppercase_checkout(self):
        assert _is_checkout("CHECKOUT")

    def test_mixed_case_finalize(self):
        assert _is_checkout("Finalize My Order")


# ---------------------------------------------------------------------------
# Bare yes/no must NOT trigger checkout
# ---------------------------------------------------------------------------

class TestBareYesNoNotCheckout:
    @pytest.mark.parametrize("phrase", [
        "yes",
        "yeah",
        "yep",
        "yup",
        "no",
        "nope",
        "nah",
    ])
    def test_bare_yes_no_not_checkout(self, phrase):
        result = _classify(phrase, state="idle")
        assert result.action != "checkout", f"'{phrase}' must not produce checkout in IDLE"


# ---------------------------------------------------------------------------
# Ambiguous or non-checkout idle phrases return none
# ---------------------------------------------------------------------------

class TestIdleNonCheckoutPhrases:
    @pytest.mark.parametrize("phrase", [
        "burger",
        "add coke",
        "what do you have",
        "how much",
        "give me a burger",
        "repeat that",
        "",
        "   ",
    ])
    def test_non_checkout_phrase_returns_none(self, phrase):
        result = _classify(phrase.strip(), state="idle")
        assert result.action == "none", f"'{phrase}' must return none, got {result.action}"


# ---------------------------------------------------------------------------
# Non-IDLE states must not fire the IDLE checkout rule
# ---------------------------------------------------------------------------

class TestNonIdleStatesUnaffected:
    @pytest.mark.parametrize("state", [
        "waiting_for_side",
        "waiting_for_modifier",
        "waiting_for_size",
        "confirming_order",
        "waiting_for_payment",
        "idle_unknown_state",
    ])
    def test_done_in_non_idle_not_checkout(self, state):
        # "done" in WAITING_FOR_SIDE → action="done" (not "checkout")
        # "done" in other non-IDLE unknown states → action="none"
        result = _classify("done", state=state)
        assert result.action != "checkout", (
            f"'done' in state '{state}' must not produce checkout"
        )

    def test_waiting_for_side_done_is_done_action(self):
        result = _classify("done", state="waiting_for_side")
        assert result.action == "done"

    def test_confirming_order_checkout_is_checkout_action(self):
        # CONFIRMING_ORDER uses its own classifier path and also returns checkout
        result = _classify("checkout", state="confirming_order")
        assert result.action == "checkout"
