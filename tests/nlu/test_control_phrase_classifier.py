# tests/nlu/test_control_phrase_classifier.py
"""Unit tests for ControlPhraseClassifier.

All tests are deterministic and require no fixtures, DB, or NLU model.
They cover every case listed in the spec plus selected edge cases.
"""
from __future__ import annotations

import pytest

from app.nlu.control_phrase_classifier import (
    ControlPhraseClassifier,
    ControlPhraseResult,
    DEFAULT_CLASSIFIER,
)

# Convenience aliases
_SIDE = "waiting_for_side"
_MOD = "waiting_for_modifier"
_CONFIRM = "confirming_order"
_IDLE = "idle"


# ===========================================================================
# Helpers
# ===========================================================================

def classify(text: str, state: str) -> ControlPhraseResult:
    return DEFAULT_CLASSIFIER.classify(text, state)


# ===========================================================================
# A. Side / modifier waiting-state cases
# ===========================================================================


class TestSkipPhrases:
    """'skip', 'skip that', 'no skip', 'no skip that' → skip in waiting states."""

    @pytest.mark.parametrize("phrase,state", [
        ("skip", _MOD),
        ("skip that", _MOD),
        ("skip it", _MOD),
        ("no skip", _MOD),
        ("no skip that", _MOD),
        ("no skip it", _MOD),
        ("skip", _SIDE),
        ("skip that", _SIDE),
        ("no skip that", _SIDE),
        ("leave it", _MOD),
        ("leave that", _MOD),
        ("leave it off", _SIDE),
        ("dont add that", _MOD),
    ])
    def test_skip_phrases(self, phrase, state):
        result = classify(phrase, state)
        assert result.action == "skip", f"{phrase!r} in {state} → expected skip, got {result.action!r}"
        assert result.confidence > 0

    def test_no_skip_that_is_not_negated_option(self):
        result = classify("no skip that", _MOD)
        assert result.action == "skip"
        assert result.normalized_target is None  # not a menu item

    def test_skip_not_triggered_in_confirm_state(self):
        result = classify("skip", _CONFIRM)
        # CONFIRM state only knows checkout/confirm/deny — skip phrases return none
        assert result.action == "none"


class TestDonePhrases:
    """'done', 'add done', 'no done', etc. → done in waiting states."""

    @pytest.mark.parametrize("phrase,state", [
        ("done", _MOD),
        ("add done", _MOD),
        ("no done", _MOD),
        ("thats all", _MOD),
        ("that is all", _MOD),
        ("all good", _SIDE),
        ("all good then", _MOD),
        ("nothing else", _MOD),
        ("no more", _SIDE),
        ("no nothing else", _MOD),
        ("im done", _MOD),
        ("i am done", _SIDE),
    ])
    def test_done_phrases(self, phrase, state):
        result = classify(phrase, state)
        assert result.action == "done", f"{phrase!r} in {state} → expected done, got {result.action!r}"

    def test_no_done_is_not_negated_option(self):
        result = classify("no done", _MOD)
        assert result.action == "done"
        assert result.normalized_target is None

    def test_no_more_is_done_not_negated_option(self):
        result = classify("no more", _MOD)
        assert result.action == "done"


class TestRepeatPhrases:
    """'repeat', 'can you repeat' → repeat (meta-clarify intercept).
    Note: 'what are the options' is intentionally NOT classified as repeat —
    it falls through to resolve_control_intent → OPTIONS_REQUEST.
    """

    @pytest.mark.parametrize("phrase,state", [
        ("repeat", _MOD),
        ("repeat that", _MOD),
        ("can you repeat", _MOD),
        ("can you repeat that", _MOD),
        ("say that again", _SIDE),
    ])
    def test_repeat_phrases(self, phrase, state):
        result = classify(phrase, state)
        assert result.action == "repeat", f"{phrase!r} in {state} → expected repeat, got {result.action!r}"

    def test_options_request_phrases_return_none(self):
        # These fall through to the existing OPTIONS_REQUEST control intent path.
        for phrase in ("what are the options", "list options", "what options do you have",
                       "what are my options"):
            result = classify(phrase, _MOD)
            assert result.action == "none", (
                f"{phrase!r} should return none (handled by OPTIONS_REQUEST resolver), got {result.action!r}"
            )


class TestNegatedOption:
    """'no bun', 'without X', etc. → negated_option with correct target."""

    def test_no_bun_in_side_state(self):
        result = classify("no bun", _SIDE)
        assert result.action == "negated_option"
        assert result.normalized_target == "bun"

    def test_no_onions_in_modifier_state(self):
        result = classify("no onions", _MOD)
        assert result.action == "negated_option"
        assert result.normalized_target == "onions"

    def test_without_mayo_is_negated_option(self):
        result = classify("without mayo", _MOD)
        assert result.action == "negated_option"
        assert result.normalized_target == "mayo"

    def test_remove_pickles_is_negated_option(self):
        result = classify("remove pickles", _MOD)
        assert result.action == "negated_option"
        assert result.normalized_target == "pickles"

    def test_hold_the_lettuce(self):
        result = classify("hold the lettuce", _MOD)
        # "the" stripped → target = "lettuce" but "hold" matches the prefix
        # The regex captures "the lettuce" as target; "the" is a stop word
        # but "lettuce" is a real noun → negated_option is correct
        assert result.action == "negated_option"

    def test_no_bun_in_confirm_state_is_not_negated_option(self):
        """'no bun' in CONFIRMING_ORDER must NOT become negated_option."""
        result = classify("no bun", _CONFIRM)
        # Confirm state only knows checkout/none — no negated_option
        assert result.action in {"none", "deny"}
        # Specifically it should be none because "no bun" is not in _DENY_EXACT
        assert result.action == "none"

    def test_no_skip_that_is_not_negated_option(self):
        """Precedence: "skip" in target → skip, not negated_option."""
        result = classify("no skip that", _MOD)
        assert result.action == "skip"
        assert result.normalized_target is None

    def test_no_done_is_not_negated_option(self):
        result = classify("no done", _MOD)
        assert result.action == "done"

    def test_no_nothing_else_is_done_not_negated_option(self):
        result = classify("no nothing else", _MOD)
        assert result.action == "done"

    def test_no_thanks_is_not_negated_option(self):
        # "thanks" is a control word → skip (polite skip)
        result = classify("no thanks", _MOD)
        assert result.action == "skip"
        assert result.normalized_target is None


# ===========================================================================
# B. Confirm-order state
# ===========================================================================


class TestCheckoutPhrases:
    """'checkout', 'i said checkout', 'oh yeah checkout', etc. → checkout."""

    @pytest.mark.parametrize("phrase", [
        "checkout",
        "check out",
        "place the order",
        "place my order",
        "confirm order",
    ])
    def test_direct_checkout(self, phrase):
        result = classify(phrase, _CONFIRM)
        assert result.action == "checkout", f"{phrase!r} → expected checkout"

    @pytest.mark.parametrize("phrase", [
        "i said checkout",
        "oh yeah checkout",
        "i said check out",
        "oh yeah check out",
        "actually checkout",
        "well actually checkout",
        "yeah checkout",
        "yes checkout",
        "ok checkout",
        "okay checkout",
        "please checkout",
    ])
    def test_prefixed_checkout(self, phrase):
        result = classify(phrase, _CONFIRM)
        assert result.action == "checkout", f"{phrase!r} → expected checkout, got {result.action!r}"
        assert result.confidence >= 0.90

    def test_just_a_cup_is_not_checkout(self):
        """'just a cup' in CONFIRMING_ORDER must NOT be treated as checkout."""
        result = classify("just a cup", _CONFIRM)
        assert result.action == "none"

    def test_checkout_not_triggered_in_side_state(self):
        result = classify("checkout", _SIDE)
        # Side state doesn't know about checkout — returns none
        assert result.action == "none"


class TestConfirmDenyInConfirmState:
    """confirm/deny in CONFIRMING_ORDER — classifier returns none (handled by
    existing control_intent_resolver).  We only add checkout detection."""

    def test_yes_returns_none_from_classifier(self):
        # "yes" as confirm is handled by existing AFFIRM phrase fallback,
        # not by ControlPhraseClassifier — classifier returns none.
        result = classify("yes", _CONFIRM)
        assert result.action == "none"

    def test_no_returns_none_from_classifier(self):
        result = classify("no", _CONFIRM)
        assert result.action == "none"


# ===========================================================================
# C. Other states → none (no-op)
# ===========================================================================


class TestNoneInOtherStates:
    @pytest.mark.parametrize("phrase,state", [
        # IDLE non-checkout phrases still return "none"
        ("skip", _IDLE),
        ("can you repeat", _IDLE),
        ("no bun", _IDLE),
        # Note: "done" and "checkout" in IDLE now return "checkout" (targeted state).
        # Those cases are covered in tests/nlu/test_idle_checkout_classifier.py.
        ("skip", "waiting_for_size"),
        ("skip", None),
    ])
    def test_no_op_in_non_targeted_states(self, phrase, state):
        result = classify(phrase, state)
        assert result.action == "none", f"{phrase!r} in {state!r} should be none"

    def test_done_in_idle_is_now_checkout(self):
        # IDLE is now a targeted state; "done" + non-empty cart → checkout coercion.
        result = classify("done", _IDLE)
        assert result.action == "checkout"

    def test_checkout_in_idle_is_checkout(self):
        result = classify("checkout", _IDLE)
        assert result.action == "checkout"


# ===========================================================================
# D. Precedence
# ===========================================================================


class TestPrecedence:
    def test_repeat_beats_skip(self):
        result = classify("repeat that", _MOD)
        assert result.action == "repeat"

    def test_skip_beats_negated_option_when_target_is_control_word(self):
        result = classify("no skip that", _MOD)
        assert result.action == "skip"

    def test_done_beats_negated_option_when_target_is_done_word(self):
        result = classify("no done", _MOD)
        assert result.action == "done"

    def test_real_item_negation_is_not_swallowed(self):
        result = classify("no extra cheese", _MOD)
        assert result.action == "negated_option"
        assert "cheese" in (result.normalized_target or "")


# ===========================================================================
# E. Singleton is reusable
# ===========================================================================


class TestSingleton:
    def test_default_classifier_is_same_class(self):
        assert isinstance(DEFAULT_CLASSIFIER, ControlPhraseClassifier)

    def test_stateless_repeated_calls(self):
        r1 = DEFAULT_CLASSIFIER.classify("skip", _MOD)
        r2 = DEFAULT_CLASSIFIER.classify("skip", _MOD)
        assert r1 == r2

    def test_none_input(self):
        result = DEFAULT_CLASSIFIER.classify("", _MOD)
        assert result.action == "none"

    def test_none_state(self):
        result = DEFAULT_CLASSIFIER.classify("skip", None)
        assert result.action == "none"
