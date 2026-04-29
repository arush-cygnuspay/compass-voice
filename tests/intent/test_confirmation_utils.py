from app.nlu.intent_resolution.confirmation_resolver import (
    is_affirmation,
    resolve_confirmation_decision,
)
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult


def _nlu(intent: Intent, confidence: float = 0.99) -> NLUResult:
    return NLUResult(
        effective_intent=intent,
        intent_confidence=confidence,
        raw_text="",
        normalized_text="",
    )


def test_canonical_affirm_maps_to_affirm() -> None:
    assert resolve_confirmation_decision(_nlu(Intent.AFFIRM), "anything") == "affirm"


def test_canonical_confirm_maps_to_affirm() -> None:
    assert resolve_confirmation_decision(_nlu(Intent.CONFIRM), "anything") == "affirm"


def test_canonical_deny_maps_to_deny() -> None:
    assert resolve_confirmation_decision(_nlu(Intent.DENY), "anything") == "deny"


def test_canonical_cancel_maps_to_cancel() -> None:
    assert resolve_confirmation_decision(_nlu(Intent.CANCEL), "anything") == "cancel"


def test_canonical_cancel_order_maps_to_cancel() -> None:
    assert resolve_confirmation_decision(_nlu(Intent.CANCEL_ORDER), "anything") == "cancel"


def test_phrase_yeah_go_ahead_maps_to_affirm_when_intent_unknown() -> None:
    assert (
        resolve_confirmation_decision(
            _nlu(Intent.UNKNOWN, confidence=0.2),
            "yeah go ahead",
            expect_confirmation=True,
        )
        == "affirm"
    )


def test_phrase_do_it_maps_to_affirm() -> None:
    assert resolve_confirmation_decision(None, "do it", expect_confirmation=True) == "affirm"


def test_phrase_proceed_maps_to_affirm() -> None:
    assert resolve_confirmation_decision(None, "proceed", expect_confirmation=True) == "affirm"


def test_phrase_no_not_that_maps_to_deny() -> None:
    assert resolve_confirmation_decision(None, "no not that", expect_confirmation=True) == "deny"


def test_phrase_wait_hold_on_maps_to_cancel() -> None:
    assert resolve_confirmation_decision(None, "wait hold on", expect_confirmation=True) == "cancel"


def test_unrelated_unknown_text_stays_unknown() -> None:
    assert resolve_confirmation_decision(None, "what time do you close", expect_confirmation=True) == "unknown"


def test_ok_only_counts_as_affirm_when_confirmation_expected() -> None:
    assert not is_affirmation(None, "ok", expect_confirmation=False)
    assert is_affirmation(None, "ok", expect_confirmation=True)
