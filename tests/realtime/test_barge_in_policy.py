# tests/realtime/test_barge_in_policy.py
"""Tests for barge-in policy helpers.

Covers:
- is_actionable_barge_in (state-aware semantic gate)
- is_within_barge_in_guard_window
- is_filler_only
- evaluate_barge_in_candidate (full acceptance pipeline)
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.realtime.barge_in_policy import (
    BargeInDecision,
    evaluate_barge_in_candidate,
    is_actionable_barge_in,
    is_filler_only,
    is_within_barge_in_guard_window,
)
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_session(state: ConversationState) -> Session:
    session = Session(session_id="barge-in-test", restaurant_id="demo")
    session.conversation_state = state
    return session


def _make_config(
    *,
    barge_in_enabled: bool = True,
    post_playback_guard_ms: int = 250,
    min_barge_in_audio_ms: int = 700,
    min_barge_in_words: int = 2,
    min_barge_in_confidence: float = 0.60,
):
    cfg = MagicMock()
    cfg.barge_in_enabled = barge_in_enabled
    cfg.post_playback_guard_ms = post_playback_guard_ms
    cfg.min_barge_in_audio_ms = min_barge_in_audio_ms
    cfg.min_barge_in_words = min_barge_in_words
    cfg.min_barge_in_confidence = min_barge_in_confidence
    return cfg


# ---------------------------------------------------------------------------
# is_within_barge_in_guard_window
# ---------------------------------------------------------------------------

def test_guard_returns_false_when_playback_not_started() -> None:
    assert is_within_barge_in_guard_window(None, 0.25, now_monotonic=10.0) is False


def test_guard_returns_false_when_guard_disabled() -> None:
    assert is_within_barge_in_guard_window(10.0, 0.0, now_monotonic=10.05) is False
    assert is_within_barge_in_guard_window(10.0, -0.1, now_monotonic=10.05) is False


def test_guard_returns_true_when_within_window() -> None:
    # Playback started 100 ms ago, guard window is 250 ms → still guarded.
    assert is_within_barge_in_guard_window(10.0, 0.25, now_monotonic=10.1) is True


def test_guard_returns_true_at_zero_age() -> None:
    assert is_within_barge_in_guard_window(10.0, 0.25, now_monotonic=10.0) is True


def test_guard_returns_false_at_boundary_and_beyond() -> None:
    assert is_within_barge_in_guard_window(10.0, 0.25, now_monotonic=10.25) is False
    assert is_within_barge_in_guard_window(10.0, 0.25, now_monotonic=10.5) is False


def test_guard_uses_real_clock_when_now_omitted(monkeypatch) -> None:
    fake_now = [42.0]
    monkeypatch.setattr("app.realtime.barge_in_policy.time.monotonic", lambda: fake_now[0])

    assert is_within_barge_in_guard_window(41.9, 0.25) is True

    fake_now[0] = 42.3
    assert is_within_barge_in_guard_window(41.9, 0.25) is False


# ---------------------------------------------------------------------------
# is_filler_only
# ---------------------------------------------------------------------------

def test_empty_string_is_filler() -> None:
    assert is_filler_only("") is True


def test_whitespace_only_is_filler() -> None:
    assert is_filler_only("   ") is True


def test_single_filler_word_is_filler() -> None:
    for word in ("uh", "um", "hmm", "uhh", "mm", "mhm", "huh", "ah", "oh"):
        assert is_filler_only(word) is True, f"Expected {word!r} to be filler"


def test_multiple_filler_words_are_filler() -> None:
    assert is_filler_only("um hmm") is True
    assert is_filler_only("uh uh") is True


def test_filler_with_punctuation_is_still_filler() -> None:
    assert is_filler_only("uh,") is True
    assert is_filler_only("hmm.") is True


def test_real_command_not_filler() -> None:
    assert is_filler_only("no") is False
    assert is_filler_only("stop") is False
    assert is_filler_only("cancel") is False
    assert is_filler_only("change that") is False
    assert is_filler_only("hold on") is False
    assert is_filler_only("agent") is False


def test_real_content_not_filler() -> None:
    assert is_filler_only("coke") is False
    assert is_filler_only("chicken burger") is False
    assert is_filler_only("I want a burger") is False


def test_filler_mixed_with_command_not_filler() -> None:
    # "uh cancel" — the word "cancel" overrides filler classification
    assert is_filler_only("uh cancel") is False


# ---------------------------------------------------------------------------
# is_actionable_barge_in (state-aware gate)
# ---------------------------------------------------------------------------

def test_modifier_done_like_phrase_is_actionable() -> None:
    session = _build_session(ConversationState.WAITING_FOR_MODIFIER)
    session.conversation_context.available_choices_values = ("Cheese", "Bacon")
    assert is_actionable_barge_in(session, "yeah thats good thanks")


def test_size_choice_is_actionable_during_playback() -> None:
    session = _build_session(ConversationState.WAITING_FOR_SIZE)
    session.conversation_context.available_choices_values = ("Small", "Medium", "Large")
    assert is_actionable_barge_in(session, "make it large")


def test_preorder_ordering_request_is_actionable() -> None:
    session = _build_session(ConversationState.WAITING_FOR_ORDER_TYPE)
    assert is_actionable_barge_in(session, "add a coke")


def test_non_contextual_filler_is_not_actionable() -> None:
    session = _build_session(ConversationState.WAITING_FOR_MODIFIER)
    session.conversation_context.available_choices_values = ("Cheese", "Bacon")
    assert not is_actionable_barge_in(session, "uh huh")


def test_delivery_zip_phrase_is_actionable_during_playback() -> None:
    session = _build_session(ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY)
    session.conversation_context.current_prompt_field = "delivery_postal_code"
    assert is_actionable_barge_in(session, "my zip code is 30000.")


def test_spoken_delivery_zip_phrase_is_actionable_during_playback() -> None:
    session = _build_session(ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY)
    session.conversation_context.current_prompt_field = "delivery_postal_code"
    assert is_actionable_barge_in(session, "it's twenty one thousand")


# ---------------------------------------------------------------------------
# evaluate_barge_in_candidate — full pipeline
# ---------------------------------------------------------------------------

def test_evaluate_rejects_when_barge_in_disabled() -> None:
    cfg = _make_config(barge_in_enabled=False)
    d = evaluate_barge_in_candidate(
        session=None, text="pickup", audio_duration_ms=800,
        confidence=0.9, playback_started_at=None, config=cfg,
    )
    assert not d.accepted
    assert d.reason == "barge_in_disabled"


def test_evaluate_rejects_inside_guard_window() -> None:
    cfg = _make_config(post_playback_guard_ms=500)
    playback_started_at = 100.0
    # 100 ms after start → inside 500 ms guard
    d = evaluate_barge_in_candidate(
        session=None, text="pickup", audio_duration_ms=800,
        confidence=0.9, playback_started_at=playback_started_at, config=cfg,
    )
    # Simulate now = 100.1 via monkeypatching would be needed for full test,
    # but guard check passes None for playback when we set it to 0 for test:
    # Instead test the explicit guard rejection path via a fresh call:
    cfg2 = _make_config(post_playback_guard_ms=5000)  # 5-second guard
    import time as _time
    recent_started = _time.monotonic() - 0.1  # 100 ms ago
    d2 = evaluate_barge_in_candidate(
        session=None, text="pickup", audio_duration_ms=800,
        confidence=0.9, playback_started_at=recent_started, config=cfg2,
    )
    assert not d2.accepted
    assert "guard_window" in d2.reason


def test_evaluate_rejects_filler_only() -> None:
    cfg = _make_config()
    d = evaluate_barge_in_candidate(
        session=None, text="uh", audio_duration_ms=800,
        confidence=0.9, playback_started_at=None, config=cfg,
    )
    assert not d.accepted
    assert d.reason == "filler_only"


def test_evaluate_rejects_short_audio() -> None:
    cfg = _make_config(min_barge_in_audio_ms=700)
    d = evaluate_barge_in_candidate(
        session=None, text="I want a coke", audio_duration_ms=300,
        confidence=0.9, playback_started_at=None, config=cfg,
    )
    assert not d.accepted
    assert "audio_too_short" in d.reason


def test_evaluate_rejects_low_confidence() -> None:
    cfg = _make_config(min_barge_in_confidence=0.70)
    session = _build_session(ConversationState.WAITING_FOR_ORDER_TYPE)
    d = evaluate_barge_in_candidate(
        session=session, text="pickup please", audio_duration_ms=900,
        confidence=0.40, playback_started_at=None, config=cfg,
    )
    assert not d.accepted
    assert "confidence_too_low" in d.reason


def test_evaluate_rejects_too_few_words() -> None:
    # Use text that is NOT actionable for this state so the word count gate fires.
    # "hi" is not pickup/delivery so it is not actionable in WAITING_FOR_ORDER_TYPE,
    # and 1 word < min_barge_in_words=3.
    cfg = _make_config(min_barge_in_words=3)
    session = _build_session(ConversationState.WAITING_FOR_ORDER_TYPE)
    d = evaluate_barge_in_candidate(
        session=session, text="hi there",  # 2 words, not actionable
        audio_duration_ms=900, confidence=0.9,
        playback_started_at=None, config=cfg,
    )
    assert not d.accepted
    assert "too_few_words" in d.reason


def test_evaluate_accepts_known_command_below_word_threshold() -> None:
    """'stop' is a one-word command that bypasses the word-count gate."""
    cfg = _make_config(min_barge_in_words=3)
    session = _build_session(ConversationState.CONFIRMING_ORDER)
    d = evaluate_barge_in_candidate(
        session=session, text="no", audio_duration_ms=900,
        confidence=0.9, playback_started_at=None, config=cfg,
    )
    assert d.accepted


def test_evaluate_rejects_non_actionable_text() -> None:
    """Grammatically correct sentence that is not relevant to current state."""
    cfg = _make_config()
    session = _build_session(ConversationState.WAITING_FOR_QUANTITY)
    # Random sentence — normalize_quantity won't match, so not actionable
    d = evaluate_barge_in_candidate(
        session=session, text="the weather is nice today", audio_duration_ms=900,
        confidence=0.9, playback_started_at=None, config=cfg,
    )
    assert not d.accepted
    assert d.reason == "not_actionable"


def test_evaluate_accepts_valid_barge_in() -> None:
    cfg = _make_config()
    session = _build_session(ConversationState.WAITING_FOR_ORDER_TYPE)
    d = evaluate_barge_in_candidate(
        session=session, text="I want pickup please", audio_duration_ms=900,
        confidence=0.9, playback_started_at=None, config=cfg,
    )
    assert d.accepted
    assert d.reason == "accepted"
