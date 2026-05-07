# tests/realtime/test_turn_commit_controller.py
"""Tests for TurnCommitController.

Covers:
- Debounce behavior (finals must NOT auto-commit via terminal punctuation)
- Multi-final merging
- UtteranceEnd and speech_final immediate commit
- turn_id monotonicity
- Duplicate suppression within same utterance
- Whitelist early-commit still works (yes/no/ok etc.)
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.realtime.turn_commit_controller import CommittedTurn, TurnCommitController


# ---------------------------------------------------------------------------
# Debounce: terminal punctuation must NOT commit early
# ---------------------------------------------------------------------------

def test_terminal_punctuation_does_not_commit_early() -> None:
    """'chicken burger.' must NOT produce an immediate commit."""
    controller = TurnCommitController()
    controller.on_speech_started()
    result = controller.on_transcript("chicken burger.", is_final=True)
    assert result is None, "Terminal punctuation must not trigger early commit"


def test_two_finals_merge_into_one_commit() -> None:
    """Two finals within a debounce window merge into a single committed turn."""
    controller = TurnCommitController()
    controller.on_speech_started()

    # First final — no early-commit for regular content
    r1 = controller.on_transcript("chicken burger", is_final=True)
    assert r1 is None

    # Second final — arrives before speech_final / utterance_end
    r2 = controller.on_transcript("with cheese", is_final=True)
    assert r2 is None

    # Utterance ends — both fragments should be merged
    committed = controller.on_utterance_end()
    assert committed is not None
    assert committed.text == "chicken burger with cheese"


def test_utterance_end_commits_merged_text_immediately() -> None:
    """UtteranceEnd triggers an immediate commit of all accumulated finals."""
    controller = TurnCommitController()
    controller.on_speech_started()

    controller.on_transcript("I want a", is_final=True)
    controller.on_transcript("chicken burger", is_final=True)

    committed = controller.on_utterance_end()
    assert committed is not None
    assert "chicken burger" in committed.text
    assert "I want a" in committed.text


def test_speech_final_commits_merged_text_immediately() -> None:
    """speech_final also triggers immediate commit."""
    controller = TurnCommitController()
    controller.on_speech_started()

    controller.on_transcript("two large cokes", is_final=True)
    committed = controller.on_speech_final()
    assert committed is not None
    assert committed.text == "two large cokes"


# ---------------------------------------------------------------------------
# Whitelist early-commit still works
# ---------------------------------------------------------------------------

def test_whitelist_reply_commits_early() -> None:
    controller = TurnCommitController()
    controller.on_speech_started()
    committed = controller.on_transcript("yes", is_final=True)
    assert committed is not None
    assert committed.text == "yes"


def test_no_commits_early() -> None:
    controller = TurnCommitController()
    controller.on_speech_started()
    committed = controller.on_transcript("no", is_final=True)
    assert committed is not None


def test_done_like_reply_commits_early() -> None:
    controller = TurnCommitController()
    controller.on_speech_started()
    committed = controller.on_transcript("no done", is_final=True)
    assert committed is not None
    assert committed.text == "no done"


# ---------------------------------------------------------------------------
# turn_id monotonicity
# ---------------------------------------------------------------------------

def test_turn_id_increments_per_commit() -> None:
    controller = TurnCommitController()

    controller.on_speech_started()
    t1 = controller.on_transcript("yes", is_final=True)
    assert t1 is not None

    controller.on_speech_started()
    t2 = controller.on_transcript("no", is_final=True)
    assert t2 is not None

    assert t2.turn_id > t1.turn_id


def test_turn_id_nonzero() -> None:
    controller = TurnCommitController()
    controller.on_speech_started()
    committed = controller.on_transcript("yes", is_final=True)
    assert committed is not None
    assert committed.turn_id >= 1


# ---------------------------------------------------------------------------
# Duplicate suppression
# ---------------------------------------------------------------------------

def test_suppresses_duplicate_commit_within_same_utterance() -> None:
    controller = TurnCommitController()

    controller.on_speech_started()
    committed = controller.on_transcript("yes", is_final=True)
    assert committed is not None

    assert controller.on_speech_final() is None
    assert controller.on_utterance_end() is None


def test_allows_same_text_in_new_utterance() -> None:
    controller = TurnCommitController()

    controller.on_speech_started()
    assert controller.on_transcript("one", is_final=True) is None
    first = controller.on_speech_final()
    assert first is not None
    assert first.text == "one"

    controller.on_speech_started()
    assert controller.on_transcript("one", is_final=True) is None
    second = controller.on_speech_final()
    assert second is not None
    assert second.text == "one"

    assert second.turn_id > first.turn_id
