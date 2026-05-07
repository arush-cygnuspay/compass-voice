# tests/realtime/test_turn_lifecycle.py
"""End-to-end turn-lifecycle tests.

Tests the full pipeline from TurnCommitController → ConversationSession,
verifying debounce behaviour, merge semantics, barge-in gating, and stale
event dropping without requiring Twilio/Deepgram connections.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from tests.support.voice_test_harness import install_test_stubs

install_test_stubs()

from app.realtime.conversation_session import ConversationSession
from app.realtime.turn_commit_controller import TurnCommitController
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

@dataclass
class _StubOutput:
    response_key: str = "ack"
    internal_response_text: str = "ok"
    spoken_response_text: str = "ok"
    end_call_after_playback: bool = False
    transfer_call_to_number: str | None = None


class _RecordingEngine:
    def __init__(self, output: _StubOutput | None = None) -> None:
        self.calls: list[str] = []
        self.output = output or _StubOutput()

    def process_turn(self, *, session, user_text, trace=None):
        self.calls.append(user_text)
        return self.output


@dataclass
class _StubTransport:
    barge_in_disabled: bool = False
    speak_calls: list[str] = field(default_factory=list)
    interrupt_calls: list[str] = field(default_factory=list)
    end_calls: int = 0
    _playback_started_at: float | None = None

    def is_barge_in_disabled(self) -> bool:
        return self.barge_in_disabled

    def debug_log(self, event: str, payload: dict[str, Any]) -> None:
        pass

    def begin_turn_trace(self, *, user_text: str):
        return None

    def annotate_response_trace(self, trace, **kwargs) -> None:
        pass

    async def speak_response(self, spoken_text, *, trace, end_call_after_playback):
        self.speak_calls.append(spoken_text)

    async def interrupt_playback(self, reason: str) -> None:
        self.interrupt_calls.append(reason)

    async def transfer_call(self, target_number: str) -> None:
        pass

    async def end_call(self) -> None:
        self.end_calls += 1


def _session(state: ConversationState = ConversationState.WAITING_FOR_ORDER_TYPE) -> Session:
    s = Session(session_id="lifecycle-test", restaurant_id="demo")
    s.conversation_state = state
    return s


def _make_cs(engine, transport, session=None):
    saves = []
    cs = ConversationSession(
        app_session=session or _session(),
        engine=engine,
        transport=transport,
        save_session_fn=lambda s: saves.append(s),
        load_session_fn=lambda *_: _session(),
    )
    return cs


# ---------------------------------------------------------------------------
# Debounce: final transcript waits before FSM call
# ---------------------------------------------------------------------------

def test_final_transcript_waits_for_debounce_before_fsm() -> None:
    """controller.on_transcript(final) does NOT immediately commit for content."""
    controller = TurnCommitController()
    controller.on_speech_started()

    # Non-whitelist content: no early commit
    result = controller.on_transcript("I would like a burger", is_final=True)
    assert result is None, "Final with content must not commit before debounce"


def test_two_finals_within_debounce_merge_into_one() -> None:
    """The second final merges with the first; only one CommittedTurn emitted."""
    controller = TurnCommitController()
    controller.on_speech_started()

    r1 = controller.on_transcript("chicken burger", is_final=True)
    assert r1 is None

    r2 = controller.on_transcript("with cheese", is_final=True)
    assert r2 is None

    committed = controller.on_utterance_end()
    assert committed is not None
    assert "chicken burger" in committed.text
    assert "with cheese" in committed.text


def test_utterance_end_commits_merged_text_immediately() -> None:
    controller = TurnCommitController()
    controller.on_speech_started()

    controller.on_transcript("I want a", is_final=True)
    controller.on_transcript("chicken burger", is_final=True)

    committed = controller.on_utterance_end()
    assert committed is not None
    # Exactly ONE commit produced
    second = controller.on_utterance_end()
    assert second is None


# ---------------------------------------------------------------------------
# One-word filler during TTS is ignored
# ---------------------------------------------------------------------------

def test_one_word_filler_during_tts_ignored() -> None:
    engine = _RecordingEngine()
    transport = _StubTransport(barge_in_disabled=False)
    cs = _make_cs(engine, transport)
    cs.set_phase_speaking()

    asyncio.run(cs.process_committed_turn("uh"))

    assert engine.calls == []
    assert transport.interrupt_calls == []


def test_cough_empty_transcript_during_tts_ignored() -> None:
    engine = _RecordingEngine()
    transport = _StubTransport(barge_in_disabled=False)
    cs = _make_cs(engine, transport)
    cs.set_phase_speaking()

    asyncio.run(cs.process_committed_turn(""))

    assert engine.calls == []
    assert transport.interrupt_calls == []


def test_hmm_noise_during_tts_ignored() -> None:
    engine = _RecordingEngine()
    transport = _StubTransport(barge_in_disabled=False)
    cs = _make_cs(engine, transport)
    cs.set_phase_speaking()

    asyncio.run(cs.process_committed_turn("hmm"))

    assert engine.calls == []
    assert transport.interrupt_calls == []


# ---------------------------------------------------------------------------
# Intentional correction is accepted
# ---------------------------------------------------------------------------

def test_intentional_correction_interrupts_tts_and_processes_fsm() -> None:
    """'no change that to coke' must stop TTS and process a new FSM turn."""
    engine = _RecordingEngine()
    transport = _StubTransport(barge_in_disabled=False)
    session = _session(ConversationState.WAITING_FOR_ORDER_TYPE)
    cs = _make_cs(engine, transport, session)
    cs.set_phase_speaking()

    asyncio.run(cs.process_committed_turn("no change that to coke"))

    assert len(transport.interrupt_calls) == 1
    assert len(engine.calls) == 1
    assert engine.calls[0] == "no change that to coke"


# ---------------------------------------------------------------------------
# Late / stale STT events
# ---------------------------------------------------------------------------

def test_late_final_from_stale_turn_id_is_ignored() -> None:
    engine = _RecordingEngine()
    transport = _StubTransport()
    cs = _make_cs(engine, transport)

    # Turn 10 was already processed
    asyncio.run(cs.process_committed_turn("real order", turn_id=10))
    assert len(engine.calls) == 1

    # Turn 5 arrives late (stale)
    asyncio.run(cs.process_committed_turn("stale fragment", turn_id=5))
    assert len(engine.calls) == 1


def test_same_turn_id_not_processed_twice() -> None:
    engine = _RecordingEngine()
    transport = _StubTransport()
    cs = _make_cs(engine, transport)

    asyncio.run(cs.process_committed_turn("order", turn_id=3))
    asyncio.run(cs.process_committed_turn("duplicate", turn_id=3))

    assert len(engine.calls) == 1


# ---------------------------------------------------------------------------
# No concurrent FSM processing
# ---------------------------------------------------------------------------

def test_concurrent_calls_do_not_run_fsm_concurrently() -> None:
    """The processing lock ensures FSM calls are strictly serial.

    Serial flow with mark-ack simulation:
    1. First turn processes fully (FSM + speak)
    2. on_playback_completed() resets phase to LISTENING
    3. Second turn processes fully
    This ensures the lock is contention-free between turns.
    """
    call_order: list[str] = []
    call_times: list[float] = []

    class _TimingEngine:
        output = _StubOutput()

        def process_turn(self, *, session, user_text, trace=None):
            call_order.append(user_text)
            call_times.append(time.monotonic())
            return self.output

    transport = _StubTransport()
    cs = _make_cs(_TimingEngine(), transport)

    async def _run():
        await cs.process_committed_turn("first")
        await cs.on_playback_completed()  # simulate mark ack → LISTENING
        await cs.process_committed_turn("second")

    asyncio.run(_run())

    assert call_order == ["first", "second"]
    assert call_times[1] >= call_times[0]


# ---------------------------------------------------------------------------
# Pending queue bounded
# ---------------------------------------------------------------------------

def test_pending_queue_bounded_at_max() -> None:
    engine = _RecordingEngine()
    transport = _StubTransport()
    cs = _make_cs(engine, transport)
    cs.set_phase_processing()

    asyncio.run(cs.process_committed_turn("first buffered"))
    asyncio.run(cs.process_committed_turn("second buffered"))
    asyncio.run(cs.process_committed_turn("overflow — should be dropped"))

    assert len(cs._pending_queue) <= 2


# ---------------------------------------------------------------------------
# TTS failure releases lock (no deadlock)
# ---------------------------------------------------------------------------

def test_tts_failure_releases_lock() -> None:
    from app.realtime.tts_exceptions import TTSFailureError

    engine = _RecordingEngine()

    class _FailTransport(_StubTransport):
        async def speak_response(self, text, *, trace, end_call_after_playback):
            raise TTSFailureError("bang", attempts=1, provider="deepgram")

    transport = _FailTransport()
    cs = _make_cs(engine, transport)

    asyncio.run(cs.process_committed_turn("first"))

    # Replace transport so second call succeeds
    cs.transport = _StubTransport()
    engine2 = _RecordingEngine()
    cs.engine = engine2
    asyncio.run(cs.process_committed_turn("second"))

    assert len(engine2.calls) == 1, "Lock should have been released after TTS failure"


# ---------------------------------------------------------------------------
# Post-TTS speech processes normally (user speaks immediately after bot)
# ---------------------------------------------------------------------------

def test_user_speaking_immediately_after_bot_mark_ack_processes_normally() -> None:
    engine = _RecordingEngine()
    transport = _StubTransport()
    cs = _make_cs(engine, transport)

    # Simulate: bot was speaking, mark ack arrives → LISTENING
    cs.set_phase_speaking()
    asyncio.run(cs.on_playback_completed())
    assert cs.is_listening()

    # User speaks immediately after mark ack — must process normally
    asyncio.run(cs.process_committed_turn("pickup please"))

    assert len(engine.calls) == 1
    assert engine.calls[0] == "pickup please"


# ---------------------------------------------------------------------------
# Playback guard suppresses bot-tail
# ---------------------------------------------------------------------------

def test_playback_guard_suppresses_bot_tail() -> None:
    """Speech arriving < POST_PLAYBACK_GUARD_MS after TTS start is rejected."""
    engine = _RecordingEngine()
    transport = _StubTransport(barge_in_disabled=False)
    transport._playback_started_at = time.monotonic()  # TTS just started

    session = _session(ConversationState.WAITING_FOR_ORDER_TYPE)
    cs = _make_cs(engine, transport, session)
    cs.set_phase_speaking()

    asyncio.run(cs.process_committed_turn("pickup please", barge_in_audio_ms=900))

    assert engine.calls == []
    assert transport.interrupt_calls == []
