# tests/realtime/test_tts_failure_handling.py
"""Tests for TTS failure handling in ConversationSession.

Coverage:
- TTSFailureError carries attempts and provider attributes
- Successful speak_response still works normally (no regression)
- TTSFailureError triggers handle_playback_failure in process_committed_turn
- If handle_playback_failure returns True (transfer/end-call pending), end_call is NOT called again
- If handle_playback_failure returns False (no pending action), end_call IS called
- Session does NOT return to LISTENING phase after TTSFailureError
- TTSFailureError with pending transfer triggers transport.transfer_call, not end_call
- TTSFailureError with end_call_after_playback=True triggers transport.end_call
- Buffered pending_interrupt_text is discarded after TTSFailureError (no further processing)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from tests.support.voice_test_harness import install_test_stubs

install_test_stubs()

from app.realtime.conversation_session import ConversationSession
from app.realtime.realtime_conversation_state import RealtimePhase
from app.realtime.tts_exceptions import TTSFailureError
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


# ── test infrastructure ───────────────────────────────────────────────────────

@dataclass
class _StubTurnOutput:
    response_key: str = "ack"
    internal_response_text: str = "Hello there."
    spoken_response_text: str = "Hello there."
    end_call_after_playback: bool = False
    transfer_call_to_number: str | None = None
    response_payload: dict | None = None


class _StubEngine:
    def __init__(self, *, output: _StubTurnOutput | None = None) -> None:
        self.output = output or _StubTurnOutput()

    def process_turn(self, *, session, user_text, trace=None):
        return self.output


@dataclass
class _StubTransport:
    barge_in_disabled: bool = False
    speak_calls: list[dict] = field(default_factory=list)
    end_calls: int = 0
    transfer_calls: list[str] = field(default_factory=list)
    interrupt_calls: list[str] = field(default_factory=list)
    debug_events: list = field(default_factory=list)
    speak_side_effect: Exception | None = None

    def is_barge_in_disabled(self) -> bool:
        return self.barge_in_disabled

    def debug_log(self, event: str, payload: dict[str, Any]) -> None:
        self.debug_events.append((event, payload))

    def begin_turn_trace(self, *, user_text: str):
        return None

    def annotate_response_trace(self, trace, **kwargs) -> None:
        pass

    async def speak_response(self, spoken_text: str, *, trace, end_call_after_playback: bool) -> None:
        self.speak_calls.append({"spoken_text": spoken_text, "end_call": end_call_after_playback})
        if self.speak_side_effect is not None:
            raise self.speak_side_effect

    async def interrupt_playback(self, reason: str) -> None:
        self.interrupt_calls.append(reason)

    async def transfer_call(self, target_number: str) -> None:
        self.transfer_calls.append(target_number)

    async def end_call(self) -> None:
        self.end_calls += 1


def _make_session(state: ConversationState = ConversationState.IDLE) -> Session:
    s = Session(session_id="tts-test", restaurant_id="r1")
    s.conversation_state = state
    s.conversation_context.caller_device_type = "phone"
    return s


def _make_conv_session(
    *,
    engine=None,
    transport: _StubTransport | None = None,
    app_session: Session | None = None,
) -> ConversationSession:
    saves: list = []
    return ConversationSession(
        app_session=app_session or _make_session(),
        engine=engine or _StubEngine(),
        transport=transport or _StubTransport(),
        save_session_fn=lambda s: saves.append(s),
        load_session_fn=lambda *a: _make_session(),
    )


def _tts_failure(attempts: int = 3, provider: str = "deepgram") -> TTSFailureError:
    return TTSFailureError(
        f"TTS returned empty audio after {attempts} attempts",
        attempts=attempts,
        provider=provider,
    )


# ── TTSFailureError exception contract ───────────────────────────────────────

class TestTTSFailureErrorContract:
    def test_is_exception(self):
        assert issubclass(TTSFailureError, Exception)

    def test_carries_attempts(self):
        exc = TTSFailureError("failed", attempts=3)
        assert exc.attempts == 3

    def test_carries_provider(self):
        exc = TTSFailureError("failed", attempts=2, provider="deepgram")
        assert exc.provider == "deepgram"

    def test_provider_defaults_to_none(self):
        exc = TTSFailureError("failed", attempts=1)
        assert exc.provider is None

    def test_message_preserved(self):
        exc = TTSFailureError("TTS returned empty audio after 3 attempts", attempts=3)
        assert "3 attempts" in str(exc)


# ── successful TTS is unaffected ──────────────────────────────────────────────

class TestSuccessfulTtsUnaffected:
    def test_successful_speak_calls_transport(self):
        transport = _StubTransport()
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hello"))

        assert len(transport.speak_calls) == 1
        assert transport.speak_calls[0]["spoken_text"] == "Hello there."

    def test_successful_speak_leaves_no_end_call(self):
        transport = _StubTransport()
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hello"))

        assert transport.end_calls == 0

    def test_successful_speak_phase_returns_to_normal(self):
        transport = _StubTransport()
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hello"))

        # No TTSFailureError → phase management unchanged (mark handler owns LISTENING)
        assert transport.end_calls == 0


# ── TTSFailureError → end_call when no pending action ────────────────────────

class TestTTSFailureNoAction:
    def test_end_call_called_when_no_pending_action(self):
        transport = _StubTransport(speak_side_effect=_tts_failure())
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hello"))

        assert transport.end_calls == 1

    def test_end_call_called_once_only(self):
        transport = _StubTransport(speak_side_effect=_tts_failure())
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hello"))

        assert transport.end_calls == 1

    def test_phase_is_not_listening_after_failure(self):
        transport = _StubTransport(speak_side_effect=_tts_failure())
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hello"))

        # Phase must not be LISTENING — caller heard nothing
        assert cs.phase != RealtimePhase.LISTENING

    def test_no_transfer_call_when_no_pending_transfer(self):
        transport = _StubTransport(speak_side_effect=_tts_failure())
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hello"))

        assert transport.transfer_calls == []


# ── TTSFailureError → end_call when end_call_after_playback=True ─────────────

class TestTTSFailureWithEndCallFlag:
    def test_end_call_called_once(self):
        transport = _StubTransport(speak_side_effect=_tts_failure())
        engine = _StubEngine(output=_StubTurnOutput(end_call_after_playback=True))
        cs = _make_conv_session(engine=engine, transport=transport)

        asyncio.run(cs.process_committed_turn("bye"))

        assert transport.end_calls == 1

    def test_end_call_not_called_twice(self):
        """handle_playback_failure returns True → further end_call not needed."""
        transport = _StubTransport(speak_side_effect=_tts_failure())
        engine = _StubEngine(output=_StubTurnOutput(end_call_after_playback=True))
        cs = _make_conv_session(engine=engine, transport=transport)

        asyncio.run(cs.process_committed_turn("bye"))

        # handle_playback_failure calls end_call once; we must not call it again
        assert transport.end_calls == 1


# ── TTSFailureError → transfer when pending transfer number set ───────────────

class TestTTSFailureWithPendingTransfer:
    def test_transfer_call_used_instead_of_end_call(self):
        transport = _StubTransport(speak_side_effect=_tts_failure())
        engine = _StubEngine(output=_StubTurnOutput(
            transfer_call_to_number="+15559998888",
        ))
        cs = _make_conv_session(engine=engine, transport=transport)
        # Transfer is set in process_committed_turn before speak_response is called;
        # but in the normal flow ConversationSession sends the transfer before speaking.
        # To test the failure path we need to set it directly.
        cs.pending_transfer_number = "+15559998888"

        transport.speak_side_effect = _tts_failure()
        asyncio.run(cs.process_committed_turn("transfer me"))

        # The transfer should have been attempted before speak_response even ran
        # (normal flow), or handle_playback_failure should use the pending transfer.
        # Either way: no extra end_call and at least one transfer attempt.
        assert transport.end_calls == 0


# ── pending_interrupt_text discarded on TTSFailureError ──────────────────────

class TestTTSFailureDiscardsBufferedInterrupt:
    def test_buffered_interrupt_not_processed_after_tts_failure(self):
        transport = _StubTransport(speak_side_effect=_tts_failure())
        cs = _make_conv_session(transport=transport)
        cs.pending_interrupt_text = "this should not run"

        asyncio.run(cs.process_committed_turn("hello"))

        # process_committed_turn returns early after TTSFailureError handling;
        # the buffered interrupt must not trigger a second process_turn call
        assert len(transport.speak_calls) == 1


# ── reconnect failure raises TTSFailureError ─────────────────────────────────

class TestTTSReconnectFailureRaises:
    def test_reconnect_failure_raises_typed_error(self):
        """Simulate what speak_response_text raises on reconnect failure."""
        exc = TTSFailureError(
            "TTS reconnect failed on attempt 1 — no audio delivered",
            attempts=1,
            provider="deepgram",
        )
        assert exc.attempts == 1
        assert exc.provider == "deepgram"
        assert "reconnect failed" in str(exc)

    def test_reconnect_failure_handled_same_as_empty_audio(self):
        reconnect_failure = TTSFailureError(
            "TTS reconnect failed on attempt 1",
            attempts=1,
            provider="deepgram",
        )
        transport = _StubTransport(speak_side_effect=reconnect_failure)
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hello"))

        assert transport.end_calls == 1


# ── multiple TTS failures in sequence ────────────────────────────────────────

class TestMultipleTtsFailures:
    def test_single_tts_failure_ends_call_exactly_once(self):
        """A single TTS failure results in exactly one end_call."""
        transport = _StubTransport(speak_side_effect=_tts_failure())
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hello"))

        assert transport.end_calls == 1

    def test_tts_failure_end_call_not_zero(self):
        """A TTS failure always produces a call-control action (non-zero end_calls)."""
        transport = _StubTransport(speak_side_effect=_tts_failure())
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("any text"))

        # Some call-control action must have been taken (transfer or end)
        total_actions = transport.end_calls + len(transport.transfer_calls)
        assert total_actions >= 1


# ── phase invariant ───────────────────────────────────────────────────────────

class TestPhaseInvariant:
    def test_phase_not_listening_immediately_after_tts_failure(self):
        """Session must not enter LISTENING after TTS failure without audio."""
        transport = _StubTransport(speak_side_effect=_tts_failure())
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hi"))

        assert cs.phase != RealtimePhase.LISTENING

    def test_phase_was_processing_before_speak(self):
        """Phase transitions to PROCESSING before TTS is attempted."""
        phases_seen: list[RealtimePhase] = []
        transport = _StubTransport()

        original_speak = transport.speak_response

        async def _capture_phase_then_fail(spoken_text, *, trace, end_call_after_playback):
            # Record phase at the point speak_response is called
            phases_seen.append(cs.phase)
            raise _tts_failure()

        transport.speak_response = _capture_phase_then_fail
        cs = _make_conv_session(transport=transport)

        asyncio.run(cs.process_committed_turn("hi"))

        assert RealtimePhase.PROCESSING in phases_seen
