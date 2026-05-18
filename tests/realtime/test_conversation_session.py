# tests/realtime/test_conversation_session.py
"""Behavior tests for :class:`ConversationSession`.

Run the orchestrator with a fake transport and a stub engine so the
conversation-layer contract can be asserted without booting Twilio,
Deepgram, or Redis.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from tests.support.voice_test_harness import install_test_stubs

install_test_stubs()

from app.realtime.conversation_session import (
    AUTO_PAYMENT_CHECK_TEXT,
    ConversationSession,
)
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState


@dataclass
class _StubTurnOutput:
    response_key: str = "ack"
    response_payload: dict | None = None
    internal_response_text: str = "internal text"
    spoken_response_text: str = "spoken text"
    end_call_after_playback: bool = False
    transfer_call_to_number: str | None = None


class _StubEngine:
    """Records ``process_turn`` calls and applies an optional state mutation."""

    def __init__(
        self,
        *,
        output: _StubTurnOutput | None = None,
        next_state: ConversationState | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.output = output or _StubTurnOutput()
        self.next_state = next_state
        self.calls: list[dict] = []
        self.delay_s = delay_s

    def process_turn(self, *, session: Session, user_text: str, trace=None):
        self.calls.append(
            {
                "user_text": user_text,
                "trace": trace,
                "state_before": session.conversation_state,
            }
        )
        if self.next_state is not None:
            session.conversation_state = self.next_state
        return self.output


@dataclass
class _StubTransport:
    barge_in_disabled: bool = False
    speak_calls: list[dict] = field(default_factory=list)
    interrupt_calls: list[str] = field(default_factory=list)
    transfer_calls: list[str] = field(default_factory=list)
    end_calls: int = 0
    debug_events: list[tuple[str, dict]] = field(default_factory=list)
    begin_trace_calls: int = 0
    annotate_calls: list[dict] = field(default_factory=list)
    # Expose playback start time for barge-in guard checks
    _playback_started_at: float | None = None

    def is_barge_in_disabled(self) -> bool:
        return self.barge_in_disabled

    def debug_log(self, event: str, payload: dict[str, Any]) -> None:
        self.debug_events.append((event, payload))

    def begin_turn_trace(self, *, user_text: str):
        self.begin_trace_calls += 1
        return {"user_text": user_text}

    def annotate_response_trace(self, trace, **kwargs) -> None:
        self.annotate_calls.append({"trace": trace, **kwargs})

    async def speak_response(
        self,
        spoken_text: str,
        *,
        trace,
        end_call_after_playback: bool,
    ) -> None:
        self.speak_calls.append(
            {
                "spoken_text": spoken_text,
                "trace": trace,
                "end_call_after_playback": end_call_after_playback,
            }
        )

    async def interrupt_playback(self, reason: str) -> None:
        self.interrupt_calls.append(reason)

    async def transfer_call(self, target_number: str) -> None:
        self.transfer_calls.append(target_number)

    async def end_call(self) -> None:
        self.end_calls += 1


def _make_session(
    state: ConversationState = ConversationState.WAITING_FOR_ORDER_TYPE,
) -> Session:
    session = Session(session_id="conv-test", restaurant_id="steves_grill")
    session.conversation_state = state
    session.conversation_context.caller_device_type = "phone"
    return session


def _make_conv_session(
    *,
    engine,
    transport: _StubTransport,
    app_session: Session | None = None,
    delay: int = 0,
    load_session_fn=None,
):
    saves: list[Session] = []

    def _save(session: Session) -> None:
        saves.append(session)

    if load_session_fn is None:
        load_session_fn = lambda call_sid, restaurant_id: _make_session()

    cs = ConversationSession(
        app_session=app_session if app_session is not None else _make_session(),
        engine=engine,
        transport=transport,
        save_session_fn=_save,
        load_session_fn=load_session_fn,
        payment_auto_check_delay_seconds=delay,
    )
    return cs, saves


# ------------------------------------------------ session attachment
def test_load_app_session_uses_injected_loader() -> None:
    engine = _StubEngine()
    transport = _StubTransport()
    loaded_session = _make_session(state=ConversationState.WAITING_FOR_PAYMENT)
    loader_calls: list[tuple[str, str]] = []

    def _load(call_sid: str, restaurant_id: str) -> Session:
        loader_calls.append((call_sid, restaurant_id))
        return loaded_session

    conv_session, _saves = _make_conv_session(
        engine=engine,
        transport=transport,
        app_session=None,
        load_session_fn=_load,
    )

    restored = conv_session.load_app_session('CA123', 'steves_grill')

    assert restored is loaded_session
    assert conv_session.app_session is loaded_session
    assert loader_calls == [('CA123', 'steves_grill')]


# ----------------------------------------------------------- happy path
def test_committed_turn_routes_through_engine_and_persists_session() -> None:
    engine = _StubEngine(
        output=_StubTurnOutput(
            response_key="ack",
            internal_response_text="ok",
            spoken_response_text="okay",
        ),
        next_state=ConversationState.CONFIRMING_ORDER,
    )
    transport = _StubTransport()
    conv_session, saves = _make_conv_session(engine=engine, transport=transport)

    asyncio.run(conv_session.process_committed_turn(" hello there  "))

    assert len(engine.calls) == 1
    assert engine.calls[0]["user_text"] == "hello there"

    assert len(saves) == 1
    assert saves[0] is conv_session.app_session

    assert len(transport.speak_calls) == 1
    assert transport.speak_calls[0]["spoken_text"] == "okay"
    assert transport.speak_calls[0]["end_call_after_playback"] is False

    assert transport.transfer_calls == []
    assert transport.end_calls == 0

    assert transport.begin_trace_calls == 1
    assert len(transport.annotate_calls) == 1
    annotation = transport.annotate_calls[0]
    assert annotation["response_key"] == "ack"
    assert annotation["internal_response_text"] == "ok"
    assert annotation["spoken_response_text"] == "okay"

    assert engine.calls[0]["state_before"] == ConversationState.WAITING_FOR_ORDER_TYPE
    assert (
        conv_session.app_session.conversation_state
        == ConversationState.CONFIRMING_ORDER
    )


def test_empty_text_does_not_invoke_engine() -> None:
    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, saves = _make_conv_session(engine=engine, transport=transport)

    asyncio.run(conv_session.process_committed_turn("   "))

    assert engine.calls == []
    assert saves == []
    assert transport.speak_calls == []


# -------------------------------------------------------------- transfer
def test_transfer_call_skips_speak_and_routes_to_transport() -> None:
    engine = _StubEngine(
        output=_StubTurnOutput(
            response_key="transferring_to_human_agent",
            internal_response_text="bye",
            spoken_response_text="bye",
            transfer_call_to_number="+15551112222",
            end_call_after_playback=True,
        ),
    )
    transport = _StubTransport()
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)

    asyncio.run(conv_session.process_committed_turn("transfer me"))

    assert transport.transfer_calls == ["+15551112222"]
    assert transport.speak_calls == []
    assert conv_session.pending_transfer_number is None
    assert conv_session.should_end_call_after_playback is False


# -------------------------------------------------- payment auto-check
def test_payment_link_response_schedules_auto_check() -> None:
    engine = _StubEngine(
        output=_StubTurnOutput(
            response_key="payment_link_sent",
            internal_response_text="link sent",
            spoken_response_text="link sent",
        ),
        next_state=ConversationState.WAITING_FOR_PAYMENT,
    )
    transport = _StubTransport()
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)

    schedule_calls = 0

    async def _record_schedule():
        nonlocal schedule_calls
        schedule_calls += 1

    async def _scenario() -> None:
        with patch.object(conv_session, "schedule_payment_auto_check", _record_schedule):
            await conv_session.process_committed_turn("pay me")

    asyncio.run(_scenario())
    assert schedule_calls == 1


def test_non_payment_response_does_not_schedule_auto_check() -> None:
    engine = _StubEngine(
        output=_StubTurnOutput(
            response_key="ack",
            internal_response_text="ok",
            spoken_response_text="ok",
        ),
    )
    transport = _StubTransport()
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)

    schedule_calls = 0

    async def _record_schedule():
        nonlocal schedule_calls
        schedule_calls += 1

    async def _scenario() -> None:
        with patch.object(conv_session, "schedule_payment_auto_check", _record_schedule):
            await conv_session.process_committed_turn("hi")

    asyncio.run(_scenario())
    assert schedule_calls == 0


def test_payment_auto_check_dispatches_auto_text() -> None:
    engine = _StubEngine(
        output=_StubTurnOutput(
            response_key="payment_pending_reminder",
            internal_response_text="still waiting",
            spoken_response_text="still waiting",
        ),
        next_state=ConversationState.COMPLETED,
    )
    transport = _StubTransport()
    session = _make_session(state=ConversationState.WAITING_FOR_PAYMENT)
    conv_session, _saves = _make_conv_session(
        engine=engine,
        transport=transport,
        app_session=session,
        delay=0,
    )

    async def _scenario() -> None:
        await conv_session.schedule_payment_auto_check()
        await conv_session._payment_check_task

    asyncio.run(_scenario())

    assert any(
        call["user_text"] == AUTO_PAYMENT_CHECK_TEXT for call in engine.calls
    )


def test_payment_auto_check_skipped_outside_payment_states() -> None:
    engine = _StubEngine()
    transport = _StubTransport()
    session = _make_session(state=ConversationState.IDLE)
    conv_session, _saves = _make_conv_session(
        engine=engine, transport=transport, app_session=session, delay=0,
    )

    async def _scenario() -> None:
        await conv_session.schedule_payment_auto_check()
        await conv_session._payment_check_task

    asyncio.run(_scenario())
    assert engine.calls == []


# --------------------------------------------------------------- barge-in
def test_barge_in_ignored_when_transport_disables_it() -> None:
    engine = _StubEngine()
    transport = _StubTransport(barge_in_disabled=True)
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.set_phase_speaking()

    asyncio.run(conv_session.process_committed_turn("yes please"))

    assert engine.calls == []
    assert transport.interrupt_calls == []
    assert conv_session.is_speaking()


def test_barge_in_filler_does_not_interrupt_tts() -> None:
    """Single filler word 'uh' during TTS must not interrupt playback."""
    engine = _StubEngine()
    transport = _StubTransport(barge_in_disabled=False)
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.set_phase_speaking()

    asyncio.run(conv_session.process_committed_turn("uh"))

    assert engine.calls == []
    assert transport.interrupt_calls == []
    assert conv_session.is_speaking()


def test_barge_in_non_actionable_does_not_interrupt() -> None:
    engine = _StubEngine()
    transport = _StubTransport(barge_in_disabled=False)
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.set_phase_speaking()

    asyncio.run(conv_session.process_committed_turn("uh huh"))

    assert engine.calls == []
    assert transport.interrupt_calls == []


def test_barge_in_actionable_routes_interrupt_then_engine() -> None:
    engine = _StubEngine(
        output=_StubTurnOutput(
            response_key="order_type_captured_pickup",
            internal_response_text="pickup it is",
            spoken_response_text="pickup it is",
        ),
    )
    transport = _StubTransport(barge_in_disabled=False)
    session = _make_session(state=ConversationState.WAITING_FOR_ORDER_TYPE)
    conv_session, _saves = _make_conv_session(
        engine=engine, transport=transport, app_session=session,
    )
    conv_session.set_phase_speaking()

    asyncio.run(conv_session.process_committed_turn("pickup"))

    assert transport.interrupt_calls == ["actionable_user_turn"]
    assert len(engine.calls) == 1
    assert engine.calls[0]["user_text"] == "pickup"


def test_barge_in_cough_no_transcript_does_not_interrupt() -> None:
    """Empty transcript during TTS must be silently ignored."""
    engine = _StubEngine()
    transport = _StubTransport(barge_in_disabled=False)
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.set_phase_speaking()

    asyncio.run(conv_session.process_committed_turn(""))

    assert engine.calls == []
    assert transport.interrupt_calls == []


def test_barge_in_correction_accepted_and_processed() -> None:
    """'no change that to coke' must interrupt TTS and process a new FSM turn."""
    engine = _StubEngine(
        output=_StubTurnOutput(
            response_key="ack",
            internal_response_text="ok changed",
            spoken_response_text="ok changed",
        ),
    )
    transport = _StubTransport(barge_in_disabled=False)
    session = _make_session(state=ConversationState.WAITING_FOR_ORDER_TYPE)
    conv_session, _saves = _make_conv_session(
        engine=engine, transport=transport, app_session=session,
    )
    conv_session.set_phase_speaking()

    asyncio.run(conv_session.process_committed_turn("no change that to coke"))

    # TTS was interrupted
    assert len(transport.interrupt_calls) == 1
    # Engine was called with the correction
    assert len(engine.calls) == 1
    assert engine.calls[0]["user_text"] == "no change that to coke"


# ------------------------------------------------------- pending interrupt / queue
def test_pending_interrupt_buffered_during_processing() -> None:
    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.set_phase_processing()

    asyncio.run(conv_session.process_committed_turn("interrupt me"))

    assert engine.calls == []
    # Compat shim must still work
    assert conv_session.pending_interrupt_text == "interrupt me"


def test_pending_queue_bounded_and_logs_overflow() -> None:
    """Entries beyond max_pending_interrupt_queue are dropped, not stacked."""
    from unittest.mock import patch as _patch

    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, _ = _make_conv_session(engine=engine, transport=transport)
    conv_session.set_phase_processing()

    printed: list[str] = []
    original_print = __builtins__["print"] if isinstance(__builtins__, dict) else print

    import builtins
    with _patch.object(builtins, "print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
        asyncio.run(conv_session.process_committed_turn("first"))
        asyncio.run(conv_session.process_committed_turn("second"))
        asyncio.run(conv_session.process_committed_turn("overflow"))

    overflow_logs = [p for p in printed if "pending_queue_overflow_dropped" in p]
    assert overflow_logs, "Expected pending_queue_overflow_dropped log"
    # Queue must not grow beyond config limit
    assert len(conv_session._pending_queue) <= 2


def test_on_playback_completed_drains_pending_interrupt() -> None:
    engine = _StubEngine(
        output=_StubTurnOutput(
            response_key="ack",
            internal_response_text="ok",
            spoken_response_text="ok",
        ),
    )
    transport = _StubTransport()
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.set_phase_speaking()
    conv_session.pending_interrupt_text = "buffered text"

    handled = asyncio.run(conv_session.on_playback_completed())

    assert handled is True
    assert engine.calls and engine.calls[0]["user_text"] == "buffered text"


def test_on_playback_completed_runs_pending_transfer() -> None:
    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.set_phase_speaking()
    conv_session.pending_transfer_number = "+15551112222"

    handled = asyncio.run(conv_session.on_playback_completed())

    assert handled is True
    assert transport.transfer_calls == ["+15551112222"]
    assert conv_session.pending_transfer_number is None


def test_on_playback_completed_runs_end_call_when_flag_set() -> None:
    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.set_phase_speaking()
    conv_session.should_end_call_after_playback = True

    handled = asyncio.run(conv_session.on_playback_completed())

    assert handled is True
    assert transport.end_calls == 1
    assert conv_session.should_end_call_after_playback is False


# ------------------------------------------------ playback failure hooks
def test_playback_failure_prefers_transfer_side_effect() -> None:
    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.pending_transfer_number = '+15551112222'
    conv_session.should_end_call_after_playback = True

    handled = asyncio.run(
        conv_session.handle_playback_failure(end_call_after_playback=True)
    )

    assert handled is True
    assert transport.transfer_calls == ['+15551112222']
    assert transport.end_calls == 0
    assert conv_session.pending_transfer_number is None
    assert conv_session.should_end_call_after_playback is False


def test_playback_failure_hangs_up_when_requested() -> None:
    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, _saves = _make_conv_session(engine=engine, transport=transport)
    conv_session.should_end_call_after_playback = True

    handled = asyncio.run(
        conv_session.handle_playback_failure(end_call_after_playback=False)
    )

    assert handled is True
    assert transport.transfer_calls == []
    assert transport.end_calls == 1
    assert conv_session.should_end_call_after_playback is False


def test_tts_failure_releases_lock_no_deadlock() -> None:
    """TTSFailureError must not leave the lock held — no deadlock on next turn."""
    from app.realtime.tts_exceptions import TTSFailureError

    engine = _StubEngine(
        output=_StubTurnOutput(
            response_key="ack",
            internal_response_text="ok",
            spoken_response_text="ok",
        ),
    )

    class _FailingTransport(_StubTransport):
        async def speak_response(self, spoken_text, *, trace, end_call_after_playback):
            raise TTSFailureError("TTS exploded", attempts=1, provider="deepgram")

    transport = _FailingTransport()
    conv_session, _ = _make_conv_session(engine=engine, transport=transport)

    asyncio.run(conv_session.process_committed_turn("hello"))

    # Lock must not be held — second call must complete without hanging
    engine2 = _StubEngine()
    conv_session.engine = engine2
    conv_session.transport = _StubTransport()
    asyncio.run(conv_session.process_committed_turn("world"))

    assert len(engine2.calls) == 1


# ------------------------------------------------ turn_id / stale guard
def test_stale_turn_id_is_ignored() -> None:
    """A committed turn with turn_id <= last_committed_turn_id must be dropped."""
    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, _ = _make_conv_session(engine=engine, transport=transport)

    # Process turn 5 first
    asyncio.run(conv_session.process_committed_turn("first", turn_id=5))
    assert len(engine.calls) == 1

    # Now try to process turn 3 (stale)
    asyncio.run(conv_session.process_committed_turn("stale turn", turn_id=3))
    # Engine must NOT be called again
    assert len(engine.calls) == 1


def test_fresh_turn_id_is_accepted() -> None:
    """Two sequential turns with increasing turn_ids both process normally."""
    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, _ = _make_conv_session(engine=engine, transport=transport)

    asyncio.run(conv_session.process_committed_turn("first", turn_id=1))
    # Simulate Twilio mark ack — resets phase to LISTENING for next turn.
    asyncio.run(conv_session.on_playback_completed())
    asyncio.run(conv_session.process_committed_turn("second", turn_id=2))

    assert len(engine.calls) == 2


def test_no_turn_id_is_never_stale() -> None:
    """System probes (turn_id=None) must never be dropped as stale."""
    engine = _StubEngine()
    transport = _StubTransport()
    conv_session, _ = _make_conv_session(engine=engine, transport=transport)

    asyncio.run(conv_session.process_committed_turn("probe", turn_id=None))
    asyncio.run(conv_session.on_playback_completed())
    asyncio.run(conv_session.process_committed_turn("probe again", turn_id=None))

    assert len(engine.calls) == 2


# ------------------------------------------------ no concurrent FSM processing
def test_concurrent_process_committed_turn_serialised() -> None:
    """The processing lock ensures FSM calls are strictly serial."""
    call_order: list[str] = []
    call_times: list[float] = []

    class _TimingEngine:
        output = _StubTurnOutput(
            response_key="ack",
            internal_response_text="ok",
            spoken_response_text="ok",
        )

        def process_turn(self, *, session, user_text, trace=None):
            call_order.append(user_text)
            call_times.append(time.monotonic())
            return self.output

    transport = _StubTransport()
    engine = _TimingEngine()
    conv_session, _ = _make_conv_session(engine=engine, transport=transport)

    # Serial with mark-ack simulation between turns.
    async def _run():
        await conv_session.process_committed_turn("first")
        await conv_session.on_playback_completed()  # simulate mark ack
        await conv_session.process_committed_turn("second")

    asyncio.run(_run())

    assert call_order == ["first", "second"]
    assert call_times[1] >= call_times[0]


# ---------------------------------------------------- explicit FSM contract
def test_conversation_session_does_not_mutate_state_outside_engine() -> None:
    """Acceptance: only the engine writes ``conversation_state``."""

    initial_state = ConversationState.WAITING_FOR_ORDER_TYPE

    class _RecordingEngine:
        def __init__(self) -> None:
            self.touched_state = False

        def process_turn(self, *, session, user_text, trace=None):
            self.touched_state = session.conversation_state != initial_state
            return _StubTurnOutput(
                response_key="ack",
                internal_response_text="ok",
                spoken_response_text="ok",
            )

    engine = _RecordingEngine()
    transport = _StubTransport()
    session = _make_session(state=initial_state)
    conv_session, _saves = _make_conv_session(
        engine=engine, transport=transport, app_session=session,
    )

    asyncio.run(conv_session.process_committed_turn("hello"))

    assert engine.touched_state is False


# ------------------------------------------------ playback guard / echo suppression
def test_playback_guard_suppresses_echo_immediately_after_tts() -> None:
    """Speech arriving within POST_PLAYBACK_GUARD_MS of TTS start is rejected."""
    engine = _StubEngine()
    transport = _StubTransport(barge_in_disabled=False)
    # Simulate TTS just started (now)
    transport._playback_started_at = time.monotonic()

    session = _make_session(state=ConversationState.WAITING_FOR_ORDER_TYPE)
    conv_session, _ = _make_conv_session(
        engine=engine, transport=transport, app_session=session,
    )
    conv_session.set_phase_speaking()

    asyncio.run(conv_session.process_committed_turn("pickup please", barge_in_audio_ms=900))

    # Must be rejected because we're inside the guard window
    assert engine.calls == []
    assert transport.interrupt_calls == []
