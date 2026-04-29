# app/realtime/conversation_session.py
"""Conversation orchestration extracted from the WebSocket handler.

:class:`ConversationSession` owns the per-call business flow that used
to be inlined in ``twilio_media_ws``:

* committed-turn dispatch into :class:`TurnEngine`
* session save/load coordination
* payment auto-check scheduling
* barge-in decision policy (delegated to
  :func:`app.realtime.barge_in_policy.is_actionable_barge_in`)
* response-text building from a :class:`TurnOutput`

The transport (Twilio WebSocket loop) injects a :class:`VoiceTransport`
adapter so this class can request audio playback, transfers, and
call-end side effects without touching Twilio frames directly.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from app.realtime.barge_in_policy import is_actionable_barge_in
from app.realtime.realtime_conversation_state import RealtimePhase
from app.realtime.tts_exceptions import TTSFailureError
from app.realtime.voice_transport import VoiceTransport
from app.services.payment_auto_check_scheduler import (
    PaymentAutoCheckConfig,
    PaymentAutoCheckScheduler,
)
from app.session.repository import (
    load_session as _default_load_session,
    save_session as _default_save_session,
)
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState

# Response keys emitted right after sending a checkout/payment link.
# Any of these arms the auto-confirm timer so the agent silently probes
# payment status ~50 s later without needing a user prompt.
PAYMENT_LINK_SENT_KEYS: frozenset[str] = frozenset({
    "ask_delivery_address_method",
    "checkout_link_sent",
    "payment_link_sent",
})

# States where an automatic payment-status probe is meaningful.
PAYMENT_AWAITING_STATES: frozenset[ConversationState] = frozenset({
    ConversationState.WAITING_FOR_PAYMENT,
    ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
})

# Initial delay before the first auto-confirm probe.  Subsequent probes
# use exponential backoff (see PaymentAutoCheckConfig).
PAYMENT_AUTO_CHECK_DELAY_SECONDS: int = 30

AUTO_PAYMENT_CHECK_TEXT = "__auto_payment_check__"


def _normalize_response_text(text: str | None) -> str:
    return " ".join((text or "").split()).strip()


def build_response_texts(turn_output: Any) -> tuple[str, str]:
    """Return ``(internal_text, spoken_text)`` from a TurnOutput.

    Falls back to the internal text when the spoken variant is empty,
    matching the prior behavior that lived inside
    ``voice_stream_server._build_response_texts``.
    """
    internal_text = _normalize_response_text(
        getattr(turn_output, "internal_response_text", None)
    )
    spoken_text = _normalize_response_text(
        getattr(turn_output, "spoken_response_text", None)
    )

    if not internal_text:
        raise ValueError(
            f"TurnEngine returned empty internal_response_text for response_key="
            f"{getattr(turn_output, 'response_key', 'unknown')}"
        )

    if not spoken_text:
        spoken_text = internal_text

    return internal_text, spoken_text


class ConversationSession:
    """Per-call conversation orchestrator.

    The instance is created when the WebSocket handler receives the
    Twilio ``start`` event and lives until the WS closes.  It is the
    single entry point for committed user turns; the transport never
    calls :class:`TurnEngine` directly.
    """

    def __init__(
        self,
        *,
        app_session: Session | None,
        engine: Any,
        transport: VoiceTransport,
        save_session_fn: Callable[[Session], None] = _default_save_session,
        load_session_fn: Callable[[str, str], Session] = _default_load_session,
        payment_auto_check_delay_seconds: int = PAYMENT_AUTO_CHECK_DELAY_SECONDS,
    ) -> None:
        self.app_session = app_session
        self.engine = engine
        self.transport = transport
        self._save_session_fn = save_session_fn
        self._load_session_fn = load_session_fn

        self.phase: RealtimePhase = RealtimePhase.LISTENING
        self.pending_interrupt_text: str | None = None
        self._processing_lock = asyncio.Lock()
        self.should_end_call_after_playback = False
        self.pending_transfer_number: str | None = None

        async def _payment_probe() -> bool:
            if (
                self.app_session is None
                or self.app_session.conversation_state not in PAYMENT_AWAITING_STATES
            ):
                return False
            await self.process_committed_turn(AUTO_PAYMENT_CHECK_TEXT)
            return (
                self.app_session is not None
                and self.app_session.conversation_state in PAYMENT_AWAITING_STATES
            )

        self._payment_scheduler = PaymentAutoCheckScheduler(
            get_phase=lambda: self.phase,
            dispatch_probe=_payment_probe,
            config=PaymentAutoCheckConfig(
                initial_delay=payment_auto_check_delay_seconds,
            ),
        )

    # ------------------------------------------------------------------ phase
    def set_phase_listening(self) -> None:
        self.phase = RealtimePhase.LISTENING

    def set_phase_speaking(self) -> None:
        self.phase = RealtimePhase.SPEAKING

    def set_phase_processing(self) -> None:
        self.phase = RealtimePhase.PROCESSING

    def is_speaking(self) -> bool:
        return self.phase == RealtimePhase.SPEAKING

    def is_processing(self) -> bool:
        return self.phase == RealtimePhase.PROCESSING

    def is_listening(self) -> bool:
        return self.phase == RealtimePhase.LISTENING

    # ------------------------------------------------ session attachment
    def load_app_session(
        self,
        call_sid: str | None,
        restaurant_id: str,
    ) -> Session | None:
        """Load and attach the persisted app session for a live call."""
        if not call_sid:
            self.app_session = None
            return None

        self.app_session = self._load_session_fn(call_sid, restaurant_id)
        return self.app_session

    # ---------------------------------------------------- payment auto-check

    @property
    def _payment_check_task(self) -> asyncio.Task | None:
        """Compatibility shim: exposes the scheduler's live task for tests."""
        state = self._payment_scheduler.state
        return state.task if state is not None else None

    async def cancel_payment_auto_check(self) -> None:
        self._payment_scheduler.cancel()

    async def schedule_payment_auto_check(self) -> None:
        """Arm the payment auto-check scheduler (single-flight, idempotent).

        Starts exponential-backoff probing after the configured initial
        delay.  Calling this while probing is already active is a no-op,
        so it is safe to call unconditionally after each payment-link
        response.  The scheduler stops automatically when:
        * the probe returns False (payment confirmed / session gone), or
        * the maximum attempt count is reached (escalation).
        Cancelled explicitly when the WebSocket closes via
        ``cancel_payment_auto_check()``.
        """
        self._payment_scheduler.schedule()

    # ----------------------------------------------------- committed turns
    async def process_committed_turn(self, user_text: str) -> None:
        """Drive a single committed user utterance through the engine.

        Behavior preserved verbatim from the previous closure inside
        ``twilio_media_ws``: barge-in policy first, then dispatch into
        :class:`TurnEngine`, persist the session, and ask the transport
        to speak/transfer/end-call as appropriate.
        """
        if self.app_session is None:
            return

        cleaned = " ".join(user_text.split()).strip()
        if not cleaned:
            return

        if self.phase == RealtimePhase.SPEAKING:
            if self.transport.is_barge_in_disabled():
                return

            if not is_actionable_barge_in(self.app_session, cleaned):
                self.transport.debug_log(
                    "[BARGE IN IGNORED]",
                    {
                        "state": getattr(self.app_session, "conversation_state", None),
                        "text": cleaned,
                    },
                )
                return

            await self.transport.interrupt_playback(reason="actionable_user_turn")
            self.set_phase_listening()

        if self.phase == RealtimePhase.PROCESSING:
            self.pending_interrupt_text = cleaned
            self.transport.debug_log(
                "[INTERRUPT BUFFERED]",
                {"text": cleaned},
            )
            return

        async with self._processing_lock:
            self.set_phase_processing()

            trace = self.transport.begin_turn_trace(user_text=cleaned)

            turn_output = self.engine.process_turn(
                session=self.app_session,
                user_text=cleaned,
                trace=trace,
            )

            self._save_session_fn(self.app_session)

            responder_start = time.perf_counter()
            internal_response_text, spoken_response_text = build_response_texts(turn_output)
            responder_end = time.perf_counter()

            end_after_playback = bool(
                getattr(turn_output, "end_call_after_playback", False)
            )
            self.should_end_call_after_playback = end_after_playback
            self.pending_transfer_number = (
                getattr(turn_output, "transfer_call_to_number", None) or None
            )

            self.transport.annotate_response_trace(
                trace,
                responder_start_monotonic=responder_start,
                responder_end_monotonic=responder_end,
                response_key=turn_output.response_key,
                internal_response_text=internal_response_text,
                spoken_response_text=spoken_response_text,
                end_call_after_playback=end_after_playback,
            )

            print(
                "[TURN OUTPUT]",
                {
                    "response_key": turn_output.response_key,
                    "session_state": getattr(self.app_session, "conversation_state", None),
                    "end_call_after_playback": end_after_playback,
                    "pending_transfer_number": self.pending_transfer_number,
                },
            )

            if self.pending_transfer_number:
                target = self.pending_transfer_number
                self.pending_transfer_number = None
                self.should_end_call_after_playback = False
                print(
                    "[TURN OUTPUT] Initiating immediate transfer handoff",
                    {
                        "target": target,
                    },
                )
                await self.transport.transfer_call(target)
                return

            try:
                await self.transport.speak_response(
                    spoken_response_text,
                    trace=trace,
                    end_call_after_playback=self.should_end_call_after_playback,
                )
            except TTSFailureError as exc:
                print(
                    "[TTS_FAILURE_HANDLED]",
                    {
                        "attempts": exc.attempts,
                        "provider": exc.provider,
                        "error": str(exc),
                    },
                )
                handled = await self.handle_playback_failure(
                    end_call_after_playback=self.should_end_call_after_playback,
                )
                if not handled:
                    # No pending transfer/end-call flag — end gracefully to
                    # prevent the caller from sitting in silence.
                    await self.transport.end_call()
                return

            # Arm the single-flight payment probe when a checkout/payment
            # link has just been spoken.  No-op if already scheduled.
            if turn_output.response_key in PAYMENT_LINK_SENT_KEYS:
                await self.schedule_payment_auto_check()

            if self.pending_interrupt_text and self.phase == RealtimePhase.LISTENING:
                buffered = self.pending_interrupt_text
                self.pending_interrupt_text = None
                await self.process_committed_turn(buffered)

    # ------------------------------------------------- post-playback drain
    async def on_playback_completed(self) -> bool:
        """Called by the transport after Twilio acks the playback mark.

        Returns ``True`` when the transport should treat the post-mark
        side effects (transfer / end_call / drain pending interrupt) as
        already handled here.  The transport still owns the WebSocket
        cleanup itself.
        """
        if self.phase == RealtimePhase.SPEAKING:
            self.set_phase_listening()

        if self.pending_transfer_number:
            target = self.pending_transfer_number
            self.pending_transfer_number = None
            self.should_end_call_after_playback = False
            await self.transport.transfer_call(target)
            return True

        if self.should_end_call_after_playback:
            self.should_end_call_after_playback = False
            await self.transport.end_call()
            return True

        if self.pending_interrupt_text:
            buffered = self.pending_interrupt_text
            self.pending_interrupt_text = None
            await self.process_committed_turn(buffered)
            return True

        return False

    async def handle_playback_failure(
        self,
        *,
        end_call_after_playback: bool,
    ) -> bool:
        """Apply call-control side effects when playback cannot complete."""
        if self.pending_transfer_number:
            target = self.pending_transfer_number
            self.pending_transfer_number = None
            self.should_end_call_after_playback = False
            await self.transport.transfer_call(target)
            return True

        if end_call_after_playback or self.should_end_call_after_playback:
            self.should_end_call_after_playback = False
            await self.transport.end_call()
            return True

        return False

