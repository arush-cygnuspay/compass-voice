# app/realtime/conversation_session.py
"""Conversation orchestration extracted from the WebSocket handler.

:class:`ConversationSession` owns the per-call business flow:

* committed-turn dispatch into :class:`TurnEngine`
* session save/load coordination
* payment auto-check scheduling
* barge-in decision policy (delegated to
  :func:`app.realtime.barge_in_policy.evaluate_barge_in_candidate`)
* response-text building from a :class:`TurnOutput`

The transport (Twilio WebSocket loop) injects a :class:`VoiceTransport`
adapter so this class can request audio playback, transfers, and
call-end side effects without touching Twilio frames directly.
"""
from __future__ import annotations

import asyncio
import collections
import time
from typing import Any, Callable

from app.config.realtime import get_realtime_turn_config
from app.realtime.barge_in_policy import evaluate_barge_in_candidate, is_actionable_barge_in
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

PAYMENT_AUTO_CHECK_DELAY_SECONDS: int = 30
AUTO_PAYMENT_CHECK_TEXT = "__auto_payment_check__"


def _normalize_response_text(text: str | None) -> str:
    return " ".join((text or "").split()).strip()


def build_response_texts(turn_output: Any) -> tuple[str, str]:
    """Return ``(internal_text, spoken_text)`` from a TurnOutput."""
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

    Created when the WebSocket handler receives the Twilio ``start`` event;
    lives until the WS closes.  Single entry point for committed user turns —
    the transport never calls :class:`TurnEngine` directly.
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
        self._processing_lock = asyncio.Lock()
        self.should_end_call_after_playback = False
        self.pending_transfer_number: str | None = None

        # Turn identity — incremented monotonically per session.
        # active_turn_id  : turn currently being processed (0 = none yet)
        # last_committed_turn_id: last turn_id that was accepted for FSM processing
        self.active_turn_id: int = 0
        self.last_committed_turn_id: int = 0

        # Bounded queue for turns that arrive while FSM is processing.
        # maxlen enforced by config at first process call; pre-set here for safety.
        self._pending_queue: collections.deque[str] = collections.deque(maxlen=2)

        # True only while the FSM is actively computing a response. Distinct from
        # phase so that post-TTS-failure state (phase=PROCESSING, FSM idle) still
        # allows the next turn to be processed rather than buffered.
        self._turn_processing: bool = False

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

    # ------------------------------------------------- backward compat shim
    @property
    def pending_interrupt_text(self) -> str | None:
        """Compatibility property — returns the front of the pending queue."""
        return self._pending_queue[0] if self._pending_queue else None

    @pending_interrupt_text.setter
    def pending_interrupt_text(self, value: str | None) -> None:
        """Compatibility setter — replaces queue with a single entry."""
        self._pending_queue.clear()
        if value:
            self._pending_queue.append(value)

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
        """Arm the payment auto-check scheduler (single-flight, idempotent)."""
        self._payment_scheduler.schedule()

    # ----------------------------------------------------- committed turns
    async def process_committed_turn(
        self,
        user_text: str,
        *,
        turn_id: int | None = None,
        barge_in_audio_ms: float | None = None,
        transcript_confidence: float | None = None,
    ) -> None:
        """Drive a single committed user utterance through the engine.

        All phase/state checks happen UNDER the processing lock so there is
        no TOCTOU window between the check and the FSM call.

        Args:
            user_text:             Committed transcript text.
            turn_id:               Monotonic turn counter from TurnCommitController.
                                   None for system-generated turns (payment probes).
            barge_in_audio_ms:     Duration (ms) of audio captured before commit.
                                   Used for barge-in acceptance gating.
            transcript_confidence: STT confidence score (0–1) if available.
        """
        if self.app_session is None:
            return

        cleaned = " ".join(user_text.split()).strip()
        if not cleaned:
            return

        config = get_realtime_turn_config()
        session_id = getattr(self.app_session, "session_id", "") or ""

        # ── Acquire the processing lock with a timeout ───────────────────────
        lock_start = time.monotonic()
        lock_acquired = False
        try:
            await asyncio.wait_for(
                self._processing_lock.acquire(),
                timeout=config.turn_lock_timeout_s,
            )
            lock_acquired = True
        except asyncio.TimeoutError:
            print(
                "[turn_lock_timeout]",
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "transcript": cleaned,
                    "phase": self.phase,
                    "timeout_s": config.turn_lock_timeout_s,
                },
            )
            return

        lock_wait_ms = (time.monotonic() - lock_start) * 1000.0
        if lock_wait_ms > 50:
            print(
                "[turn_lock_wait_ms]",
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "lock_wait_ms": round(lock_wait_ms, 1),
                    "phase": self.phase,
                },
            )

        try:
            # ── Phase-aware routing (all under the lock) ─────────────────────

            if self.phase == RealtimePhase.SPEAKING:
                # --- Controlled barge-in evaluation ---
                if self.transport.is_barge_in_disabled():
                    self.transport.debug_log(
                        "[barge_in_rejected]",
                        {
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "reason": "barge_in_globally_disabled",
                            "transcript": cleaned,
                        },
                    )
                    return

                # Playback_started_at lives in the transport layer but we
                # forward it via the VoiceTransport interface if available.
                playback_started_at = getattr(
                    self.transport, "_playback_started_at", None
                )

                decision = evaluate_barge_in_candidate(
                    session=self.app_session,
                    text=cleaned,
                    audio_duration_ms=barge_in_audio_ms,
                    confidence=transcript_confidence,
                    playback_started_at=playback_started_at,
                    config=config,
                )

                if not decision.accepted:
                    self.transport.debug_log(
                        "[barge_in_rejected]",
                        {
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "reason": decision.reason,
                            "transcript": cleaned,
                            "audio_ms": barge_in_audio_ms,
                            "confidence": transcript_confidence,
                            "phase": self.phase,
                        },
                    )
                    return

                # Barge-in accepted — log, interrupt TTS, continue as normal turn
                print(
                    "[barge_in_accepted]",
                    {
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "reason": decision.reason,
                        "transcript": cleaned,
                        "audio_ms": barge_in_audio_ms,
                        "confidence": transcript_confidence,
                    },
                )
                await self.transport.interrupt_playback(reason="actionable_user_turn")
                self.set_phase_listening()

            if self.phase == RealtimePhase.PROCESSING and self._turn_processing:
                # Buffer into the bounded pending queue.
                if len(self._pending_queue) >= config.max_pending_interrupt_queue:
                    print(
                        "[pending_queue_overflow_dropped]",
                        {
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "transcript": cleaned,
                            "queue_depth": len(self._pending_queue),
                            "max": config.max_pending_interrupt_queue,
                        },
                    )
                    return

                # Deduplicate against most-recent queued entry.
                if not self._pending_queue or self._pending_queue[-1] != cleaned:
                    self._pending_queue.append(cleaned)
                    self.transport.debug_log(
                        "[INTERRUPT BUFFERED]",
                        {
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "text": cleaned,
                            "queue_depth": len(self._pending_queue),
                        },
                    )
                return

            # ── Stale-turn guard ─────────────────────────────────────────────
            # Only applies to turns that carry an explicit turn_id from the
            # TurnCommitController; system-generated probes (turn_id=None) skip.
            if turn_id is not None and turn_id <= self.last_committed_turn_id:
                print(
                    "[stale_turn_event_ignored]",
                    {
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "last_committed_turn_id": self.last_committed_turn_id,
                        "transcript": cleaned,
                    },
                )
                return

            if turn_id is not None:
                self.last_committed_turn_id = turn_id
                self.active_turn_id = turn_id

            # ── FSM processing ───────────────────────────────────────────────
            self.set_phase_processing()
            self._turn_processing = True

            print(
                "[turn_processing_started]",
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "transcript": cleaned,
                    "state": getattr(self.app_session, "conversation_state", None),
                },
            )

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
                "[turn_processing_finished]",
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
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
                    "[turn_processing_finished] Initiating immediate transfer handoff",
                    {"target": target},
                )
                await self.transport.transfer_call(target)
                return

            # ── Human-like response pause ────────────────────────────────────
            response_delay_s = config.post_user_turn_response_delay_ms / 1000.0
            if response_delay_s > 0:
                await asyncio.sleep(response_delay_s)

            print(
                "[assistant_tts_started]",
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "spoken_text_chars": len(spoken_response_text),
                },
            )

            # Phase stays PROCESSING until the transport confirms audio has
            # started.  voice_stream_server calls set_phase_speaking() on its
            # first Deepgram audio chunk, so we never pre-announce SPEAKING.
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
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "attempts": exc.attempts,
                        "provider": exc.provider,
                        "error": str(exc),
                    },
                )
                # No audio was delivered — keep phase as PROCESSING until
                # call termination so callers never see a spurious LISTENING.
                handled = await self.handle_playback_failure(
                    end_call_after_playback=self.should_end_call_after_playback,
                )
                if not handled:
                    await self.transport.end_call()
                return

            print(
                "[assistant_tts_finished]",
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                },
            )

            if turn_output.response_key in PAYMENT_LINK_SENT_KEYS:
                await self.schedule_payment_auto_check()

        finally:
            self._turn_processing = False
            if lock_acquired:
                self._processing_lock.release()

        # ── Drain pending queue (outside the lock) ───────────────────────────
        depth = 0
        while self._pending_queue and depth < config.max_drain_depth:
            if self.is_speaking():
                # Don't drain while TTS is playing; on_playback_completed will
                # pick up the queue.
                break
            buffered = self._pending_queue.popleft()
            depth += 1
            await self.process_committed_turn(buffered)

    # ------------------------------------------------- post-playback drain
    async def on_playback_completed(self) -> bool:
        """Called by the transport after Twilio acks the playback mark."""
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

        if self._pending_queue:
            buffered = self._pending_queue.popleft()
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
