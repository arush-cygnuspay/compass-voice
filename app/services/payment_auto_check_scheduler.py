# app/services/payment_auto_check_scheduler.py
"""Single-flight payment auto-check scheduler.

Replaces the self-rescheduling loop that previously lived in
ConversationSession._run_payment_auto_check with a bounded poller
that applies exponential backoff and escalates after a configurable
number of failed probes.

Usage
-----
Construct one scheduler per call session inside ConversationSession:

    scheduler = PaymentAutoCheckScheduler(
        get_phase=lambda: self.phase,
        dispatch_probe=self._payment_probe,
    )

Then call scheduler.schedule() to arm it and scheduler.cancel() to
tear it down on disconnect.  The scheduler is single-flight: calling
schedule() while a task is already running is a no-op.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.realtime.realtime_conversation_state import RealtimePhase

logger = logging.getLogger(__name__)

# ── default config values ─────────────────────────────────────────────────────

INITIAL_DELAY_SECONDS: float = 30.0
BACKOFF_MULTIPLIER: float = 2.0
MAX_DELAY_SECONDS: float = 300.0
MAX_PROBE_ATTEMPTS: int = 8
GUARD_RETRY_DELAY_SECONDS: float = 10.0


# ── config / state ────────────────────────────────────────────────────────────

@dataclass
class PaymentAutoCheckConfig:
    """Tunable knobs for a PaymentAutoCheckScheduler instance."""

    initial_delay: float = INITIAL_DELAY_SECONDS
    backoff_multiplier: float = BACKOFF_MULTIPLIER
    max_delay: float = MAX_DELAY_SECONDS
    max_attempts: int = MAX_PROBE_ATTEMPTS
    guard_retry_delay: float = GUARD_RETRY_DELAY_SECONDS


@dataclass
class PaymentAutoCheckState:
    """Runtime state for one scheduling lifecycle."""

    attempt: int = 0
    last_probe_at: float | None = None  # monotonic seconds; None until first probe
    next_delay: float = INITIAL_DELAY_SECONDS
    escalated: bool = False
    task: asyncio.Task | None = field(default=None, repr=False)


# ── scheduler ────────────────────────────────────────────────────────────────

class PaymentAutoCheckScheduler:
    """Single-flight payment auto-check scheduler for one call session.

    Lifecycle
    ---------
    * ``schedule()`` — idempotent; starts the polling task if not already
      alive.  Duplicate calls are safe (single-flight guarantee).
    * ``cancel()`` — cancels the task and resets state; safe to call when
      nothing is scheduled.
    * ``is_scheduled()`` — True while the background task is alive.

    Probe contract
    --------------
    ``dispatch_probe`` must be an async callable that:

    * Checks whether the session is still in a payment-awaiting state.
    * Fires the auto-check turn if appropriate.
    * Returns ``True`` to continue probing, ``False`` to stop.

    Phase guard
    -----------
    If the session phase is not ``LISTENING`` when a probe slot arrives
    (e.g. TTS is playing or a user turn is being processed), the probe is
    deferred by ``guard_retry_delay`` seconds and the backoff counter is
    *not* advanced.  This prevents disruptive mid-utterance probes.
    """

    def __init__(
        self,
        *,
        get_phase: Callable[[], RealtimePhase],
        dispatch_probe: Callable[[], Awaitable[bool]],
        on_escalate: Callable[[], Awaitable[None]] | None = None,
        config: PaymentAutoCheckConfig | None = None,
    ) -> None:
        self._get_phase = get_phase
        self._dispatch_probe = dispatch_probe
        self._on_escalate = on_escalate
        self._config = config or PaymentAutoCheckConfig()
        self._state: PaymentAutoCheckState | None = None

    # ----------------------------------------------------------------- public

    def schedule(self) -> None:
        """Arm the auto-check task.  No-op if already running."""
        if self._is_task_alive():
            return
        state = PaymentAutoCheckState(next_delay=self._config.initial_delay)
        state.task = asyncio.create_task(self._run(state))
        self._state = state

    def cancel(self) -> None:
        """Cancel any running task and clear scheduler state."""
        if self._is_task_alive():
            assert self._state is not None
            self._state.task.cancel()  # type: ignore[union-attr]
        self._state = None

    def is_scheduled(self) -> bool:
        """True while the background polling task is alive."""
        return self._is_task_alive()

    @property
    def state(self) -> PaymentAutoCheckState | None:
        """Current runtime state; ``None`` when no task has been scheduled."""
        return self._state

    # --------------------------------------------------------------- internal

    def _is_task_alive(self) -> bool:
        return (
            self._state is not None
            and self._state.task is not None
            and not self._state.task.done()
        )

    async def _run(self, state: PaymentAutoCheckState) -> None:
        config = self._config

        while True:
            # ── wait for the current backoff delay ───────────────────────────
            try:
                await asyncio.sleep(state.next_delay)
            except asyncio.CancelledError:
                return

            # ── phase guard ──────────────────────────────────────────────────
            # Only probe when the session is idle (LISTENING).  If TTS is
            # playing or a user turn is being processed, wait a short
            # dead-zone and retry without advancing the backoff sequence.
            if self._get_phase() != RealtimePhase.LISTENING:
                try:
                    await asyncio.sleep(config.guard_retry_delay)
                except asyncio.CancelledError:
                    return
                continue  # re-enter loop; attempt counter unchanged

            # ── fire probe ───────────────────────────────────────────────────
            state.last_probe_at = time.monotonic()
            state.attempt += 1

            should_continue = True
            try:
                should_continue = await self._dispatch_probe()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    "payment_auto_check probe raised unexpectedly "
                    "(attempt=%d): %s",
                    state.attempt,
                    exc,
                )
                # Treat as transient; continue with backoff.

            if not should_continue:
                logger.info(
                    "payment_auto_check stopped after %d attempt(s) "
                    "(payment resolved or session gone)",
                    state.attempt,
                )
                return

            # ── escalation ───────────────────────────────────────────────────
            if state.attempt >= config.max_attempts:
                state.escalated = True
                logger.warning(
                    "payment_auto_check escalating after %d attempts "
                    "without payment confirmation",
                    state.attempt,
                )
                if self._on_escalate is not None:
                    try:
                        await self._on_escalate()
                    except Exception as exc:
                        logger.error(
                            "payment_auto_check escalation callback failed: %s",
                            exc,
                        )
                return

            # ── advance exponential backoff ──────────────────────────────────
            state.next_delay = min(
                state.next_delay * config.backoff_multiplier,
                config.max_delay,
            )
