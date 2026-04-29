# tests/services/test_payment_auto_check_scheduler.py
"""Tests for PaymentAutoCheckScheduler.

Coverage:
- Scheduling twice for the same session creates only one task (single-flight)
- Payment probe uses exponential backoff: initial → × multiplier, capped at max
- last_probe_at uses monotonic time
- Probe is suspended (phase guard) when phase != LISTENING
- Successful payment (probe returns False) cancels the scheduler
- Probe exception continues with backoff (does not crash task)
- Max attempts triggers escalation and sets state.escalated = True
- on_escalate callback is invoked exactly once at escalation
- Task cancellation is handled gracefully (CancelledError → return)
- cancel() stops a running task; is_scheduled() returns False afterward
- Guard retry does not advance the attempt counter
- Backoff does not exceed max_delay
- last_probe_at reflects monotonic time at probe call
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, patch

from app.realtime.realtime_conversation_state import RealtimePhase
from app.services.payment_auto_check_scheduler import (
    PaymentAutoCheckConfig,
    PaymentAutoCheckScheduler,
    PaymentAutoCheckState,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _config(
    *,
    initial_delay: float = 0.0,
    backoff_multiplier: float = 2.0,
    max_delay: float = 300.0,
    max_attempts: int = 8,
    guard_retry_delay: float = 0.0,
) -> PaymentAutoCheckConfig:
    return PaymentAutoCheckConfig(
        initial_delay=initial_delay,
        backoff_multiplier=backoff_multiplier,
        max_delay=max_delay,
        max_attempts=max_attempts,
        guard_retry_delay=guard_retry_delay,
    )


def _make_scheduler(
    *,
    get_phase=None,
    dispatch_probe=None,
    on_escalate=None,
    config: PaymentAutoCheckConfig | None = None,
) -> PaymentAutoCheckScheduler:
    async def _default_probe() -> bool:
        return False

    return PaymentAutoCheckScheduler(
        get_phase=get_phase or (lambda: RealtimePhase.LISTENING),
        dispatch_probe=dispatch_probe or _default_probe,
        on_escalate=on_escalate,
        config=config or _config(),
    )


def _run(coro):
    return asyncio.run(coro)


# ── TTSFailureError contract ─ (re-used pattern: exceptions exist) ────────────

# ── single-flight guarantee ───────────────────────────────────────────────────

class TestSingleFlight:
    def test_schedule_creates_task(self):
        scheduler = _make_scheduler()

        async def _go():
            scheduler.schedule()
            assert scheduler.is_scheduled()
            await scheduler.state.task

        _run(_go())

    def test_schedule_twice_same_task(self):
        """Second schedule() call must not replace the running task."""
        probe_calls: list[int] = []

        async def _slow_probe() -> bool:
            probe_calls.append(1)
            await asyncio.sleep(0)  # yield
            return False

        scheduler = _make_scheduler(dispatch_probe=_slow_probe)

        async def _go():
            scheduler.schedule()
            first_task = scheduler.state.task

            scheduler.schedule()  # second call while first is running
            second_task = scheduler.state.task

            assert first_task is second_task, "Second schedule() must not create a new task"
            await first_task

        _run(_go())

    def test_schedule_twice_only_one_probe(self):
        probe_calls = 0

        async def _probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            return False

        scheduler = _make_scheduler(dispatch_probe=_probe)

        async def _go():
            scheduler.schedule()
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        assert probe_calls == 1

    def test_is_scheduled_false_before_schedule(self):
        scheduler = _make_scheduler()
        assert not scheduler.is_scheduled()

    def test_is_scheduled_false_after_task_completes(self):
        scheduler = _make_scheduler()

        async def _go():
            scheduler.schedule()
            await scheduler.state.task
            assert not scheduler.is_scheduled()

        _run(_go())


# ── cancel ────────────────────────────────────────────────────────────────────

class TestCancel:
    def test_cancel_stops_running_task(self):
        probe_calls = 0

        async def _blocking_probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            await asyncio.sleep(10)  # would run forever in tests
            return True

        scheduler = _make_scheduler(
            dispatch_probe=_blocking_probe,
            config=_config(initial_delay=0),
        )

        async def _go():
            scheduler.schedule()
            await asyncio.sleep(0)  # let the task start
            assert scheduler.is_scheduled()
            scheduler.cancel()
            assert not scheduler.is_scheduled()

        _run(_go())

    def test_cancel_before_schedule_is_safe(self):
        scheduler = _make_scheduler()
        scheduler.cancel()  # must not raise
        assert not scheduler.is_scheduled()

    def test_cancel_clears_state(self):
        scheduler = _make_scheduler()

        async def _go():
            scheduler.schedule()
            scheduler.cancel()
            assert scheduler.state is None

        _run(_go())

    def test_cancel_mid_sleep_returns_gracefully(self):
        scheduler = _make_scheduler(config=_config(initial_delay=60.0))

        async def _go():
            scheduler.schedule()
            await asyncio.sleep(0)  # let task enter asyncio.sleep(60)
            task = scheduler.state.task
            scheduler.cancel()
            # Task should complete without raising
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.CancelledError:
                pass  # acceptable; task was cancelled

        _run(_go())


# ── probe dispatch ────────────────────────────────────────────────────────────

class TestProbeDispatch:
    def test_probe_called_once_per_backoff_cycle(self):
        calls = 0

        async def _probe() -> bool:
            nonlocal calls
            calls += 1
            return calls < 3  # 3 probes then stop

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            config=_config(initial_delay=0, max_attempts=10),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        assert calls == 3

    def test_probe_false_stops_loop(self):
        calls = 0

        async def _probe() -> bool:
            nonlocal calls
            calls += 1
            return False  # stop immediately

        scheduler = _make_scheduler(dispatch_probe=_probe)

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        assert calls == 1

    def test_probe_exception_continues_loop(self):
        """An exception from the probe must not crash the task."""
        calls = 0

        async def _fragile_probe() -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient probe error")
            return False  # stop on second call

        scheduler = _make_scheduler(
            dispatch_probe=_fragile_probe,
            config=_config(initial_delay=0, max_attempts=10),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task  # must complete, not raise

        _run(_go())
        assert calls == 2


# ── phase guard ───────────────────────────────────────────────────────────────

class TestPhaseGuard:
    def test_probe_not_called_when_processing(self):
        """Probe must not fire when phase == PROCESSING."""
        phases = iter([RealtimePhase.PROCESSING, RealtimePhase.LISTENING])
        probe_calls = 0

        async def _probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            return False

        scheduler = _make_scheduler(
            get_phase=lambda: next(phases, RealtimePhase.LISTENING),
            dispatch_probe=_probe,
            config=_config(initial_delay=0, guard_retry_delay=0),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        # First phase check → PROCESSING → guard fires; second → LISTENING → probe
        assert probe_calls == 1

    def test_probe_not_called_when_speaking(self):
        """Probe must not fire when phase == SPEAKING."""
        phases = iter([RealtimePhase.SPEAKING, RealtimePhase.LISTENING])
        probe_calls = 0

        async def _probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            return False

        scheduler = _make_scheduler(
            get_phase=lambda: next(phases, RealtimePhase.LISTENING),
            dispatch_probe=_probe,
            config=_config(initial_delay=0, guard_retry_delay=0),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        assert probe_calls == 1

    def test_guard_does_not_advance_attempt(self):
        """A guarded (skipped) probe slot must not consume an attempt."""
        # Phase: PROCESSING on slot 1, LISTENING on slot 2
        call_count = 0
        phases = iter([RealtimePhase.PROCESSING, RealtimePhase.LISTENING])

        async def _probe() -> bool:
            nonlocal call_count
            call_count += 1
            return False

        scheduler = _make_scheduler(
            get_phase=lambda: next(phases, RealtimePhase.LISTENING),
            dispatch_probe=_probe,
            config=_config(initial_delay=0, guard_retry_delay=0),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        # Probe fired once; attempt counter must be 1, not 2.
        assert scheduler.state is not None
        assert scheduler.state.attempt == 1


# ── backoff ───────────────────────────────────────────────────────────────────

class TestExponentialBackoff:
    def test_backoff_sequence_with_patched_sleep(self):
        """Delays passed to asyncio.sleep follow the configured sequence."""
        delays_slept: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            delays_slept.append(seconds)

        probe_count = 0

        async def _probe() -> bool:
            nonlocal probe_count
            probe_count += 1
            return probe_count < 5  # 5 probes then stop

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            config=_config(
                initial_delay=30.0,
                backoff_multiplier=2.0,
                max_delay=300.0,
                max_attempts=10,
                guard_retry_delay=0.0,
            ),
        )

        async def _go():
            with patch("asyncio.sleep", side_effect=_fake_sleep):
                scheduler.schedule()
                await scheduler.state.task

        _run(_go())

        # delays_slept should be: [30, 60, 120, 240, 300] (5 cycles)
        assert delays_slept[0] == 30.0
        assert delays_slept[1] == 60.0
        assert delays_slept[2] == 120.0
        assert delays_slept[3] == 240.0
        assert delays_slept[4] == 300.0  # min(240*2=480, 300)

    def test_backoff_capped_at_max_delay(self):
        delays_slept: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            delays_slept.append(seconds)

        probe_count = 0

        async def _probe() -> bool:
            nonlocal probe_count
            probe_count += 1
            return probe_count < 8

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            config=_config(
                initial_delay=30.0,
                backoff_multiplier=2.0,
                max_delay=120.0,  # cap lower than natural sequence
                max_attempts=10,
                guard_retry_delay=0.0,
            ),
        )

        async def _go():
            with patch("asyncio.sleep", side_effect=_fake_sleep):
                scheduler.schedule()
                await scheduler.state.task

        _run(_go())

        # After 30, 60, 120, all remaining delays must be 120
        for delay in delays_slept[2:]:
            assert delay <= 120.0, f"delay {delay} exceeded max_delay=120"

    def test_state_next_delay_advances_after_probe(self):
        """state.next_delay reflects the delay for the NEXT probe."""
        delays_after_probe: list[float] = []
        probe_count = 0

        scheduler = _make_scheduler(
            config=_config(
                initial_delay=30.0,
                backoff_multiplier=2.0,
                max_delay=300.0,
                max_attempts=10,
            ),
        )

        async def _probe() -> bool:
            nonlocal probe_count
            probe_count += 1
            # Read next_delay after probe fires but before scheduler advances it.
            # The advancement happens after dispatch_probe returns.
            return probe_count < 3

        async def _fake_sleep(seconds: float) -> None:
            # skip sleeping
            pass

        scheduler2 = _make_scheduler(
            dispatch_probe=_probe,
            config=_config(
                initial_delay=30.0,
                backoff_multiplier=2.0,
                max_delay=300.0,
                max_attempts=10,
                guard_retry_delay=0.0,
            ),
        )

        async def _go():
            with patch("asyncio.sleep", side_effect=_fake_sleep):
                scheduler2.schedule()
                await scheduler2.state.task

        _run(_go())

        # After 3 probes (probe_count reaches 3, loop stops):
        # initial=30 → after probe1: 60 → after probe2: 120 → probe3 fires → stop
        # state.next_delay is 120 (advanced after probe2, before probe3 would advance)
        assert scheduler2.state.next_delay == 120.0


# ── last_probe_at ─────────────────────────────────────────────────────────────

class TestLastProbeAt:
    def test_last_probe_at_none_before_any_probe(self):
        scheduler = _make_scheduler()

        async def _go():
            scheduler.schedule()
            # Cancel immediately before probe fires
            scheduler.cancel()

        _run(_go())
        # State is cleared by cancel(); None is fine either way
        assert scheduler.state is None or scheduler.state.last_probe_at is None

    def test_last_probe_at_set_after_probe(self):
        before = time.monotonic()

        async def _probe() -> bool:
            return False

        scheduler = _make_scheduler(dispatch_probe=_probe)

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())

        after = time.monotonic()
        assert scheduler.state is not None
        assert scheduler.state.last_probe_at is not None
        assert before <= scheduler.state.last_probe_at <= after

    def test_last_probe_at_uses_monotonic_not_wall_clock(self):
        """last_probe_at must come from time.monotonic(), not time.time()."""
        recorded_at: list[float] = []

        original_monotonic = time.monotonic

        def _fake_monotonic() -> float:
            val = original_monotonic()
            recorded_at.append(val)
            return val

        async def _probe() -> bool:
            return False

        scheduler = _make_scheduler(dispatch_probe=_probe)

        async def _go():
            with patch("app.services.payment_auto_check_scheduler.time.monotonic", _fake_monotonic):
                scheduler.schedule()
                await scheduler.state.task

        _run(_go())

        assert scheduler.state is not None
        assert scheduler.state.last_probe_at is not None
        # last_probe_at must equal one of the values our fake recorded
        assert scheduler.state.last_probe_at in recorded_at


# ── escalation ────────────────────────────────────────────────────────────────

class TestEscalation:
    def test_escalation_called_at_max_attempts(self):
        escalated = False

        async def _on_escalate() -> None:
            nonlocal escalated
            escalated = True

        probe_count = 0

        async def _probe() -> bool:
            nonlocal probe_count
            probe_count += 1
            return True  # always continue so max_attempts is reached

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            on_escalate=_on_escalate,
            config=_config(initial_delay=0, max_attempts=3),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        assert escalated
        assert probe_count == 3  # ran exactly max_attempts probes

    def test_escalation_sets_state_flag(self):
        async def _probe() -> bool:
            return True

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            config=_config(initial_delay=0, max_attempts=2),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        assert scheduler.state is not None
        assert scheduler.state.escalated is True

    def test_escalation_callback_exception_does_not_crash(self):
        async def _bad_escalate() -> None:
            raise RuntimeError("escalation blew up")

        async def _probe() -> bool:
            return True

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            on_escalate=_bad_escalate,
            config=_config(initial_delay=0, max_attempts=1),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task  # must complete, not raise

        _run(_go())  # no exception propagation

    def test_no_escalation_when_probe_stops_early(self):
        escalated = False

        async def _on_escalate() -> None:
            nonlocal escalated
            escalated = True

        async def _probe() -> bool:
            return False  # payment resolved before max_attempts

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            on_escalate=_on_escalate,
            config=_config(initial_delay=0, max_attempts=5),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        assert not escalated

    def test_escalation_attempt_count_equals_max_attempts(self):
        probe_count = 0

        async def _probe() -> bool:
            nonlocal probe_count
            probe_count += 1
            return True

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            config=_config(initial_delay=0, max_attempts=4),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        assert probe_count == 4
        assert scheduler.state is not None
        assert scheduler.state.attempt == 4


# ── no_escalate callback when not provided ────────────────────────────────────

class TestNoEscalateCallback:
    def test_escalation_without_callback_does_not_raise(self):
        """Escalation with on_escalate=None must not raise."""
        async def _probe() -> bool:
            return True

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            on_escalate=None,
            config=_config(initial_delay=0, max_attempts=1),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())  # no exception


# ── rapid schedule / cancel cycles ───────────────────────────────────────────

class TestRapidCycles:
    def test_reschedule_after_completion_starts_new_task(self):
        """After a task completes naturally, schedule() must create a new task."""
        probe_calls = 0

        async def _probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            return False

        scheduler = _make_scheduler(dispatch_probe=_probe)

        async def _go():
            # First run
            scheduler.schedule()
            first_task = scheduler.state.task
            await first_task

            assert not scheduler.is_scheduled()

            # Second run
            scheduler.schedule()
            second_task = scheduler.state.task
            await second_task

            assert first_task is not second_task

        _run(_go())
        assert probe_calls == 2

    def test_cancel_then_reschedule(self):
        probe_calls = 0

        async def _probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            return False

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            config=_config(initial_delay=60.0),
        )

        async def _go():
            scheduler.schedule()
            await asyncio.sleep(0)  # let task start
            scheduler.cancel()

            # Now reschedule with zero delay
            scheduler2 = _make_scheduler(dispatch_probe=_probe)
            scheduler2.schedule()
            await scheduler2.state.task

        _run(_go())
        assert probe_calls == 1  # only the second schedule fired a probe


# ── PaymentAutoCheckState contract ───────────────────────────────────────────

class TestStateContract:
    def test_initial_state_values(self):
        state = PaymentAutoCheckState()
        assert state.attempt == 0
        assert state.last_probe_at is None
        assert state.escalated is False
        assert state.task is None

    def test_state_attempt_increments_each_probe(self):
        probe_count = 0

        async def _probe() -> bool:
            nonlocal probe_count
            probe_count += 1
            return probe_count < 3

        scheduler = _make_scheduler(
            dispatch_probe=_probe,
            config=_config(initial_delay=0, max_attempts=10),
        )

        async def _go():
            scheduler.schedule()
            await scheduler.state.task

        _run(_go())
        assert scheduler.state is not None
        assert scheduler.state.attempt == 3


# ── config defaults ───────────────────────────────────────────────────────────

class TestConfigDefaults:
    def test_default_config_values(self):
        cfg = PaymentAutoCheckConfig()
        assert cfg.initial_delay == 30.0
        assert cfg.backoff_multiplier == 2.0
        assert cfg.max_delay == 300.0
        assert cfg.max_attempts == 8
        assert cfg.guard_retry_delay == 10.0

    def test_custom_config_overrides(self):
        cfg = PaymentAutoCheckConfig(
            initial_delay=5.0,
            backoff_multiplier=1.5,
            max_delay=60.0,
            max_attempts=3,
            guard_retry_delay=2.0,
        )
        assert cfg.initial_delay == 5.0
        assert cfg.backoff_multiplier == 1.5
        assert cfg.max_delay == 60.0
        assert cfg.max_attempts == 3
        assert cfg.guard_retry_delay == 2.0
