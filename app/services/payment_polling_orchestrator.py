# app/services/payment_polling_orchestrator.py
"""Payment polling lifecycle — no threading.Thread.

Replaces the daemon-thread approach from CheckoutService with a
concurrent.futures.ThreadPoolExecutor so the polling API exposes
explicit start / cancel / status handles instead of fire-and-forget
threads.

Design
------
* ``start(token)``  — submit a poll loop; returns the Future (or None if
  one is already running for that token).
* ``cancel(token)`` — signal the poll loop to exit at its next sleep
  boundary via a threading.Event.
* ``is_active(token)`` — True while a poll is in-flight.
* ``_poll(token, stop_event)`` — internal; called by the executor thread.
  Uses ``stop_event.wait(timeout)`` instead of ``time.sleep`` so
  cancellation is near-instant.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger(__name__)

PAYMENT_FAILURE_STATUSES = {"cancelled", "canceled", "expired", "failed", "declined"}


class PaymentPollingOrchestrator:
    """Owns the payment polling lifecycle for all active checkout tokens.

    Parameters
    ----------
    verify_fn:
        ``(token: str) -> dict`` — must return a dict with at least the
        keys ``payment_completed`` (bool) and ``status`` (str).  Typically
        ``CheckoutService.verify_payment_with_provider``.
    poll_interval:
        Seconds between Datacap status checks.
    poll_max_duration:
        Maximum total seconds to poll before giving up (timeout).
    failure_statuses:
        Set of status strings that indicate a terminal payment failure.
    executor:
        Optional ``ThreadPoolExecutor`` to use.  A default one is created
        if not supplied (useful for injecting a controlled executor in tests).
    """

    def __init__(
        self,
        verify_fn: Callable[[str], dict],
        *,
        poll_interval: float,
        poll_max_duration: float,
        failure_statuses: frozenset[str] = frozenset(PAYMENT_FAILURE_STATUSES),
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._verify_fn = verify_fn
        self._poll_interval = poll_interval
        self._poll_max_duration = poll_max_duration
        self._failure_statuses = failure_statuses
        self._executor = executor or ThreadPoolExecutor(
            max_workers=16,
            thread_name_prefix="payment-poller",
        )
        # token → Future for the running poll
        self._active: dict[str, Future] = {}
        # token → Event used to signal early termination
        self._stop_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, token: str) -> Future | None:
        """Start polling for *token*.

        Returns the Future that represents the poll loop, or ``None`` if a
        poll is already running for this token (deduplicated — safe to call
        multiple times).
        """
        with self._lock:
            existing = self._active.get(token)
            if existing is not None and not existing.done():
                logger.info(
                    "Poller already running for token=%s - skipping.", token
                )
                return None

            stop_event = threading.Event()
            self._stop_events[token] = stop_event
            future = self._executor.submit(self._poll, token, stop_event)
            self._active[token] = future

        logger.info(
            "Started Datacap payment poller for token=%s (interval=%.1fs, max=%.0fs)",
            token,
            self._poll_interval,
            self._poll_max_duration,
        )
        return future

    def cancel(self, token: str) -> bool:
        """Signal the poll loop for *token* to stop at its next wait boundary.

        Returns ``True`` if a running poll was found and signalled.
        """
        with self._lock:
            event = self._stop_events.get(token)
        if event is not None:
            event.set()
            return True
        return False

    def is_active(self, token: str) -> bool:
        """Return True while a poll future for *token* is still running."""
        with self._lock:
            future = self._active.get(token)
        return future is not None and not future.done()

    # ------------------------------------------------------------------
    # Internal poll loop
    # ------------------------------------------------------------------

    def _poll(self, token: str, stop_event: threading.Event) -> None:
        deadline = time.monotonic() + self._poll_max_duration

        try:
            while time.monotonic() < deadline:
                # Use Event.wait instead of time.sleep — cancelled immediately
                # when stop_event is set.
                remaining = deadline - time.monotonic()
                wait_secs = min(self._poll_interval, max(remaining, 0))
                if stop_event.wait(timeout=wait_secs):
                    logger.info(
                        "payment-poller: cancelled for token=%s", token
                    )
                    return

                try:
                    result = self._verify_fn(token)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "payment-poller: unexpected error for %s: %s", token, exc
                    )
                    continue

                if result.get("payment_completed"):
                    logger.info(
                        "payment-poller: payment confirmed for token=%s via Datacap.",
                        token,
                    )
                    return

                status_lower = str(result.get("status") or "").lower()
                if status_lower in self._failure_statuses:
                    logger.info(
                        "payment-poller: stopping for token=%s - status=%s",
                        token,
                        status_lower,
                    )
                    return

            logger.info(
                "payment-poller: timeout reached for token=%s after %ds",
                token,
                int(self._poll_max_duration),
            )
        finally:
            with self._lock:
                self._active.pop(token, None)
                self._stop_events.pop(token, None)
