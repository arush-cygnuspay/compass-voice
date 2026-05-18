# app/nlu/semantic_repair/daily_budget.py
"""In-memory daily GPT call budget with UTC date-based reset.

The counter resets automatically at UTC midnight.  No persistent DB is used —
the budget counter is per-process and resets on process restart or day rollover.

Thread-safe: all mutations are serialised under a single lock.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone


def _utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class GptDailyBudget:
    """Rate-limiter: allow at most *limit* GPT calls per UTC day.

    Parameters
    ----------
    limit:
        Maximum number of calls per UTC day.  0 = unlimited.
    """

    def __init__(self, limit: int = 10000) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._count: int = 0
        self._date: str = _utc_date_str()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_consume(self) -> bool:
        """Increment the daily counter if under budget.

        Returns True (call allowed) or False (budget exceeded).
        The counter is incremented atomically only when True is returned.
        """
        if self._limit == 0:
            return True  # unlimited
        with self._lock:
            self._maybe_reset()
            if self._count >= self._limit:
                return False
            self._count += 1
            return True

    def is_exceeded(self) -> bool:
        """Return True if the daily budget is currently exceeded."""
        if self._limit == 0:
            return False
        with self._lock:
            self._maybe_reset()
            return self._count >= self._limit

    @property
    def count(self) -> int:
        """Current call count for today (read-only snapshot)."""
        with self._lock:
            self._maybe_reset()
            return self._count

    @property
    def limit(self) -> int:
        return self._limit

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_reset(self) -> None:
        """Reset counter if UTC date has rolled over. Caller must hold _lock."""
        today = _utc_date_str()
        if today != self._date:
            self._count = 0
            self._date = today
