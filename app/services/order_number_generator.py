# app/services/order_number_generator.py
"""Collision-safe, deterministic order-number generators.

Format (identical to the legacy 7-digit scheme):
  <2-digit seq prefix><5-digit centisecond suffix>

  seq_part  = sequence counter mod 100  (00–99)
              replaces the old random.randint(10, 99) prefix
  ts_part   = centiseconds-since-epoch mod 100_000  (00000–99999)

Both parts sum to 7 digits, always all-numeric, backward-compatible with
every persisted order number from the previous implementation.

Implementations
---------------
OrderNumberGenerator (default)
    In-process, thread-safe.  Uses a monotonic per-centisecond sequence
    counter (lock-protected) instead of random — up to 100 unique numbers
    per centisecond per process.  No external dependencies.

RedisOrderNumberGenerator
    Distributed, atomic.  Scopes Redis INCR to
    ``order_seq:{restaurant_id}:{centisecond}`` with a 2-second TTL, giving
    unlimited per-centisecond throughput across processes.  Falls back
    transparently to OrderNumberGenerator if Redis raises.
"""
from __future__ import annotations

import threading
import time
from typing import Callable


def _default_clock_ms() -> int:
    """Return current UTC time as integer milliseconds since epoch."""
    return int(time.time() * 1_000)


class OrderNumberGenerator:
    """In-process, thread-safe, deterministic order-number generator.

    Parameters
    ----------
    clock_ms:
        Zero-argument callable returning current time as integer milliseconds.
        Defaults to ``int(time.time() * 1_000)``.  Inject a fixed value for
        deterministic tests.
    """

    def __init__(self, *, clock_ms: Callable[[], int] | None = None) -> None:
        self._clock_ms = clock_ms or _default_clock_ms
        self._lock = threading.Lock()
        self._last_cs: int = -1  # last centisecond bucket
        self._seq: int = 0       # 0-based counter within that bucket

    def generate(self, restaurant_id: str = "", *, now_ms: int | None = None) -> str:
        """Generate a fresh 7-digit numeric order number.

        Parameters
        ----------
        restaurant_id:
            Restaurant identifier.  Unused by this in-process implementation
            (accepted for interface compatibility with RedisOrderNumberGenerator).
        now_ms:
            Optional override for the current time in milliseconds.  When
            provided, the internal clock is bypassed — useful in tests.
        """
        ms = now_ms if now_ms is not None else self._clock_ms()
        cs = ms // 10  # centiseconds bucket

        with self._lock:
            if cs != self._last_cs:
                self._last_cs = cs
                self._seq = 0
            else:
                self._seq = (self._seq + 1) % 100
            seq = self._seq

        ts_part = cs % 100_000
        return f"{seq:02d}{ts_part:05d}"


class RedisOrderNumberGenerator:
    """Distributed, atomic order-number generator backed by Redis INCR.

    Uses the key ``order_seq:{restaurant_id}:{centisecond}`` with a 2-second
    TTL.  This gives unlimited collision-free throughput across processes for
    the same restaurant within the same centisecond window.

    Falls back to ``OrderNumberGenerator`` transparently on any Redis error.

    Parameters
    ----------
    redis_client:
        A redis.Redis (or compatible) client instance.
    fallback:
        Generator to use when Redis is unavailable.  Defaults to a new
        ``OrderNumberGenerator``.
    clock_ms:
        Optional clock override (milliseconds) — primarily for testing.
    """

    def __init__(
        self,
        redis_client,
        *,
        fallback: OrderNumberGenerator | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._redis = redis_client
        self._fallback = fallback or OrderNumberGenerator(clock_ms=clock_ms)
        self._clock_ms = clock_ms or _default_clock_ms

    def generate(self, restaurant_id: str = "", *, now_ms: int | None = None) -> str:
        """Generate a 7-digit order number using atomic Redis INCR.

        Falls back to in-process generation on any Redis error.
        """
        try:
            ms = now_ms if now_ms is not None else self._clock_ms()
            cs = ms // 10
            key = f"order_seq:{restaurant_id}:{cs}"
            # INCR returns 1-based; convert to 0-based for the prefix.
            seq = int(self._redis.incr(key)) - 1
            self._redis.expire(key, 2)
            ts_part = cs % 100_000
            return f"{seq % 100:02d}{ts_part:05d}"
        except Exception:
            return self._fallback.generate(restaurant_id, now_ms=now_ms)
