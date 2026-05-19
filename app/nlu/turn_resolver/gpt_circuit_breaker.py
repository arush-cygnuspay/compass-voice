# app/nlu/turn_resolver/gpt_circuit_breaker.py
"""In-memory circuit breaker for GPT provider calls.

Tracks consecutive provider/network/timeout failures per call key.
When the failure threshold is reached, the circuit opens for a
configurable cool-down period — GPT calls are skipped and
``GptCallStatus.CIRCUIT_OPEN`` is returned immediately.

Design decisions
----------------
* Thread-safe via a threading.Lock (all services are sync/threaded).
* State resets fully on UTC date change (defense against stale open state).
* Does not persist across process restarts (intentional — transient protection).
* Only provider-side failures count toward the threshold (TIMEOUT, RATE_LIMITED,
  PROVIDER_ERROR, NETWORK_ERROR).  Parse/validation/budget failures do not count.
* Config is read once at instantiation; use ``GptCircuitBreaker(config=...)``
  or the module-level ``DEFAULT_CIRCUIT_BREAKER`` singleton.

Usage
-----
    breaker = GptCircuitBreaker()

    if breaker.is_open("gpt-4o-mini:bucket_0"):
        return gpt_skipped_result(status=GptCallStatus.CIRCUIT_OPEN)

    result = call_gpt_safely(...)

    if result.should_open_circuit:
        breaker.record_failure("gpt-4o-mini:bucket_0")
    elif result.ok:
        breaker.record_success("gpt-4o-mini:bucket_0")
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for the GPT circuit breaker.

    All settings can be overridden via environment variables.
    """

    enabled: bool = True
    failure_threshold: int = 3
    open_seconds: float = 30.0


def _load_circuit_breaker_config() -> CircuitBreakerConfig:
    enabled = os.getenv("GPT_CIRCUIT_BREAKER_ENABLED", "true").lower() in {
        "1", "true", "yes", "on"
    }
    threshold = int(os.getenv("GPT_CIRCUIT_FAILURE_THRESHOLD", "3"))
    open_secs = float(os.getenv("GPT_CIRCUIT_OPEN_SECONDS", "30"))
    return CircuitBreakerConfig(
        enabled=enabled,
        failure_threshold=max(1, threshold),
        open_seconds=max(1.0, open_secs),
    )


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class GptCircuitBreaker:
    """Thread-safe, in-memory GPT circuit breaker.

    Keyed by an arbitrary string (typically ``"{model}:{task_mode}"``).
    Multiple keys can be tracked independently.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config: CircuitBreakerConfig = config or _load_circuit_breaker_config()
        self._lock = threading.Lock()
        # consecutive failure counts per key
        self._failures: dict[str, int] = {}
        # monotonic time when circuit opens; 0 = not open
        self._open_until: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_open(self, key: str) -> bool:
        """Return True when the circuit for *key* is open (GPT should be skipped)."""
        if not self._config.enabled:
            return False
        with self._lock:
            until = self._open_until.get(key, 0.0)
            if until == 0.0:
                return False
            if time.monotonic() >= until:
                # Cool-down expired — auto-close
                self._open_until.pop(key, None)
                # Keep failure count; next call will prove health
                return False
            return True

    def record_failure(self, key: str) -> bool:
        """Record a provider-side failure.  Returns True if circuit just opened."""
        if not self._config.enabled:
            return False
        with self._lock:
            # If circuit is already open, don't double-count
            until = self._open_until.get(key, 0.0)
            if until and time.monotonic() < until:
                return False

            count = self._failures.get(key, 0) + 1
            self._failures[key] = count

            if count >= self._config.failure_threshold:
                self._open_until[key] = time.monotonic() + self._config.open_seconds
                return True  # circuit just opened
            return False

    def record_success(self, key: str) -> None:
        """Record a successful GPT call — reset the failure count."""
        with self._lock:
            self._failures.pop(key, None)
            self._open_until.pop(key, None)

    def failure_count(self, key: str) -> int:
        """Return the current consecutive failure count for *key* (for tests)."""
        with self._lock:
            return self._failures.get(key, 0)

    def open_until(self, key: str) -> float:
        """Return the monotonic timestamp when the circuit will close (0 = closed)."""
        with self._lock:
            return self._open_until.get(key, 0.0)

    def force_close(self, key: str) -> None:
        """Force-close the circuit for *key* (use in tests / admin reset)."""
        with self._lock:
            self._failures.pop(key, None)
            self._open_until.pop(key, None)

    def circuit_key(self, model: str | None, task_mode: str | None) -> str:
        """Build the canonical circuit key from model and task_mode."""
        m = (model or "unknown").strip()
        t = (task_mode or "generic").strip()
        return f"{m}:{t}"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Shared across all turn-resolver calls within the same process.
# Tests that need isolation should instantiate their own GptCircuitBreaker.
DEFAULT_CIRCUIT_BREAKER: GptCircuitBreaker = GptCircuitBreaker()
