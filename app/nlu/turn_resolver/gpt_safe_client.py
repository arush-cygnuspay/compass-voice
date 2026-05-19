# app/nlu/turn_resolver/gpt_safe_client.py
"""Safe GPT call wrapper — structured result, never raises into callers.

Usage (legacy synchronous)
--------------------------
Instead of calling the OpenAI client directly, wrap the call::

    result = call_gpt_safely(
        gpt_callable=lambda: client.chat.completions.create(...),
        parse_fn=my_json_parser,
        task_mode="idle_menu_item_resolution",
        timeout_ms=700,
        model=cfg.model,
    )
    if not result.ok:
        # fall back to local deterministic path
        ...

Usage (async GptSafeClient)
---------------------------
    client = GptSafeClient(underlying_client=my_async_fn, circuit_breaker=breaker)
    result = await client.call(
        task_mode="idle_menu_item_resolution",
        messages=[...],
        model="gpt-4o-mini",
        timeout_ms=700,
        parse_fn=json.loads,
    )

Safety contract
---------------
* Never raises into the caller regardless of the exception type.
* Sanitises error messages — API keys and secrets are redacted.
* ``should_fallback_to_local=True`` on all failure statuses.
* ``should_open_circuit=True`` only for provider/network/timeout/unknown failures
  (not for validation or low-confidence failures).
"""
from __future__ import annotations

import asyncio
import inspect
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------


class GptCallStatus:
    """String constants for GPT call outcome classification."""

    OK = "ok"
    DISABLED = "disabled"
    API_KEY_MISSING = "api_key_missing"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    PROVIDER_ERROR = "provider_error"
    NETWORK_ERROR = "network_error"
    INVALID_JSON = "invalid_json"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    LOW_CONFIDENCE = "low_confidence"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN_ERROR = "unknown_error"


# Statuses that indicate the provider-side is unhealthy
_CIRCUIT_TRIGGERING_STATUSES: frozenset[str] = frozenset({
    GptCallStatus.TIMEOUT,
    GptCallStatus.RATE_LIMITED,
    GptCallStatus.PROVIDER_ERROR,
    GptCallStatus.NETWORK_ERROR,
    GptCallStatus.UNKNOWN_ERROR,
})

# Maximum raw_text length preserved in result (to limit memory use)
_MAX_RAW_TEXT_BYTES = 2000

# Maximum error_message length (prevents log flooding)
_MAX_ERROR_MSG_BYTES = 300


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GptSafeResult:
    """Structured outcome of a safe-wrapped GPT call.

    Fields
    ------
    ok:
        True when the GPT call succeeded and parse_fn returned a valid result.
    status:
        One of the ``GptCallStatus`` constants.
    task_mode:
        The bucket / task mode string passed at call time (for logging).
    parsed:
        The object returned by ``parse_fn`` when ``ok=True``.  None otherwise.
    raw_text:
        The raw GPT response text (truncated to _MAX_RAW_TEXT_BYTES).
        Useful for debugging INVALID_JSON / SCHEMA_VALIDATION_FAILED.
    error_message:
        Short human-readable error description (API key redacted).
    latency_ms:
        Wall-clock time of the GPT API call (ms).  None when not attempted.
    model:
        OpenAI model used.
    timeout_ms:
        The timeout that was enforced.
    should_fallback_to_local:
        Always True on failure — callers must use local deterministic path.
    should_open_circuit:
        True when the failure indicates the provider is unhealthy (TIMEOUT,
        PROVIDER_ERROR, NETWORK_ERROR, RATE_LIMITED).  False for parse /
        validation / budget failures.
    """

    ok: bool
    status: str
    task_mode: str | None = None
    parsed: Any = field(default=None, compare=False)
    raw_text: str | None = None
    error_message: str | None = None
    latency_ms: float | None = None
    model: str | None = None
    timeout_ms: int | None = None
    provider: str | None = None
    should_fallback_to_local: bool = True
    should_open_circuit: bool = False
    metadata: dict = field(default_factory=dict, compare=False)


# Sentinel for "call not attempted"
GPT_SAFE_RESULT_NOT_CALLED = GptSafeResult(
    ok=False,
    status=GptCallStatus.DISABLED,
    should_fallback_to_local=True,
    should_open_circuit=False,
)


# ---------------------------------------------------------------------------
# Exception classification
# ---------------------------------------------------------------------------


def classify_exception(exc: Exception) -> str:
    """Map an exception to a GptCallStatus constant (public API)."""
    exc_name = type(exc).__name__
    exc_str = str(exc).lower()

    if "RateLimitError" in exc_name or "429" in exc_str or "rate limit" in exc_str:
        return GptCallStatus.RATE_LIMITED
    if (
        isinstance(exc, (asyncio.TimeoutError, TimeoutError))
        or "Timeout" in exc_name
        or "timeout" in exc_name.lower()
        or "timed out" in exc_str
    ):
        return GptCallStatus.TIMEOUT
    if "APIStatusError" in exc_name or "InternalServerError" in exc_name or "500" in exc_str:
        return GptCallStatus.PROVIDER_ERROR
    if (
        "Connection" in exc_name
        or "Network" in exc_name
        or "connection" in exc_str
        or "network" in exc_str
    ):
        return GptCallStatus.NETWORK_ERROR
    if "JSON" in exc_name or "json" in exc_str or isinstance(exc, (ValueError, KeyError)):
        return GptCallStatus.INVALID_JSON

    return GptCallStatus.UNKNOWN_ERROR


def sanitize_error_message(message: str, *, max_chars: int = _MAX_ERROR_MSG_BYTES) -> str:
    """Return a sanitised, truncated error string with API keys redacted."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and api_key in message:
        message = message.replace(api_key, "[REDACTED]")
    return message[:max_chars]


def truncate_raw_text(raw_text: str, max_chars: int = _MAX_RAW_TEXT_BYTES) -> str:
    """Truncate raw GPT output to at most *max_chars* characters."""
    if not raw_text:
        return raw_text
    return raw_text[:max_chars]


def _classify_exception(exc: Exception) -> str:
    """Map an exception to a GptCallStatus constant."""
    exc_name = type(exc).__name__
    exc_str = str(exc).lower()

    if "RateLimitError" in exc_name or "429" in exc_str or "rate limit" in exc_str:
        return GptCallStatus.RATE_LIMITED
    if "Timeout" in exc_name or "timeout" in exc_name.lower() or "timed out" in exc_str:
        return GptCallStatus.TIMEOUT
    if "APIStatusError" in exc_name or "InternalServerError" in exc_name or "500" in exc_str:
        return GptCallStatus.PROVIDER_ERROR
    if (
        "Connection" in exc_name
        or "Network" in exc_name
        or "connection" in exc_str
        or "network" in exc_str
    ):
        return GptCallStatus.NETWORK_ERROR
    if "JSON" in exc_name or "json" in exc_str or isinstance(exc, (ValueError, KeyError)):
        return GptCallStatus.INVALID_JSON

    return GptCallStatus.UNKNOWN_ERROR


def _sanitize_error(exc: Exception) -> str:
    """Return a sanitised, truncated error string with API keys redacted."""
    raw = f"{type(exc).__name__}: {exc}"
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and api_key in raw:
        raw = raw.replace(api_key, "[REDACTED]")
    return raw[:_MAX_ERROR_MSG_BYTES]


# ---------------------------------------------------------------------------
# Async safe client (Priority 2 — preferred for new GPT call sites)
# ---------------------------------------------------------------------------


class GptSafeClient:
    """Async, never-raise GPT call wrapper with circuit-breaker integration.

    Parameters
    ----------
    underlying_client:
        Callable ``(messages, model, timeout_s) -> str`` (sync or async).
        Returns the raw GPT response text.  May be None — returns
        API_KEY_MISSING when None (client not configured).
    circuit_breaker:
        ``GptCircuitBreaker`` instance.  Defaults to DEFAULT_CIRCUIT_BREAKER.
    config:
        ``SemanticRepairConfig`` (or any object with ``gpt_max_timeout_ms``).
        Used to cap timeout.  May be None.
    parse_fn:
        Default parse function — overridden per-call via ``call(parse_fn=...)``.
    logger:
        Optional logger (unused internally; available for subclasses / hooks).
    provider:
        Provider label for metadata (default ``"openai"``).
    """

    def __init__(
        self,
        underlying_client: Callable | None = None,
        circuit_breaker: Any = None,
        config: Any = None,
        parse_fn: Callable[[str], Any] | None = None,
        logger: Any = None,
        provider: str | None = None,
    ) -> None:
        self._client = underlying_client
        self._breaker = circuit_breaker
        self._config = config
        self._default_parse_fn = parse_fn
        self._logger = logger
        self._provider = provider or "openai"

    # ── Public API ────────────────────────────────────────────────────────────

    async def call(
        self,
        *,
        task_mode: str,
        messages: list[dict],
        model: str,
        timeout_ms: int,
        parse_fn: Callable[[str], Any],
        enabled: bool = True,
        budget_allowed: bool = True,
        metadata: dict | None = None,
    ) -> "GptSafeResult":
        """Make a GPT call with full exception isolation.  Never raises.

        Returns ``GptSafeResult`` for every outcome including failures.
        """
        try:
            return await self._call_impl(
                task_mode=task_mode,
                messages=messages,
                model=model,
                timeout_ms=timeout_ms,
                parse_fn=parse_fn,
                enabled=enabled,
                budget_allowed=budget_allowed,
                metadata=metadata,
            )
        except Exception as exc:
            return GptSafeResult(
                ok=False,
                status=GptCallStatus.UNKNOWN_ERROR,
                task_mode=task_mode,
                error_message=sanitize_error_message(f"{type(exc).__name__}: {exc}"),
                model=model,
                timeout_ms=timeout_ms,
                provider=self._provider,
                should_fallback_to_local=True,
                should_open_circuit=True,
                metadata=dict(metadata or {}),
            )

    # ── Implementation ────────────────────────────────────────────────────────

    async def _call_impl(
        self,
        *,
        task_mode: str,
        messages: list[dict],
        model: str,
        timeout_ms: int,
        parse_fn: Callable[[str], Any],
        enabled: bool,
        budget_allowed: bool,
        metadata: dict | None,
    ) -> "GptSafeResult":
        t0 = time.perf_counter()
        _meta = dict(metadata or {})

        # Cap timeout at config max
        max_ms = 1200
        if self._config is not None:
            max_ms = int(getattr(self._config, "gpt_max_timeout_ms", 1200))
        effective_ms = min(int(timeout_ms), max_ms)
        timeout_s = effective_ms / 1000.0

        def _result(ok: bool, status: str, **kw: Any) -> "GptSafeResult":
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            return GptSafeResult(
                ok=ok,
                status=status,
                task_mode=task_mode,
                model=model,
                timeout_ms=effective_ms,
                provider=self._provider,
                should_fallback_to_local=not ok,
                should_open_circuit=False,
                metadata=_meta,
                latency_ms=kw.pop("latency_ms", elapsed),
                **kw,
            )

        # Pre-call guards (in order: enabled → budget → api_key → circuit)
        if not enabled:
            return _result(False, GptCallStatus.DISABLED)

        if not budget_allowed:
            return _result(False, GptCallStatus.BUDGET_EXCEEDED)

        if self._client is None or not os.getenv("OPENAI_API_KEY", ""):
            return _result(
                False,
                GptCallStatus.API_KEY_MISSING,
                error_message="OPENAI_API_KEY not set or client unavailable",
            )

        circuit_key = self._circuit_key(model, task_mode)
        if self._breaker is not None and self._breaker.is_open(circuit_key):
            return _result(False, GptCallStatus.CIRCUIT_OPEN)

        # ── GPT call with timeout ─────────────────────────────────────────────
        raw_text: str | None = None
        call_exc: Exception | None = None
        call_status: str = GptCallStatus.OK

        try:
            if inspect.iscoroutinefunction(self._client):
                coro = self._client(messages, model, timeout_s)
            else:
                coro = asyncio.to_thread(self._client, messages, model, timeout_s)
            raw_text = await asyncio.wait_for(coro, timeout=timeout_s)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        except asyncio.TimeoutError as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            call_exc = exc
            call_status = GptCallStatus.TIMEOUT
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            call_exc = exc
            call_status = classify_exception(exc)

        if call_exc is not None:
            should_open = call_status in _CIRCUIT_TRIGGERING_STATUSES
            if self._breaker is not None and should_open:
                self._breaker.record_failure(circuit_key)
            return GptSafeResult(
                ok=False,
                status=call_status,
                task_mode=task_mode,
                error_message=sanitize_error_message(f"{type(call_exc).__name__}: {call_exc}"),
                latency_ms=latency_ms,
                model=model,
                timeout_ms=effective_ms,
                provider=self._provider,
                should_fallback_to_local=True,
                should_open_circuit=should_open,
                metadata=_meta,
            )

        # Empty / null response
        raw_safe = (raw_text or "").strip()
        if not raw_safe:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return GptSafeResult(
                ok=False,
                status=GptCallStatus.INVALID_JSON,
                task_mode=task_mode,
                raw_text="",
                error_message="empty_response",
                latency_ms=latency_ms,
                model=model,
                timeout_ms=effective_ms,
                provider=self._provider,
                should_fallback_to_local=True,
                should_open_circuit=False,
                metadata=_meta,
            )

        raw_stored = truncate_raw_text(raw_safe)

        # ── Parse ─────────────────────────────────────────────────────────────
        parsed: Any = None
        parse_exc: Exception | None = None
        try:
            parsed = parse_fn(raw_safe)
        except Exception as exc:
            parse_exc = exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if parse_exc is not None:
            return GptSafeResult(
                ok=False,
                status=GptCallStatus.INVALID_JSON,
                task_mode=task_mode,
                raw_text=raw_stored,
                error_message=sanitize_error_message(f"{type(parse_exc).__name__}: {parse_exc}"),
                latency_ms=latency_ms,
                model=model,
                timeout_ms=effective_ms,
                provider=self._provider,
                should_fallback_to_local=True,
                should_open_circuit=False,
                metadata=_meta,
            )

        # ── Success ───────────────────────────────────────────────────────────
        if self._breaker is not None:
            self._breaker.record_success(circuit_key)
        return GptSafeResult(
            ok=True,
            status=GptCallStatus.OK,
            task_mode=task_mode,
            parsed=parsed,
            raw_text=raw_stored,
            latency_ms=latency_ms,
            model=model,
            timeout_ms=effective_ms,
            provider=self._provider,
            should_fallback_to_local=False,
            should_open_circuit=False,
            metadata=_meta,
        )

    def _circuit_key(self, model: str, task_mode: str) -> str:
        if self._breaker is not None and hasattr(self._breaker, "circuit_key"):
            return self._breaker.circuit_key(model, task_mode)
        return f"{(model or 'unknown').strip()}:{(task_mode or 'generic').strip()}"


# ---------------------------------------------------------------------------
# Public API (legacy synchronous wrapper — kept for backward compat)
# ---------------------------------------------------------------------------


def call_gpt_safely(
    gpt_callable: Callable[[], str],
    *,
    parse_fn: Callable[[str], Any],
    task_mode: str | None = None,
    timeout_ms: int = 700,
    model: str | None = None,
    metadata: dict | None = None,
) -> GptSafeResult:
    """Call ``gpt_callable`` with full exception isolation and timeout handling.

    Parameters
    ----------
    gpt_callable:
        A zero-argument callable that calls the OpenAI client and returns
        the raw response text (``response.choices[0].message.content``).
        Must not perform its own exception swallowing.
    parse_fn:
        A callable that takes the raw text string and returns a parsed object.
        Should raise ``ValueError`` or ``json.JSONDecodeError`` on parse failure.
    task_mode:
        Bucket / task label for logging (e.g. "idle_menu_item_resolution").
    timeout_ms:
        The timeout enforced on the call.  Informational only — the actual
        timeout must be passed to the OpenAI client constructor or the
        ``timeout=`` parameter of ``create()``.
    model:
        OpenAI model name (for logging).
    metadata:
        Optional extra metadata dict for structured logging (not used here,
        but passed through for caller convenience).

    Returns
    -------
    ``GptSafeResult`` — never raises.
    """
    t0 = time.perf_counter()

    # ── Guard: API key missing ───────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return GptSafeResult(
            ok=False,
            status=GptCallStatus.API_KEY_MISSING,
            task_mode=task_mode,
            error_message="OPENAI_API_KEY not set",
            model=model,
            timeout_ms=timeout_ms,
            should_fallback_to_local=True,
            should_open_circuit=False,
        )

    # ── Raw GPT call ─────────────────────────────────────────────────────────
    raw_text: str | None = None
    latency_ms: float | None = None
    call_exc: Exception | None = None
    call_status: str = GptCallStatus.OK

    try:
        raw_text = gpt_callable()
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        call_exc = exc
        call_status = _classify_exception(exc)

    if call_exc is not None:
        return GptSafeResult(
            ok=False,
            status=call_status,
            task_mode=task_mode,
            raw_text=None,
            error_message=_sanitize_error(call_exc),
            latency_ms=latency_ms,
            model=model,
            timeout_ms=timeout_ms,
            should_fallback_to_local=True,
            should_open_circuit=call_status in _CIRCUIT_TRIGGERING_STATUSES,
        )

    # Empty / null response
    raw_safe = (raw_text or "").strip()
    if not raw_safe:
        return GptSafeResult(
            ok=False,
            status=GptCallStatus.INVALID_JSON,
            task_mode=task_mode,
            raw_text="",
            error_message="empty_response",
            latency_ms=latency_ms,
            model=model,
            timeout_ms=timeout_ms,
            should_fallback_to_local=True,
            should_open_circuit=False,
        )

    # Truncate raw text for result storage
    raw_stored = raw_safe[:_MAX_RAW_TEXT_BYTES]

    # ── Parse ─────────────────────────────────────────────────────────────────
    parse_exc: Exception | None = None
    parsed: Any = None

    try:
        parsed = parse_fn(raw_safe)
    except Exception as exc:
        parse_exc = exc

    if parse_exc is not None:
        return GptSafeResult(
            ok=False,
            status=GptCallStatus.INVALID_JSON,
            task_mode=task_mode,
            raw_text=raw_stored,
            error_message=_sanitize_error(parse_exc),
            latency_ms=latency_ms,
            model=model,
            timeout_ms=timeout_ms,
            should_fallback_to_local=True,
            should_open_circuit=False,
        )

    # ── Success ───────────────────────────────────────────────────────────────
    return GptSafeResult(
        ok=True,
        status=GptCallStatus.OK,
        task_mode=task_mode,
        parsed=parsed,
        raw_text=raw_stored,
        latency_ms=latency_ms,
        model=model,
        timeout_ms=timeout_ms,
        should_fallback_to_local=False,
        should_open_circuit=False,
    )
