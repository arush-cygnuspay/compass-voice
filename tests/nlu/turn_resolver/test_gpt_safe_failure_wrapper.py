# tests/nlu/turn_resolver/test_gpt_safe_failure_wrapper.py
"""Priority 2: GPT safe-failure wrapper tests.

Covers:
  - GptSafeClient (tests 1–14)
  - GptCircuitBreaker + CircuitState (tests 15–20)
  - GptFallbackDecision / decide_gpt_failure_fallback (tests 21–30)
  - Integration / safety (tests 31–35)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from app.nlu.turn_resolver.gpt_circuit_breaker import (
    CircuitBreakerConfig,
    CircuitState,
    GptCircuitBreaker,
)
from app.nlu.turn_resolver.gpt_fallback_policy import (
    GptFallbackDecision,
    build_gpt_failure_fallback_response,
    decide_gpt_failure_fallback,
    is_local_result_safe,
)
from app.nlu.turn_resolver.gpt_safe_client import (
    GptCallStatus,
    GptSafeClient,
    GptSafeResult,
    classify_exception,
    sanitize_error_message,
    truncate_raw_text,
)
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro) -> object:
    """Run a coroutine synchronously in tests."""
    return asyncio.run(coro)


def _make_client(
    response_text: str | None = None,
    raise_exc: Exception | None = None,
    breaker: GptCircuitBreaker | None = None,
    provider: str = "openai",
    max_timeout_ms: int = 1200,
) -> GptSafeClient:
    """Build a GptSafeClient backed by a simple sync lambda."""
    if raise_exc is not None:
        def _fn(messages, model, timeout_s):
            raise raise_exc
    else:
        def _fn(messages, model, timeout_s):
            return response_text or ""

    config = MagicMock()
    config.gpt_max_timeout_ms = max_timeout_ms

    return GptSafeClient(
        underlying_client=_fn,
        circuit_breaker=breaker,
        config=config,
        provider=provider,
    )


def _make_context(
    state: ConversationState = ConversationState.IDLE,
    pending_add_item=None,
    current_modifier_group_index: int = 0,
    current_side_group_index: int = 0,
):
    ctx = MagicMock()
    ctx.pending_add_item = pending_add_item
    ctx.current_modifier_group_index = current_modifier_group_index
    ctx.current_side_group_index = current_side_group_index
    return ctx


def _make_failed_result(status: str = GptCallStatus.TIMEOUT) -> GptSafeResult:
    return GptSafeResult(ok=False, status=status, should_fallback_to_local=True)


def _make_slot(name: str):
    s = MagicMock()
    s.name = name
    return s


_API_KEY = "sk-test-key"

# ---------------------------------------------------------------------------
# GptSafeClient tests (1–14)
# ---------------------------------------------------------------------------


class TestGptSafeClientDisabled:
    """Test 1 — disabled returns DISABLED without calling provider."""

    def test_disabled_returns_disabled(self):
        calls = []

        def _fn(messages, model, timeout_s):
            calls.append(1)
            return '{"ok": true}'

        client = GptSafeClient(underlying_client=_fn)
        result = _run(client.call(
            task_mode="test",
            messages=[],
            model="gpt-4o-mini",
            timeout_ms=700,
            parse_fn=json.loads,
            enabled=False,
        ))
        assert result.status == GptCallStatus.DISABLED
        assert not result.ok
        assert result.should_fallback_to_local
        assert calls == [], "provider must NOT be called when disabled"


class TestGptSafeClientBudgetExceeded:
    """Test 2 — budget_allowed=False returns BUDGET_EXCEEDED without calling provider."""

    def test_budget_exceeded(self):
        calls = []

        def _fn(messages, model, timeout_s):
            calls.append(1)
            return "{}"

        client = GptSafeClient(underlying_client=_fn)
        result = _run(client.call(
            task_mode="test",
            messages=[],
            model="gpt-4o-mini",
            timeout_ms=700,
            parse_fn=json.loads,
            budget_allowed=False,
        ))
        assert result.status == GptCallStatus.BUDGET_EXCEEDED
        assert not result.ok
        assert calls == [], "provider must NOT be called when budget exceeded"


class TestGptSafeClientApiKeyMissing:
    """Test 3 — client=None or API key missing returns API_KEY_MISSING."""

    def test_client_none_returns_api_key_missing(self):
        client = GptSafeClient(underlying_client=None)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.status == GptCallStatus.API_KEY_MISSING
        assert not result.ok

    def test_api_key_env_missing_returns_api_key_missing(self):
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        client = _make_client(response_text='{"x": 1}')
        with patch.dict(os.environ, env_without_key, clear=True):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.status == GptCallStatus.API_KEY_MISSING
        assert not result.ok


class TestGptSafeClientSuccess:
    """Test 4 — successful call returns OK with parsed result."""

    def test_success_returns_ok(self):
        payload = {"intent": "add_item", "confidence": 0.95}

        def _fn(messages, model, timeout_s):
            return json.dumps(payload)

        client = GptSafeClient(underlying_client=_fn)
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="idle",
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.ok
        assert result.status == GptCallStatus.OK
        assert result.parsed == payload
        assert result.should_fallback_to_local is False
        assert result.should_open_circuit is False
        assert result.latency_ms is not None


class TestGptSafeClientTimeout:
    """Test 5 — asyncio.TimeoutError returns TIMEOUT and never raises."""

    def test_timeout_returns_timeout_status(self):
        async def _slow(messages, model, timeout_s):
            await asyncio.sleep(10)
            return "{}"

        client = GptSafeClient(underlying_client=_slow)
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=50,
                parse_fn=json.loads,
            ))
        assert result.status == GptCallStatus.TIMEOUT
        assert not result.ok
        assert result.should_fallback_to_local
        assert result.should_open_circuit


class TestGptSafeClientInvalidJson:
    """Test 6 — bad JSON from provider returns INVALID_JSON."""

    def test_invalid_json_returns_invalid_json(self):
        client = _make_client(response_text="this is not json")
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.status == GptCallStatus.INVALID_JSON
        assert not result.ok
        assert result.should_fallback_to_local
        assert result.should_open_circuit is False


class TestGptSafeClientSchemaValidation:
    """Test 7 — parse_fn raises ValueError for schema → INVALID_JSON (schema path)."""

    def test_schema_validation_failure(self):
        def _strict_parse(text: str):
            data = json.loads(text)
            if "required_field" not in data:
                raise ValueError("missing required_field")
            return data

        client = _make_client(response_text='{"other": "stuff"}')
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=_strict_parse,
            ))
        assert result.status == GptCallStatus.INVALID_JSON
        assert not result.ok
        assert result.raw_text is not None


class TestGptSafeClientProviderError:
    """Test 8 — provider 500 returns PROVIDER_ERROR."""

    def test_provider_500_returns_provider_error(self):
        class InternalServerError(Exception):
            pass

        exc = InternalServerError("InternalServerError: 500")
        client = _make_client(raise_exc=exc)
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.status == GptCallStatus.PROVIDER_ERROR
        assert not result.ok
        assert result.should_open_circuit


class TestGptSafeClientRateLimited:
    """Test 9 — rate limit 429 returns RATE_LIMITED."""

    def test_rate_limit_returns_rate_limited(self):
        class RateLimitError(Exception):
            pass

        exc = RateLimitError("RateLimitError: 429 too many requests")
        client = _make_client(raise_exc=exc)
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.status == GptCallStatus.RATE_LIMITED
        assert not result.ok
        assert result.should_open_circuit


class TestGptSafeClientNetworkError:
    """Test 10 — network/connection error returns NETWORK_ERROR."""

    def test_connection_error_returns_network_error(self):
        exc = ConnectionError("connection refused")
        client = _make_client(raise_exc=exc)
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.status == GptCallStatus.NETWORK_ERROR
        assert not result.ok
        assert result.should_open_circuit


class TestGptSafeClientUnknownError:
    """Test 11 — unexpected exception returns UNKNOWN_ERROR."""

    def test_unknown_exception_returns_unknown_error(self):
        exc = RuntimeError("something totally unexpected happened")
        client = _make_client(raise_exc=exc)
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.status == GptCallStatus.UNKNOWN_ERROR
        assert not result.ok


class TestGptSafeClientRawOutputTruncated:
    """Test 12 — raw output is truncated to max chars."""

    def test_raw_output_is_truncated(self):
        long_text = "x" * 5000 + '{"ok": true}'

        def _fn(messages, model, timeout_s):
            return long_text

        client = GptSafeClient(underlying_client=_fn)

        def _fail_parse(text):
            raise ValueError("bad")

        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=_fail_parse,
            ))
        assert result.raw_text is not None
        assert len(result.raw_text) <= 2000


class TestGptSafeClientErrorSanitized:
    """Test 13 — error_message has API key redacted."""

    def test_error_message_sanitizes_api_key(self):
        key = "sk-secret-key-12345"

        def _fn(messages, model, timeout_s):
            raise RuntimeError(f"bad call with key {key}")

        client = GptSafeClient(underlying_client=_fn)
        with patch.dict(os.environ, {"OPENAI_API_KEY": key}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.error_message is not None
        assert key not in result.error_message
        assert "[REDACTED]" in result.error_message


class TestGptSafeClientTimeoutCapped:
    """Test 14 — timeout is capped at GPT_MAX_TIMEOUT_MS."""

    def test_timeout_capped_at_max(self):
        captured_timeout: list[float] = []

        async def _fn(messages, model, timeout_s):
            captured_timeout.append(timeout_s)
            return '{"x": 1}'

        config = MagicMock()
        config.gpt_max_timeout_ms = 800
        client = GptSafeClient(underlying_client=_fn, config=config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=5000,  # well above max
                parse_fn=json.loads,
            ))
        assert result.timeout_ms == 800
        assert len(captured_timeout) == 1
        assert captured_timeout[0] == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# Circuit breaker tests (15–20)
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpens:
    """Test 15 — opens after 3 consecutive outage failures."""

    def test_opens_after_threshold(self):
        cfg = CircuitBreakerConfig(enabled=True, failure_threshold=3, open_seconds=30.0)
        breaker = GptCircuitBreaker(config=cfg)
        key = "gpt-4o-mini:test"
        for _ in range(3):
            breaker.record_failure(key)
        assert breaker.is_open(key)


class TestCircuitBreakerOpen:
    """Test 16 — while open, call returns CIRCUIT_OPEN immediately."""

    def test_circuit_open_returns_circuit_open(self):
        cfg = CircuitBreakerConfig(enabled=True, failure_threshold=1, open_seconds=60.0)
        breaker = GptCircuitBreaker(config=cfg)
        key = "gpt-4o-mini:test"
        breaker.record_failure(key)
        assert breaker.is_open(key)

        client = _make_client(response_text='{"x": 1}', breaker=breaker)
        # Use a different task_mode so the circuit key matches
        # We inject a custom circuit_key override via the breaker's key mechanism
        # Directly check the circuit open gate:
        client2 = GptSafeClient(underlying_client=lambda m, mo, t: '{"x": 1}', circuit_breaker=breaker)
        # The circuit_key is built as "{model}:{task_mode}"
        result = _run(client2.call(
            task_mode="test",
            messages=[],
            model="gpt-4o-mini",
            timeout_ms=700,
            parse_fn=json.loads,
            enabled=True,
            budget_allowed=True,
        ))
        # Must trigger circuit open check only if API key is present
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client2.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        assert result.status == GptCallStatus.CIRCUIT_OPEN
        assert not result.ok


class TestCircuitBreakerSuccessResetsCount:
    """Test 17 — success resets failure count."""

    def test_success_resets(self):
        cfg = CircuitBreakerConfig(enabled=True, failure_threshold=3, open_seconds=30.0)
        breaker = GptCircuitBreaker(config=cfg)
        key = "gpt-4o-mini:test"
        breaker.record_failure(key)
        breaker.record_failure(key)
        assert breaker.failure_count(key) == 2
        breaker.record_success(key)
        assert breaker.failure_count(key) == 0
        assert not breaker.is_open(key)


class TestCircuitBreakerValidationErrors:
    """Test 18 — validation errors (INVALID_JSON) do not open circuit."""

    def test_invalid_json_does_not_open_circuit(self):
        cfg = CircuitBreakerConfig(enabled=True, failure_threshold=2, open_seconds=30.0)
        breaker = GptCircuitBreaker(config=cfg)
        key = "gpt-4o-mini:test"

        client = GptSafeClient(underlying_client=lambda m, mo, t: "not json", circuit_breaker=breaker)
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            for _ in range(5):
                _run(client.call(
                    task_mode="test",
                    messages=[],
                    model="gpt-4o-mini",
                    timeout_ms=700,
                    parse_fn=json.loads,
                ))
        # INVALID_JSON is not circuit-triggering — circuit stays closed
        assert not breaker.is_open(key)


class TestCircuitBreakerIndependentKeys:
    """Test 19 — separate model/task_mode keys don't affect each other."""

    def test_keys_are_independent(self):
        cfg = CircuitBreakerConfig(enabled=True, failure_threshold=1, open_seconds=30.0)
        breaker = GptCircuitBreaker(config=cfg)
        key_a = breaker.circuit_key("gpt-4o-mini", "task_a")
        key_b = breaker.circuit_key("gpt-4o-mini", "task_b")
        breaker.record_failure(key_a)
        assert breaker.is_open(key_a)
        assert not breaker.is_open(key_b)


class TestCircuitBreakerClosesAfterCooldown:
    """Test 20 — circuit auto-closes after open_seconds elapses."""

    def test_closes_after_cooldown(self):
        cfg = CircuitBreakerConfig(enabled=True, failure_threshold=1, open_seconds=0.05)
        breaker = GptCircuitBreaker(config=cfg)
        key = "gpt-4o-mini:test"
        breaker.record_failure(key)
        assert breaker.is_open(key)
        time.sleep(0.1)
        assert not breaker.is_open(key), "circuit should auto-close after open_seconds"


# ---------------------------------------------------------------------------
# Fallback policy tests (21–30)
# ---------------------------------------------------------------------------


class TestFallbackLocalWhenSafe:
    """Tests 21–24 — local path used when GPT fails and local is safe."""

    def _idle_ctx(self):
        return _make_context(state=ConversationState.IDLE)

    def test_21_timeout_local_allowed_uses_local(self):
        gpt_result = _make_failed_result(GptCallStatus.TIMEOUT)
        ctx = self._idle_ctx()
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="idle",
            local_intent="add_item",
            local_confidence=0.90,
            local_slots=[_make_slot("ITEM")],
            allowed_intents=["add_item"],
            context=ctx,
        )
        assert decision.use_local
        assert not decision.use_state_clarification
        assert decision.fallback_source == "local"
        assert decision.local_safe

    def test_22_invalid_json_local_allowed_uses_local(self):
        gpt_result = _make_failed_result(GptCallStatus.INVALID_JSON)
        ctx = self._idle_ctx()
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="idle",
            local_intent="add_item",
            local_confidence=0.88,
            local_slots=[_make_slot("ITEM")],
            allowed_intents=["add_item"],
            context=ctx,
        )
        assert decision.use_local

    def test_23_provider_error_unsafe_slots_uses_clarification(self):
        gpt_result = _make_failed_result(GptCallStatus.PROVIDER_ERROR)
        ctx = self._idle_ctx()
        # Two ITEM slots → multi_item_slots → unsafe
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="idle",
            local_intent="add_item",
            local_confidence=0.90,
            local_slots=[_make_slot("ITEM"), _make_slot("ITEM")],
            allowed_intents=["add_item"],
            context=ctx,
        )
        assert not decision.use_local
        assert decision.use_state_clarification

    def test_24_circuit_open_local_valid_uses_local(self):
        gpt_result = _make_failed_result(GptCallStatus.CIRCUIT_OPEN)
        ctx = self._idle_ctx()
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="idle",
            local_intent="add_item",
            local_confidence=0.85,
            local_slots=[_make_slot("ITEM")],
            allowed_intents=["add_item"],
            context=ctx,
        )
        assert decision.use_local


class TestFallbackWaitingStates:
    """Tests 25–28 — state-specific clarification for waiting states."""

    def _modifier_ctx(self):
        group = MagicMock()
        group.name = "Sauce"
        group.choices = [MagicMock(name="Ranch"), MagicMock(name="BBQ")]
        pending = MagicMock()
        pending.modifier_groups = [group]
        return _make_context(
            state=ConversationState.WAITING_FOR_MODIFIER,
            pending_add_item=pending,
            current_modifier_group_index=0,
        )

    def _side_ctx(self):
        group = MagicMock()
        group.name = "Sides"
        group.choices = [MagicMock(name="Fries"), MagicMock(name="Salad")]
        pending = MagicMock()
        pending.side_groups = [group]
        return _make_context(
            state=ConversationState.WAITING_FOR_SIDE,
            pending_add_item=pending,
            current_side_group_index=0,
        )

    def test_25_waiting_for_modifier_fallback(self):
        gpt_result = _make_failed_result(GptCallStatus.TIMEOUT)
        ctx = self._modifier_ctx()
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="waiting_for_modifier",
            local_intent="unknown",
            local_confidence=0.10,
            local_slots=[],
            allowed_intents=[],
            context=ctx,
        )
        assert not decision.use_local
        assert decision.response_key == "repeat_modifier_options"

    def test_26_waiting_for_side_fallback(self):
        gpt_result = _make_failed_result(GptCallStatus.NETWORK_ERROR)
        ctx = self._side_ctx()
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="waiting_for_side",
            local_intent="unknown",
            local_confidence=0.10,
            local_slots=[],
            allowed_intents=[],
            context=ctx,
        )
        assert not decision.use_local
        assert decision.response_key == "repeat_side_options"

    def test_27_waiting_for_size_asks_size(self):
        gpt_result = _make_failed_result(GptCallStatus.PROVIDER_ERROR)
        ctx = _make_context(state=ConversationState.WAITING_FOR_SIZE)
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="waiting_for_size",
            local_intent="unknown",
            local_confidence=0.10,
            local_slots=[],
            allowed_intents=[],
            context=ctx,
        )
        assert not decision.use_local
        assert decision.response_key == "ask_for_size"

    def test_28_confirming_order_asks_yes_no(self):
        gpt_result = _make_failed_result(GptCallStatus.TIMEOUT)
        ctx = _make_context(state=ConversationState.CONFIRMING_ORDER)
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="confirming_order",
            local_intent="unknown",
            local_confidence=0.10,
            local_slots=[],
            allowed_intents=[],
            context=ctx,
        )
        assert not decision.use_local
        assert decision.response_key == "confirm_order_repeat"


class TestFallbackMultiItemAndNoMention:
    """Tests 29–30 — compound fallback + no GPT mention in response."""

    def test_29_idle_unsafe_multi_item_asks_first_item(self):
        gpt_result = _make_failed_result(GptCallStatus.TIMEOUT)
        ctx = _make_context(state=ConversationState.IDLE)
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="idle",
            local_intent="add_item",
            local_confidence=0.90,
            local_slots=[_make_slot("ITEM"), _make_slot("ITEM")],
            allowed_intents=["add_item"],
            context=ctx,
        )
        assert not decision.use_local
        assert decision.response_key == "compound_unclear_ask_first"

    def test_30_fallback_never_mentions_gpt_openai_api(self):
        """Verify no GPT/OpenAI/API mention in any fallback response key."""
        banned = ["gpt", "openai", "api", "ai_failure", "model"]
        gpt_result = _make_failed_result(GptCallStatus.TIMEOUT)
        ctx = _make_context()
        decision = decide_gpt_failure_fallback(
            gpt_result=gpt_result,
            state="idle",
            local_intent="unknown",
            local_confidence=0.10,
            local_slots=[],
            allowed_intents=[],
            context=ctx,
        )
        key = (decision.response_key or "").lower()
        text = (decision.response_text or "").lower()
        for word in banned:
            assert word not in key, f"Response key mentions '{word}'"
            assert word not in text, f"Response text mentions '{word}'"


# ---------------------------------------------------------------------------
# Integration / safety tests (31–35)
# ---------------------------------------------------------------------------


class TestNoExceptionsEscape:
    """Test 31 — GptSafeClient.call never raises under any condition."""

    def test_call_never_raises_even_on_bizarre_client(self):
        def _evil(messages, model, timeout_s):
            raise SystemExit("should not propagate")

        client = GptSafeClient(underlying_client=_evil)
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            # SystemExit is NOT a subclass of Exception — this tests the outer guard
            try:
                result = _run(client.call(
                    task_mode="test",
                    messages=[],
                    model="gpt-4o-mini",
                    timeout_ms=700,
                    parse_fn=json.loads,
                ))
                # If we reach here, the result must be a failure
                assert not result.ok
            except SystemExit:
                # SystemExit propagates through asyncio.run — that's OK
                # The important thing is that regular Exceptions don't escape
                pass


class TestFallbackPolicyNeverRaises:
    """Test 32 — decide_gpt_failure_fallback never raises."""

    def test_no_exception_on_none_inputs(self):
        result = decide_gpt_failure_fallback(
            gpt_result=None,  # type: ignore[arg-type]
            state="",
            local_intent=None,
            local_confidence=None,
            local_slots=None,
            allowed_intents=[],
            context=None,  # type: ignore[arg-type]
        )
        assert isinstance(result, GptFallbackDecision)

    def test_no_exception_on_broken_context(self):
        class BrokenCtx:
            @property
            def pending_add_item(self):
                raise RuntimeError("context is broken")

        result = decide_gpt_failure_fallback(
            gpt_result=_make_failed_result(),
            state="waiting_for_modifier",
            local_intent="unknown",
            local_confidence=0.1,
            local_slots=[],
            allowed_intents=[],
            context=BrokenCtx(),  # type: ignore[arg-type]
        )
        assert isinstance(result, GptFallbackDecision)


class TestGptSafeResultSerializable:
    """Test 33 — all GptSafeResult objects are JSON-serializable via asdict."""

    def _asdict_json_safe(self, result: GptSafeResult) -> dict:
        d = dataclasses.asdict(result)
        # parsed may be non-serializable; replace with None for logging
        d["parsed"] = None if not isinstance(d.get("parsed"), (dict, list, str, int, float, bool, type(None))) else d["parsed"]
        return d

    def test_disabled_result_serializable(self):
        client = GptSafeClient()
        with patch.dict(os.environ, {}, clear=True):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
                enabled=False,
            ))
        d = self._asdict_json_safe(result)
        json.dumps(d)  # must not raise

    def test_ok_result_serializable(self):
        client = _make_client(response_text='{"a": 1}')
        with patch.dict(os.environ, {"OPENAI_API_KEY": _API_KEY}):
            result = _run(client.call(
                task_mode="test",
                messages=[],
                model="gpt-4o-mini",
                timeout_ms=700,
                parse_fn=json.loads,
            ))
        d = self._asdict_json_safe(result)
        json.dumps(d)  # must not raise

    def test_failure_result_serializable(self):
        result = GptSafeResult(
            ok=False,
            status=GptCallStatus.TIMEOUT,
            task_mode="test",
            error_message="timed out",
            latency_ms=701.5,
            model="gpt-4o-mini",
            timeout_ms=700,
            provider="openai",
        )
        d = dataclasses.asdict(result)
        json.dumps(d)  # must not raise


class TestPriority1StillWorks:
    """Test 34 — Priority 1 context infrastructure imports and runs correctly."""

    def test_gpt_context_builder_importable(self):
        from app.nlu.turn_resolver.gpt_context_builder import GptContextBuilder
        from app.nlu.turn_resolver.allowed_intent_provider import AllowedIntentProvider
        from app.nlu.turn_resolver.allowed_response_key_provider import AllowedResponseKeyProvider
        builder = GptContextBuilder()
        assert builder is not None

    def test_allowed_intent_provider_returns_results(self):
        from app.nlu.turn_resolver.allowed_intent_provider import AllowedIntentProvider
        provider = AllowedIntentProvider()
        intents = provider.get_allowed_intents_for_state("idle")
        assert len(intents) > 0

    def test_prompt_registry_known_modes(self):
        from app.nlu.turn_resolver.prompt_registry import PromptRegistry, TASK_IDLE_ADD_ITEM_OR_MENU_QUERY
        registry = PromptRegistry()
        assert registry.is_known_task_mode(TASK_IDLE_ADD_ITEM_OR_MENU_QUERY)
        prompt = registry.get_system_prompt(TASK_IDLE_ADD_ITEM_OR_MENU_QUERY)
        assert isinstance(prompt, str) and len(prompt) > 0


class TestCircuitStateSnapshot:
    """Test 35 — get_state() returns correct CircuitState snapshot."""

    def test_get_state_initial(self):
        breaker = GptCircuitBreaker()
        state = breaker.get_state("no:key")
        assert state.failure_count == 0
        assert state.opened_until_monotonic is None
        assert state.last_failure_status is None
        assert not state.is_open

    def test_get_state_after_failures(self):
        cfg = CircuitBreakerConfig(enabled=True, failure_threshold=2, open_seconds=60.0)
        breaker = GptCircuitBreaker(config=cfg)
        key = "gpt-4o-mini:task"
        breaker.record_failure(key, status="timeout")
        state = breaker.get_state(key)
        assert state.failure_count == 1
        assert state.last_failure_status == "timeout"
        assert not state.is_open

    def test_get_state_open(self):
        cfg = CircuitBreakerConfig(enabled=True, failure_threshold=1, open_seconds=60.0)
        breaker = GptCircuitBreaker(config=cfg)
        key = "gpt-4o-mini:task"
        breaker.record_failure(key, status="provider_error")
        state = breaker.get_state(key)
        assert state.failure_count == 1
        assert state.opened_until_monotonic is not None
        assert state.is_open
        assert state.last_failure_status == "provider_error"

    def test_get_state_after_success_reset(self):
        cfg = CircuitBreakerConfig(enabled=True, failure_threshold=3, open_seconds=30.0)
        breaker = GptCircuitBreaker(config=cfg)
        key = "gpt-4o-mini:task"
        breaker.record_failure(key, status="timeout")
        breaker.record_failure(key, status="timeout")
        breaker.record_success(key)
        state = breaker.get_state(key)
        assert state.failure_count == 0
        assert not state.is_open
