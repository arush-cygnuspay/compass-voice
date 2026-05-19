# tests/nlu/turn_resolver/test_gpt_failure_isolation.py
"""GPT failure isolation tests — 20 scenarios.

Verifies that:
  * Every GPT failure mode degrades gracefully to local deterministic flow.
  * Cart is never mutated on GPT failure paths.
  * Circuit breaker opens after threshold failures and closes after cool-down.
  * State-specific fallback responses are correct.
  * Exception propagation from GPT never reaches TurnEngine / handlers.
  * Log fields (gpt_status, fallback_source, etc.) are always populated.

Tests map directly to the spec items 1–20.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from app.config.semantic_repair import SemanticRepairConfig
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult, SlotValue
from app.nlu.turn_resolver.bucket_policy import BUCKET_IDLE_ITEM, BUCKET_MULTI_ITEM, BUCKET_OPTION
from app.nlu.turn_resolver.final_turn_decision_resolver import resolve
from app.nlu.turn_resolver.gpt_circuit_breaker import (
    CircuitBreakerConfig,
    GptCircuitBreaker,
)
from app.nlu.turn_resolver.gpt_fallback_policy import (
    GptFallbackResponse,
    build_gpt_failure_fallback_response,
    is_local_result_safe,
)
from app.nlu.turn_resolver.gpt_safe_client import (
    GptCallStatus,
    GptSafeResult,
    call_gpt_safely,
)
from app.nlu.turn_resolver.schemas import GptTurnResolution, ResolvedItemPlan
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDLE = ConversationState.IDLE
_WAITING_MOD = ConversationState.WAITING_FOR_MODIFIER
_WAITING_SIDE = ConversationState.WAITING_FOR_SIDE
_WAITING_SIZE = ConversationState.WAITING_FOR_SIZE
_CONFIRMING_ORDER = ConversationState.CONFIRMING_ORDER


def _nlu(
    intent: Intent = Intent.ADD_ITEM,
    confidence: float = 0.9,
    text: str = "large burger",
    slots: tuple = (),
) -> NLUResult:
    return NLUResult(
        effective_intent=intent,
        intent_confidence=confidence,
        raw_text=text,
        normalized_text=text,
        slots=slots,
    )


def _slot(name: str, value: str) -> SlotValue:
    return SlotValue(name=name, value=value)


def _cfg(b0: str = "inline", b2: str = "inline", b3: str = "inline") -> SemanticRepairConfig:
    return SemanticRepairConfig(
        phase=2,
        model="gpt-4o-mini",
        timeout_seconds=0.35,
        bucket_0_mode=b0,
        bucket_2_mode=b2,
        bucket_3_mode=b3,
    )


def _ctx() -> ConversationContext:
    return ConversationContext()


def _breaker(threshold: int = 3, open_seconds: float = 30.0) -> GptCircuitBreaker:
    """Fresh circuit breaker for test isolation."""
    return GptCircuitBreaker(
        config=CircuitBreakerConfig(
            enabled=True,
            failure_threshold=threshold,
            open_seconds=open_seconds,
        )
    )


def _gpt_error_result(bucket: str, error_keyword: str = "timeout") -> GptTurnResolution:
    return GptTurnResolution(
        bucket=bucket,
        decision="error",
        gpt_called=True,
        parse_error=f"{error_keyword}_error_simulated",
    )


# ---------------------------------------------------------------------------
# Test 1 — GPT timeout + safe local ADD_ITEM → local add flow continues
# ---------------------------------------------------------------------------

class TestT01_GptTimeoutSafeLocal:
    def test_gpt_timeout_uses_local_when_safe(self) -> None:
        # confidence=0.4 < LOW_CONFIDENCE_THRESHOLD (0.55) → triggers Bucket 0
        # ADD_ITEM intent is safe for local execution (is_local_result_safe returns True)
        nlu = _nlu(intent=Intent.ADD_ITEM, confidence=0.4, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        timeout_result = _gpt_error_result(BUCKET_IDLE_ITEM, "timeout")
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=timeout_result,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                circuit_breaker=_breaker(),
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False
        assert decision.local_is_safe is True
        assert decision.gpt_status in {GptCallStatus.TIMEOUT, GptCallStatus.UNKNOWN_ERROR}


# ---------------------------------------------------------------------------
# Test 2 — GPT timeout + unsafe multi-item slots → no cart mutation, ask first item
# ---------------------------------------------------------------------------

class TestT02_GptTimeoutUnsafeMultiItem:
    def test_gpt_timeout_unsafe_multi_item_slots_blocked(self) -> None:
        # Two ITEM slots = compound utterance = unsafe for local execution
        slots = (_slot("ITEM", "burger"), _slot("ITEM", "fries"))
        nlu = _nlu(
            intent=Intent.ADD_ITEM,
            confidence=0.7,
            text="burger and fries",
            slots=slots,
        )
        cfg = _cfg(b3="inline")
        ctx = _ctx()

        timeout_result = _gpt_error_result(BUCKET_MULTI_ITEM, "timeout")
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=timeout_result,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                circuit_breaker=_breaker(),
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False
        assert decision.local_is_safe is False

        # Verify cart not mutated
        assert ctx.pending_add_item is None
        assert len(ctx.staged_item_queue) == 0

        # Ask-first-item fallback
        fb = build_gpt_failure_fallback_response(
            state=_IDLE,
            context=ctx,
            local_intent_value=nlu.effective_intent.value,
            local_slots=nlu.slots,
            local_confidence=nlu.intent_confidence,
            gpt_status=decision.gpt_status,
        )
        assert fb.response_key == "compound_unclear_ask_first"
        assert fb.use_local is False
        assert "compound" in fb.response_key


# ---------------------------------------------------------------------------
# Test 3 — GPT invalid JSON + local valid modifier → local modifier path
# ---------------------------------------------------------------------------

class TestT03_InvalidJsonLocalModifier:
    def test_invalid_json_falls_back_to_local_modifier(self) -> None:
        nlu = _nlu(intent=Intent.ADD_ITEM, confidence=0.8, text="cheddar")
        cfg = _cfg(b2="inline")
        ctx = _ctx()

        json_error_result = _gpt_error_result(BUCKET_OPTION, "json")
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=json_error_result,
        ):
            decision = resolve(
                state=_WAITING_MOD,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                option_match_failed=True,
                choices=["Cheddar", "Swiss"],
                circuit_breaker=_breaker(),
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False
        # json error is INVALID_JSON or UNKNOWN_ERROR
        assert decision.gpt_status in {GptCallStatus.INVALID_JSON, GptCallStatus.UNKNOWN_ERROR}


# ---------------------------------------------------------------------------
# Test 4 — GPT intent not allowed in state → GPT rejected, local used
# ---------------------------------------------------------------------------

class TestT04_InvalidGptIntent:
    def test_gpt_intent_not_in_allowed_rejects(self) -> None:
        from app.nlu.turn_resolver.validators import validate_bucket0_result
        gpt = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="add_items",
            intent="payment_request",  # not in allowed
            items=(ResolvedItemPlan(item_name="Burger"),),
            confidence=0.92,
            gpt_called=True,
        )
        result = validate_bucket0_result(
            gpt,
            allowed_intents=["add_item", "show_menu"],
        )
        assert result.is_safe is False
        assert "intent_not_allowed" in (result.reject_reason or "")


# ---------------------------------------------------------------------------
# Test 5 — GPT provider 500 → local fallback, no crash
# ---------------------------------------------------------------------------

class TestT05_Provider500:
    def test_provider_error_uses_local(self) -> None:
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        provider_error = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="error",
            gpt_called=True,
            parse_error="APIStatusError: 500 internal_server_error",
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=provider_error,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                circuit_breaker=_breaker(),
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False
        assert not isinstance(decision, Exception)


# ---------------------------------------------------------------------------
# Test 6 — GPT rate limit 429 → local fallback, no crash
# ---------------------------------------------------------------------------

class TestT06_RateLimit429:
    def test_rate_limit_falls_back_to_local(self) -> None:
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        rate_limit_result = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="error",
            gpt_called=True,
            parse_error="RateLimitError: 429 rate_limit_exceeded",
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=rate_limit_result,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                circuit_breaker=_breaker(),
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False


# ---------------------------------------------------------------------------
# Test 7 — API key missing → GPT skipped, local flow continues
# ---------------------------------------------------------------------------

class TestT07_ApiKeyMissing:
    def test_api_key_missing_returns_safe_local(self) -> None:
        result = call_gpt_safely(
            gpt_callable=lambda: (_ for _ in ()).throw(RuntimeError("test")),
            parse_fn=lambda x: x,
            task_mode="idle_menu_item_resolution",
            timeout_ms=700,
            model="gpt-4o-mini",
        )
        # Without env var, should return API_KEY_MISSING
        import os
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result2 = call_gpt_safely(
                gpt_callable=lambda: "{}",
                parse_fn=lambda x: {},
                task_mode="test",
                timeout_ms=700,
                model="gpt-4o-mini",
            )
            assert result2.ok is False
            assert result2.status == GptCallStatus.API_KEY_MISSING
            assert result2.should_fallback_to_local is True
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved

    def test_gpt_skipped_api_key_missing_decision_uses_local(self) -> None:
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        skipped_result = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="skipped",
            gpt_called=False,
            skipped_reason="missing_api_key",
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=skipped_result,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                circuit_breaker=_breaker(),
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False
        assert decision.gpt_status == GptCallStatus.API_KEY_MISSING


# ---------------------------------------------------------------------------
# Test 8 — Budget exceeded → GPT skipped, local deterministic flow
# ---------------------------------------------------------------------------

class TestT08_BudgetExceeded:
    def test_budget_exceeded_uses_local(self) -> None:
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        budget_result = GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision="skipped",
            gpt_called=False,
            skipped_reason="daily_budget_exceeded",
        )
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=budget_result,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                circuit_breaker=_breaker(),
            )

        assert decision.source == "local"
        assert decision.apply_gpt is False
        assert decision.gpt_status == GptCallStatus.BUDGET_EXCEEDED


# ---------------------------------------------------------------------------
# Test 9 — Circuit breaker opens after 3 provider failures
# ---------------------------------------------------------------------------

class TestT09_CircuitBreakerOpens:
    def test_circuit_opens_after_threshold_failures(self) -> None:
        breaker = _breaker(threshold=3, open_seconds=30.0)
        key = "gpt-4o-mini:bucket_0"

        assert not breaker.is_open(key)
        for i in range(3):
            breaker.record_failure(key)

        assert breaker.is_open(key)
        assert breaker.failure_count(key) >= 3

    def test_circuit_stays_closed_below_threshold(self) -> None:
        breaker = _breaker(threshold=3)
        key = "gpt-4o-mini:bucket_0"

        breaker.record_failure(key)
        breaker.record_failure(key)
        assert not breaker.is_open(key)


# ---------------------------------------------------------------------------
# Test 10 — Circuit open skips GPT and uses local
# ---------------------------------------------------------------------------

class TestT10_CircuitOpenSkipsGpt:
    def test_circuit_open_skips_gpt_entirely(self) -> None:
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        breaker = _breaker(threshold=1)
        key = breaker.circuit_key(cfg.model, BUCKET_IDLE_ITEM)
        breaker.record_failure(key)
        assert breaker.is_open(key)

        call_count = {"n": 0}
        def _should_not_be_called(*a, **kw):
            call_count["n"] += 1
            return GptTurnResolution(bucket=BUCKET_IDLE_ITEM, decision="add_items", gpt_called=True)

        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            side_effect=_should_not_be_called,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                circuit_breaker=breaker,
            )

        # GPT must NOT have been called
        assert call_count["n"] == 0
        assert decision.gpt_circuit_open is True
        assert decision.gpt_status == GptCallStatus.CIRCUIT_OPEN
        assert decision.source == "local"


# ---------------------------------------------------------------------------
# Test 11 — Circuit resets after success
# ---------------------------------------------------------------------------

class TestT11_CircuitResetsOnSuccess:
    def test_circuit_resets_on_success(self) -> None:
        breaker = _breaker(threshold=3)
        key = "gpt-4o-mini:test"

        breaker.record_failure(key)
        breaker.record_failure(key)
        assert breaker.failure_count(key) == 2

        breaker.record_success(key)
        assert breaker.failure_count(key) == 0
        assert not breaker.is_open(key)

    def test_circuit_auto_closes_after_cooldown(self) -> None:
        breaker = _breaker(threshold=1, open_seconds=0.01)
        key = "gpt-4o-mini:test"
        breaker.record_failure(key)
        assert breaker.is_open(key)

        time.sleep(0.02)  # wait for cool-down
        assert not breaker.is_open(key)


# ---------------------------------------------------------------------------
# Test 12 — WAITING_FOR_SIDE + GPT failure → repeats side options, no crash
# ---------------------------------------------------------------------------

class TestT12_WaitingForSideGptFailure:
    def test_side_options_repeated_on_gpt_failure(self) -> None:
        fb = build_gpt_failure_fallback_response(
            state=_WAITING_SIDE,
            context=_ctx(),
            local_intent_value="add_item",
            local_slots=(),
            local_confidence=0.8,
            gpt_status=GptCallStatus.TIMEOUT,
        )
        assert fb.response_key == "repeat_side_options"
        assert fb.fallback_source == "state_clarification"
        assert fb.use_local is False

    def test_side_fallback_does_not_raise(self) -> None:
        # Even with a completely empty context, must not crash
        ctx = _ctx()
        fb = build_gpt_failure_fallback_response(
            state=_WAITING_SIDE,
            context=ctx,
            local_intent_value="unknown",
            local_slots=(),
            local_confidence=0.1,
            gpt_status=GptCallStatus.PROVIDER_ERROR,
        )
        assert isinstance(fb, GptFallbackResponse)
        assert fb.response_key  # non-empty


# ---------------------------------------------------------------------------
# Test 13 — WAITING_FOR_MODIFIER + GPT failure → repeats modifier options, no crash
# ---------------------------------------------------------------------------

class TestT13_WaitingForModifierGptFailure:
    def test_modifier_options_repeated_on_gpt_failure(self) -> None:
        fb = build_gpt_failure_fallback_response(
            state=_WAITING_MOD,
            context=_ctx(),
            local_intent_value="add_item",
            local_slots=(),
            local_confidence=0.8,
            gpt_status=GptCallStatus.INVALID_JSON,
        )
        assert fb.response_key == "repeat_modifier_options"
        assert fb.fallback_source == "state_clarification"

    def test_modifier_fallback_payload_safe(self) -> None:
        fb = build_gpt_failure_fallback_response(
            state=_WAITING_MOD,
            context=_ctx(),
            local_intent_value="add_item",
            local_slots=(),
            local_confidence=0.7,
            gpt_status=GptCallStatus.NETWORK_ERROR,
        )
        # Must have all required fields for repeat_modifier_options
        assert "group_name" in fb.response_payload
        assert "top_choices" in fb.response_payload


# ---------------------------------------------------------------------------
# Test 14 — CONFIRMING_ORDER + GPT failure → yes/no repeated, no payment trigger
# ---------------------------------------------------------------------------

class TestT14_ConfirmingOrderGptFailure:
    def test_confirm_order_repeat_on_gpt_failure(self) -> None:
        fb = build_gpt_failure_fallback_response(
            state=_CONFIRMING_ORDER,
            context=_ctx(),
            local_intent_value="confirm",
            local_slots=(),
            local_confidence=0.6,
            gpt_status=GptCallStatus.TIMEOUT,
        )
        assert fb.response_key == "confirm_order_repeat"
        assert fb.use_local is False


# ---------------------------------------------------------------------------
# Test 15 — Payment flow does not depend on GPT
# ---------------------------------------------------------------------------

class TestT15_PaymentNotDependentOnGpt:
    def test_payment_state_uses_local_on_gpt_failure(self) -> None:
        fb = build_gpt_failure_fallback_response(
            state=ConversationState.WAITING_FOR_PAYMENT,
            context=_ctx(),
            local_intent_value="payment_done",
            local_slots=(),
            local_confidence=0.9,
            gpt_status=GptCallStatus.CIRCUIT_OPEN,
        )
        assert fb.use_local is True
        assert fb.fallback_source == "local"


# ---------------------------------------------------------------------------
# Test 16 — Logs contain gpt_status + fallback_source for every GPT failure
# ---------------------------------------------------------------------------

class TestT16_LoggingFields:
    def test_decision_carries_gpt_status_on_timeout(self) -> None:
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        timeout_result = _gpt_error_result(BUCKET_IDLE_ITEM, "timeout")
        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=timeout_result,
        ):
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                circuit_breaker=_breaker(),
            )

        # gpt_status must be on the decision (logged by caller)
        assert decision.gpt_status in {
            GptCallStatus.TIMEOUT,
            GptCallStatus.UNKNOWN_ERROR,
        }
        assert isinstance(decision.local_is_safe, bool)

    def test_circuit_open_reflected_in_decision(self) -> None:
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()
        breaker = _breaker(threshold=1)
        key = breaker.circuit_key(cfg.model, BUCKET_IDLE_ITEM)
        breaker.record_failure(key)

        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            return_value=GptTurnResolution(
                bucket=BUCKET_IDLE_ITEM, decision="skipped", gpt_called=False
            ),
        ):
            decision = resolve(
                state=_IDLE, local_nlu=nlu, context=ctx, config=cfg,
                circuit_breaker=breaker,
            )

        assert decision.gpt_circuit_open is True
        assert decision.gpt_status == GptCallStatus.CIRCUIT_OPEN


# ---------------------------------------------------------------------------
# Test 17 — GPT exception does not escape resolve()
# ---------------------------------------------------------------------------

class TestT17_ExceptionDoesNotEscape:
    def test_gpt_call_raises_does_not_propagate(self) -> None:
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.2, text="large burger")
        cfg = _cfg(b0="inline")
        ctx = _ctx()

        with patch(
            "app.nlu.turn_resolver.final_turn_decision_resolver._call_gpt_for_bucket",
            side_effect=RuntimeError("catastrophic GPT failure"),
        ):
            # Must not raise
            decision = resolve(
                state=_IDLE,
                local_nlu=nlu,
                context=ctx,
                config=cfg,
                circuit_breaker=_breaker(),
            )

        assert decision.source == "local"
        assert "exception" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Test 18 — GPT exception does not escape call_gpt_safely
# ---------------------------------------------------------------------------

class TestT18_SafeClientNeverRaises:
    def test_any_exception_returns_safe_result(self) -> None:
        import os
        os.environ.setdefault("OPENAI_API_KEY", "test-key-for-isolation")

        for exc_type in [RuntimeError, ValueError, MemoryError, KeyError]:
            result = call_gpt_safely(
                gpt_callable=lambda: (_ for _ in ()).throw(exc_type("boom")),
                parse_fn=lambda x: x,
                task_mode="test",
                timeout_ms=700,
                model="gpt-4o-mini",
            )
            assert isinstance(result, GptSafeResult)
            assert result.ok is False
            assert result.should_fallback_to_local is True

    def test_parse_failure_returns_safe_result(self) -> None:
        import os
        os.environ.setdefault("OPENAI_API_KEY", "test-key-for-isolation")

        result = call_gpt_safely(
            gpt_callable=lambda: "NOT JSON AT ALL !!!",
            parse_fn=lambda x: (_ for _ in ()).throw(ValueError("bad json")),
            task_mode="test",
            timeout_ms=700,
            model="gpt-4o-mini",
        )
        assert result.ok is False
        assert result.status == GptCallStatus.INVALID_JSON
        assert result.raw_text is not None


# ---------------------------------------------------------------------------
# Test 19 — Waiting handlers: GPT failure does not leave inconsistent state
# ---------------------------------------------------------------------------

class TestT19_WaitingHandlerStateConsistency:
    def test_waiting_modifier_fallback_no_context_mutation(self) -> None:
        ctx = _ctx()
        original_modifier_groups = dict(ctx.selected_modifier_groups)

        fb = build_gpt_failure_fallback_response(
            state=_WAITING_MOD,
            context=ctx,
            local_intent_value="add_item",
            local_slots=(),
            local_confidence=0.5,
            gpt_status=GptCallStatus.TIMEOUT,
        )

        # Context must not be mutated
        assert dict(ctx.selected_modifier_groups) == original_modifier_groups
        assert fb.response_key == "repeat_modifier_options"

    def test_waiting_side_fallback_no_context_mutation(self) -> None:
        ctx = _ctx()
        original_selected = dict(ctx.selected_side_groups)

        build_gpt_failure_fallback_response(
            state=_WAITING_SIDE,
            context=ctx,
            local_intent_value="add_item",
            local_slots=(),
            local_confidence=0.5,
            gpt_status=GptCallStatus.PROVIDER_ERROR,
        )

        assert dict(ctx.selected_side_groups) == original_selected


# ---------------------------------------------------------------------------
# Test 20 — Existing deterministic suite still passes (sanity)
# ---------------------------------------------------------------------------

class TestT20_DeterministicSuiteUnaffected:
    def test_all_buckets_disabled_uses_local(self) -> None:
        """With all buckets disabled, every turn returns local."""
        nlu = _nlu(intent=Intent.UNKNOWN, confidence=0.1, text="large burger")
        cfg = SemanticRepairConfig(
            phase=2, model="gpt-4o-mini", timeout_seconds=0.35,
            bucket_0_mode="disabled",
            bucket_2_mode="disabled",
            bucket_3_mode="disabled",
        )
        ctx = _ctx()
        decision = resolve(state=_IDLE, local_nlu=nlu, context=ctx, config=cfg)

        assert decision.source == "local"
        assert decision.bucket is None
        assert decision.apply_gpt is False
        assert decision.gpt_status == GptCallStatus.DISABLED

    def test_local_safety_check_multi_item_slots_flagged(self) -> None:
        """Two ITEM slots always produce local_is_safe=False."""
        slots = (_slot("ITEM", "burger"), _slot("ITEM", "fries"))
        safe, reason = is_local_result_safe(
            state=_IDLE,
            local_intent_value="add_item",
            local_slots=slots,
            local_confidence=0.9,
        )
        assert safe is False
        assert reason == "multi_item_slots"

    def test_single_item_slot_is_safe(self) -> None:
        """Single ITEM slot in IDLE with high confidence is safe."""
        slots = (_slot("ITEM", "burger"),)
        safe, reason = is_local_result_safe(
            state=_IDLE,
            local_intent_value="add_item",
            local_slots=slots,
            local_confidence=0.9,
        )
        assert safe is True
        assert reason is None

    def test_api_key_sanitized_in_error_message(self) -> None:
        """API keys are never included in error_message."""
        import os
        os.environ["OPENAI_API_KEY"] = "sk-super-secret-key-12345"
        try:
            exc = RuntimeError("Call failed: Bearer sk-super-secret-key-12345")
            from app.nlu.turn_resolver.gpt_safe_client import _sanitize_error
            msg = _sanitize_error(exc)
            assert "sk-super-secret-key-12345" not in msg
            assert "[REDACTED]" in msg
        finally:
            os.environ.pop("OPENAI_API_KEY", None)
