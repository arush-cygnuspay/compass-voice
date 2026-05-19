# tests/nlu/turn_resolver/test_control_flow_resolver.py
"""Tests for Priority-6 control-flow GPT resolver.

Tests cover:
- Policy (CF-01 to CF-06): should_call_control_flow_gpt()
- Validator (CF-07 to CF-13): validate_control_flow_resolution()
- Resolver with mocked GPT (CF-14 to CF-22)
- Integration-style scenarios (CF-23 to CF-30)
- Config additions (CF-31 to CF-35)
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Imports ───────────────────────────────────────────────────────────────────

from app.nlu.turn_resolver.control_flow_policy import (
    should_call_control_flow_gpt,
    _is_low_confidence,
    _CHECKOUT_STATES,
    _PAYMENT_PERMISSION_STATES,
    _ORDER_TYPE_CHANGEABLE_STATES,
)
from app.nlu.turn_resolver.control_flow_resolver import (
    ControlFlowAction,
    ControlFlowResolution,
    ControlFlowResolver,
    CONTROL_FLOW_NOT_CALLED,
    _parse_json_response,
)
from app.nlu.turn_resolver.control_flow_validator import (
    ControlFlowValidationResult,
    validate_control_flow_resolution,
    VALIDATION_OK,
)
from app.nlu.turn_resolver.prompt_registry import (
    TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
    TASK_PAYMENT_PERMISSION_RESOLUTION,
    TASK_PICKUP_DELIVERY_INITIAL,
    TASK_ORDER_TYPE_CHANGE,
    PromptRegistry,
)
from app.nlu.turn_resolver.gpt_safe_client import GptCallStatus


# ── Shared stubs ──────────────────────────────────────────────────────────────

@dataclass
class _FakeCart:
    empty: bool = False

    def is_empty(self) -> bool:
        return self.empty


@dataclass
class _FakeAddr:
    collected: bool = False
    payment_link: str = ""
    payment_link_send_attempts: int = 0
    area_serviceable: bool | None = None


@dataclass
class _FakeContext:
    order_type: str = "pickup"
    pending_add_item: Any = None
    staged_item_queue: list = field(default_factory=list)
    pending_item_queue: list = field(default_factory=list)
    cart: Any = None
    delivery_address: Any = None
    delivery_address_required: bool = False
    delivery_available: bool = True
    payment_link_sent: bool = False
    order_submitted: bool = False


def _make_resolution(
    *,
    action: str = ControlFlowAction.FALLBACK,
    ok: bool = False,
    confidence: float = 0.85,
    requested_order_type: str | None = None,
    payment_preference: str | None = None,
    intent: str = "",
) -> ControlFlowResolution:
    return ControlFlowResolution(
        ok=ok,
        action=action,
        confidence=confidence,
        requested_order_type=requested_order_type,
        payment_preference=payment_preference,
        intent=intent,
        reason="test",
    )


def _make_gpt_result(parsed: dict | None, ok: bool = True, status: str = GptCallStatus.OK):
    r = MagicMock()
    r.ok = ok
    r.parsed = parsed
    r.status = status
    r.latency_ms = 42
    r.model = "gpt-4o-mini"
    r.error_message = None
    return r


# ─────────────────────────────────────────────────────────────────────────────
# CF-01 to CF-06  Policy tests
# ─────────────────────────────────────────────────────────────────────────────


class TestControlFlowPolicy:
    """CF-01 through CF-06: Policy trigger logic."""

    # CF-01
    def test_thats_it_in_idle_triggers_checkout(self):
        called, task_mode, reason = should_call_control_flow_gpt(
            state="idle",
            user_text="that's it",
            normalized_text="that's it",
            local_intent="UNKNOWN",
            local_confidence=0.30,
            previous_assistant_prompt=None,
        )
        assert called is True
        assert task_mode == TASK_CHECKOUT_CONFIRMATION_RESOLUTION
        assert reason == "checkout_phrase"

    # CF-02
    def test_yeah_do_it_in_confirming_order_triggers_checkout(self):
        called, task_mode, reason = should_call_control_flow_gpt(
            state="confirming_order",
            user_text="yeah do it",
            normalized_text="yeah do it",
            local_intent="AFFIRM",
            local_confidence=0.55,
            previous_assistant_prompt="Shall I place your order?",
        )
        assert called is True
        assert task_mode == TASK_CHECKOUT_CONFIRMATION_RESOLUTION

    # CF-03
    def test_no_payment_link_in_sms_permission_state_triggers_payment(self):
        called, task_mode, reason = should_call_control_flow_gpt(
            state="waiting_for_pickup_sms_permission",
            user_text="no payment link",
            normalized_text="no payment link",
            local_intent="UNKNOWN",
            local_confidence=0.20,
            previous_assistant_prompt="Would you like a payment link?",
        )
        assert called is True
        assert task_mode == TASK_PAYMENT_PERMISSION_RESOLUTION
        assert reason == "payment_phrase"

    # CF-04
    def test_make_it_delivery_triggers_order_type_change(self):
        called, task_mode, reason = should_call_control_flow_gpt(
            state="idle",
            user_text="make it delivery",
            normalized_text="make it delivery",
            local_intent="UNKNOWN",
            local_confidence=0.30,
            previous_assistant_prompt=None,
        )
        assert called is True
        assert task_mode == TASK_ORDER_TYPE_CHANGE
        assert reason == "order_type_phrase"

    # CF-05
    def test_ill_come_get_it_triggers_order_type_change(self):
        called, task_mode, reason = should_call_control_flow_gpt(
            state="confirming_order",
            user_text="I'll come get it",
            normalized_text="i'll come get it",
            local_intent="UNKNOWN",
            local_confidence=0.40,
            previous_assistant_prompt=None,
        )
        assert called is True
        assert task_mode == TASK_ORDER_TYPE_CHANGE
        assert reason == "order_type_phrase"

    # CF-06
    def test_terminal_state_does_not_trigger(self):
        for terminal_state in ("completed", "transferring_to_human_agent", "error_recovery"):
            called, task_mode, reason = should_call_control_flow_gpt(
                state=terminal_state,
                user_text="yeah do it",
                normalized_text="yeah do it",
                local_intent="AFFIRM",
                local_confidence=0.90,
                previous_assistant_prompt=None,
            )
            assert called is False, f"Expected no trigger in state={terminal_state!r}"
            assert task_mode == ""
            assert reason == "terminal_state"

    def test_empty_text_does_not_trigger(self):
        called, _, reason = should_call_control_flow_gpt(
            state="idle",
            user_text="",
            normalized_text="",
            local_intent=None,
            local_confidence=None,
            previous_assistant_prompt=None,
        )
        assert called is False
        assert reason == "empty_text"

    def test_waiting_for_order_type_initial_phrase(self):
        called, task_mode, reason = should_call_control_flow_gpt(
            state="waiting_for_order_type",
            user_text="pickup",
            normalized_text="pickup",
            local_intent=None,
            local_confidence=None,
            previous_assistant_prompt=None,
        )
        assert called is True
        assert task_mode == TASK_PICKUP_DELIVERY_INITIAL
        assert reason == "initial_order_type_phrase"

    def test_send_the_link_triggers_payment_permission(self):
        called, task_mode, reason = should_call_control_flow_gpt(
            state="waiting_for_pickup_sms_permission",
            user_text="send the link",
            normalized_text="send the link",
            local_intent="UNKNOWN",
            local_confidence=0.20,
            previous_assistant_prompt=None,
        )
        assert called is True
        assert task_mode == TASK_PAYMENT_PERMISSION_RESOLUTION

    def test_low_confidence_in_confirming_order_triggers(self):
        called, task_mode, reason = should_call_control_flow_gpt(
            state="confirming_order",
            user_text="um yeah",
            normalized_text="um yeah",
            local_intent="UNKNOWN",
            local_confidence=0.30,
            previous_assistant_prompt="Shall I place your order?",
        )
        assert called is True
        assert task_mode == TASK_CHECKOUT_CONFIRMATION_RESOLUTION
        assert reason == "low_confidence_confirming"

    def test_delivery_phrase_in_waiting_modifier_triggers_order_type(self):
        """Order type change phrase fires even during a waiting-for-modifier turn."""
        called, task_mode, reason = should_call_control_flow_gpt(
            state="waiting_for_modifier",
            user_text="actually make it delivery",
            normalized_text="actually make it delivery",
            local_intent="UNKNOWN",
            local_confidence=0.20,
            previous_assistant_prompt=None,
        )
        assert called is True
        assert task_mode == TASK_ORDER_TYPE_CHANGE


# ─────────────────────────────────────────────────────────────────────────────
# CF-07 to CF-13  Validator tests
# ─────────────────────────────────────────────────────────────────────────────


class TestControlFlowValidator:
    """CF-07 through CF-13: Validation rules."""

    # CF-07
    def test_checkout_blocked_with_pending_add_item(self):
        ctx = _FakeContext()
        ctx.pending_add_item = MagicMock()  # non-None pending item
        ctx.cart = _FakeCart(empty=False)

        # Patch at the source location — validator imports it inside a function.
        from app.services.order_lifecycle_guard import LifecycleDecision, LifecycleCode
        block_decision = LifecycleDecision(
            code=LifecycleCode.CART_INCOMPLETE,
            blocking=True,
            response="I still need to finish adding your item.",
        )
        resolution = _make_resolution(action=ControlFlowAction.CONFIRM_CHECKOUT, ok=True, confidence=0.90)

        with patch(
            "app.services.order_lifecycle_guard.can_checkout",
            return_value=block_decision,
        ):
            result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert result.is_valid is False
        assert "lifecycle_blocked" in result.reason

    # CF-08
    def test_checkout_blocked_with_staged_item_queue(self):
        ctx = _FakeContext()
        ctx.staged_item_queue = ["item1", "item2"]  # non-empty
        ctx.cart = _FakeCart(empty=False)

        from app.services.order_lifecycle_guard import LifecycleDecision, LifecycleCode
        block_decision = LifecycleDecision(
            code=LifecycleCode.CART_INCOMPLETE,
            blocking=True,
            response="I still need to finish the remaining items first.",
        )
        resolution = _make_resolution(action=ControlFlowAction.CONFIRM_CHECKOUT, ok=True, confidence=0.90)

        with patch(
            "app.services.order_lifecycle_guard.can_checkout",
            return_value=block_decision,
        ):
            result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert result.is_valid is False

    # CF-09
    def test_checkout_blocked_with_empty_cart(self):
        ctx = _FakeContext()
        ctx.cart = _FakeCart(empty=True)

        from app.services.order_lifecycle_guard import LifecycleDecision, LifecycleCode
        block_decision = LifecycleDecision(
            code=LifecycleCode.CART_EMPTY,
            blocking=True,
            response="Your cart is empty.",
        )
        resolution = _make_resolution(action=ControlFlowAction.CONFIRM_CHECKOUT, ok=True, confidence=0.90)

        with patch(
            "app.services.order_lifecycle_guard.can_checkout",
            return_value=block_decision,
        ):
            result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert result.is_valid is False
        assert "cart_empty" in result.reason.lower()

    # CF-10
    def test_confirm_payment_link_invalid_outside_payment_state(self):
        ctx = _FakeContext()
        resolution = _make_resolution(
            action=ControlFlowAction.CONFIRM_PAYMENT_LINK, ok=True, confidence=0.85
        )
        # "idle" is NOT a payment permission state
        result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert result.is_valid is False
        assert "invalid_state_for_payment_permission" in result.reason

    def test_confirm_payment_link_valid_in_payment_state(self):
        ctx = _FakeContext()
        resolution = _make_resolution(
            action=ControlFlowAction.CONFIRM_PAYMENT_LINK, ok=True, confidence=0.85
        )
        result = validate_control_flow_resolution(
            resolution, ctx, "waiting_for_pickup_sms_permission"
        )
        assert result.is_valid is True

    # CF-11
    def test_order_type_change_blocked_after_order_submitted(self):
        ctx = _FakeContext()
        ctx.order_submitted = True
        resolution = _make_resolution(
            action=ControlFlowAction.CHANGE_ORDER_TYPE,
            ok=True,
            confidence=0.90,
            requested_order_type="delivery",
        )
        result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert result.is_valid is False
        assert "submitted" in result.reason

    def test_order_type_change_blocked_after_payment_sent(self):
        ctx = _FakeContext()
        ctx.payment_link_sent = True
        resolution = _make_resolution(
            action=ControlFlowAction.CHANGE_ORDER_TYPE,
            ok=True,
            confidence=0.90,
            requested_order_type="pickup",
        )
        result = validate_control_flow_resolution(resolution, ctx, "confirming_order")
        assert result.is_valid is False
        assert "payment_sent" in result.reason

    # CF-12
    def test_delivery_selection_valid_with_address_required(self):
        """Delivery change with address required passes validator (handler routes to address flow)."""
        ctx = _FakeContext()
        resolution = _make_resolution(
            action=ControlFlowAction.CHANGE_ORDER_TYPE,
            ok=True,
            confidence=0.90,
            requested_order_type="delivery",
        )
        result = validate_control_flow_resolution(resolution, ctx, "idle")
        # Validator passes — OrderTypeService handles the address routing.
        assert result.is_valid is True

    # CF-13
    def test_low_confidence_blocked(self):
        ctx = _FakeContext()
        resolution = _make_resolution(
            action=ControlFlowAction.CONFIRM_CHECKOUT,
            ok=True,
            confidence=0.50,  # below threshold
        )
        # Low confidence check fires before lifecycle guard, no patch needed.
        result = validate_control_flow_resolution(
            resolution, ctx, "idle", min_confidence=0.70
        )
        assert result.is_valid is False
        assert result.reason == "low_confidence"

    def test_clarify_action_always_valid(self):
        ctx = _FakeContext()
        resolution = _make_resolution(action=ControlFlowAction.CLARIFY, ok=False, confidence=0.20)
        result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert result.is_valid is True

    def test_fallback_action_always_valid(self):
        ctx = _FakeContext()
        resolution = _make_resolution(action=ControlFlowAction.FALLBACK, ok=False, confidence=0.10)
        result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert result.is_valid is True

    def test_invalid_order_type_rejected(self):
        ctx = _FakeContext()
        resolution = _make_resolution(
            action=ControlFlowAction.CHANGE_ORDER_TYPE,
            ok=True,
            confidence=0.90,
            requested_order_type="drive_through",  # invalid
        )
        result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert result.is_valid is False
        assert "invalid_requested_order_type" in result.reason

    def test_order_type_blocked_in_payment_state(self):
        ctx = _FakeContext()
        resolution = _make_resolution(
            action=ControlFlowAction.CHANGE_ORDER_TYPE,
            ok=True,
            confidence=0.90,
            requested_order_type="pickup",
        )
        result = validate_control_flow_resolution(
            resolution, ctx, "waiting_for_payment"
        )
        assert result.is_valid is False
        assert "payment_state" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# CF-14 to CF-22  Resolver with mocked GPT
# ─────────────────────────────────────────────────────────────────────────────


class TestControlFlowResolverWithMockedGPT:
    """CF-14 through CF-22: Resolver with injected mock GptSafeClient."""

    def _make_resolver(self, gpt_parsed: dict | None, gpt_ok: bool = True, mode: str = "inline"):
        """Build a resolver with a mocked GptSafeClient and config."""
        cfg = MagicMock()
        cfg.model = "gpt-4o-mini"
        cfg.bucket_5_mode = mode
        cfg.bucket_6_mode = mode
        cfg.bucket_payment_mode = mode
        cfg.control_flow_timeout_ms = 700
        cfg.control_flow_min_confidence = 0.70

        mock_client = MagicMock()
        gpt_result = _make_gpt_result(gpt_parsed, ok=gpt_ok)
        mock_client.call = AsyncMock(return_value=gpt_result)

        ctx_builder = MagicMock()
        ctx_builder.build = MagicMock(return_value={
            "current_state": "idle",
            "user_text": "test",
            "normalized_text": "test",
            "local_intent": "UNKNOWN",
            "local_confidence": 0.30,
            "previous_turns": [],
            "previous_assistant_prompt": None,
        })

        return ControlFlowResolver(
            gpt_client=mock_client,
            context_builder=ctx_builder,
            config=cfg,
        )

    def _resolve_async(self, resolver, **kwargs):
        return asyncio.get_event_loop().run_until_complete(resolver.resolve(**kwargs))

    def _ctx(self):
        return _FakeContext()

    # CF-14
    def test_thats_it_returns_request_checkout(self):
        resolver = self._make_resolver({
            "action": "request_checkout",
            "intent": "checkout",
            "confidence": 0.88,
            "reason": "cart is non-empty, customer says done",
        })
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="that's it",
            normalized_text="that's it",
            task_mode=TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
            state="idle",
        ))
        assert result.action == ControlFlowAction.REQUEST_CHECKOUT
        assert result.ok is True
        assert result.confidence >= 0.80

    # CF-15
    def test_yeah_do_it_after_order_summary_returns_confirm_checkout(self):
        resolver = self._make_resolver({
            "action": "confirm_checkout",
            "intent": "confirm_order",
            "confidence": 0.92,
            "reason": "customer affirmed order",
        })
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="yeah do it",
            normalized_text="yeah do it",
            task_mode=TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
            state="confirming_order",
        ))
        assert result.action == ControlFlowAction.CONFIRM_CHECKOUT
        assert result.ok is True

    # CF-16
    def test_yeah_do_it_after_payment_prompt_returns_confirm_payment_link(self):
        resolver = self._make_resolver({
            "action": "confirm_payment_link",
            "intent": "payment_link",
            "payment_preference": "send_link",
            "confidence": 0.90,
            "reason": "customer said yes after payment-link prompt",
        })
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="yeah do it",
            normalized_text="yeah do it",
            task_mode=TASK_PAYMENT_PERMISSION_RESOLUTION,
            state="waiting_for_pickup_sms_permission",
        ))
        assert result.action == ControlFlowAction.CONFIRM_PAYMENT_LINK
        assert result.payment_preference == "send_link"
        assert result.ok is True

    # CF-17
    def test_no_payment_link_returns_deny_payment_link(self):
        resolver = self._make_resolver({
            "action": "deny_payment_link",
            "intent": "no_payment_link",
            "payment_preference": "pay_on_arrival",
            "confidence": 0.91,
            "reason": "customer explicitly declined payment link",
        })
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="no payment link",
            normalized_text="no payment link",
            task_mode=TASK_PAYMENT_PERMISSION_RESOLUTION,
            state="waiting_for_pickup_sms_permission",
        ))
        assert result.action == ControlFlowAction.DENY_PAYMENT_LINK
        assert result.payment_preference == "pay_on_arrival"
        assert result.ok is True

    # CF-18
    def test_make_it_delivery_returns_delivery_order_type(self):
        resolver = self._make_resolver({
            "action": "change_order_type",
            "intent": "change_order_type",
            "requested_order_type": "delivery",
            "confidence": 0.93,
            "reason": "customer said make it delivery",
        })
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="make it delivery",
            normalized_text="make it delivery",
            task_mode=TASK_ORDER_TYPE_CHANGE,
            state="idle",
        ))
        assert result.action == ControlFlowAction.CHANGE_ORDER_TYPE
        assert result.requested_order_type == "delivery"
        assert result.ok is True

    # CF-19
    def test_ill_come_get_it_returns_pickup(self):
        resolver = self._make_resolver({
            "action": "change_order_type",
            "intent": "change_to_pickup",
            "requested_order_type": "pickup",
            "confidence": 0.91,
            "reason": "customer said I'll come get it",
        })
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="I'll come get it",
            normalized_text="i'll come get it",
            task_mode=TASK_ORDER_TYPE_CHANGE,
            state="idle",
        ))
        assert result.action == ControlFlowAction.CHANGE_ORDER_TYPE
        assert result.requested_order_type == "pickup"
        assert result.ok is True

    # CF-20
    def test_gpt_timeout_returns_deterministic_fallback(self):
        resolver = self._make_resolver(None, gpt_ok=False)
        # Simulate a failing GPT client (ok=False, parsed=None)
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="yeah do it",
            normalized_text="yeah do it",
            task_mode=TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
            state="confirming_order",
        ))
        assert result.ok is False
        assert result.action == ControlFlowAction.FALLBACK
        assert "gpt_failed" in result.reason

    # CF-21
    def test_gpt_invalid_json_returns_fallback(self):
        cfg = MagicMock()
        cfg.model = "gpt-4o-mini"
        cfg.bucket_5_mode = "inline"
        cfg.bucket_6_mode = "inline"
        cfg.bucket_payment_mode = "inline"
        cfg.control_flow_timeout_ms = 700
        cfg.control_flow_min_confidence = 0.70

        # Return invalid JSON → parse_fn will raise → client returns ok=False
        bad_result = MagicMock()
        bad_result.ok = False
        bad_result.parsed = None
        bad_result.status = GptCallStatus.INVALID_JSON
        bad_result.latency_ms = 10
        bad_result.model = "gpt-4o-mini"
        bad_result.error_message = "JSON parse error"

        mock_client = MagicMock()
        mock_client.call = AsyncMock(return_value=bad_result)

        ctx_builder = MagicMock()
        ctx_builder.build = MagicMock(return_value={
            "current_state": "idle",
            "user_text": "yeah",
            "normalized_text": "yeah",
            "local_intent": "UNKNOWN",
            "local_confidence": 0.20,
            "previous_turns": [],
            "previous_assistant_prompt": None,
        })

        resolver = ControlFlowResolver(
            gpt_client=mock_client,
            context_builder=ctx_builder,
            config=cfg,
        )
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=_FakeContext(),
            user_text="yeah",
            normalized_text="yeah",
            task_mode=TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
            state="confirming_order",
        ))
        assert result.ok is False
        assert result.action == ControlFlowAction.FALLBACK

    # CF-22
    def test_gpt_clarify_action_returns_not_ok(self):
        """clarify action from GPT should have ok=False (control action)."""
        resolver = self._make_resolver({
            "action": "clarify",
            "intent": "unknown",
            "confidence": 0.55,
            "clarification_text": "Did you mean yes or no?",
            "reason": "ambiguous",
        })
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="um",
            normalized_text="um",
            task_mode=TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
            state="confirming_order",
        ))
        assert result.action == ControlFlowAction.CLARIFY
        assert result.ok is False  # clarify is a control action — not "applied"
        assert result.clarification_text == "Did you mean yes or no?"

    def test_shadow_mode_does_not_apply(self):
        """shadow mode: GPT runs but ok=False is returned."""
        resolver = self._make_resolver({
            "action": "confirm_checkout",
            "intent": "checkout",
            "confidence": 0.90,
            "reason": "customer confirmed",
        }, mode="shadow")
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="yeah do it",
            normalized_text="yeah do it",
            task_mode=TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
            state="confirming_order",
        ))
        assert result.ok is False
        assert result.reason == "shadow_mode_not_applied"

    def test_disabled_mode_returns_sentinel(self):
        """disabled mode: resolver returns CONTROL_FLOW_NOT_CALLED immediately."""
        resolver = self._make_resolver({}, mode="disabled")
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=self._ctx(),
            user_text="yeah do it",
            normalized_text="yeah do it",
            task_mode=TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
            state="confirming_order",
        ))
        assert result is CONTROL_FLOW_NOT_CALLED or (
            result.ok is False and result.reason == "not_called"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CF-23 to CF-30  Integration-style tests
# ─────────────────────────────────────────────────────────────────────────────


class TestControlFlowIntegration:
    """CF-23 through CF-30: End-to-end scenario validation."""

    # CF-23
    def test_idle_cart_thats_it_starts_checkout(self):
        """idle + non-empty cart + 'that's it' → policy fires, validator passes."""
        ctx = _FakeContext()
        ctx.cart = _FakeCart(empty=False)

        called, task_mode, reason = should_call_control_flow_gpt(
            state="idle",
            user_text="that's it",
            normalized_text="that's it",
            local_intent="UNKNOWN",
            local_confidence=0.20,
            previous_assistant_prompt=None,
        )
        assert called is True
        assert task_mode == TASK_CHECKOUT_CONFIRMATION_RESOLUTION

        # Validator should pass if cart is non-empty (lifecycle guard says OK)
        resolution = _make_resolution(
            action=ControlFlowAction.REQUEST_CHECKOUT, ok=True, confidence=0.85
        )
        from app.services.order_lifecycle_guard import LifecycleDecision, LifecycleCode
        ok_decision = LifecycleDecision(code=LifecycleCode.OK, blocking=False, response="")
        with patch(
            "app.services.order_lifecycle_guard.can_checkout",
            return_value=ok_decision,
        ):
            val_result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert val_result.is_valid is True

    # CF-24
    def test_waiting_for_modifier_thats_it_blocks_checkout_via_lifecycle(self):
        """waiting_for_modifier + 'that's it' → lifecycle guard blocks checkout."""
        ctx = _FakeContext()
        ctx.pending_add_item = MagicMock()
        ctx.cart = _FakeCart(empty=False)

        resolution = _make_resolution(
            action=ControlFlowAction.CONFIRM_CHECKOUT, ok=True, confidence=0.85
        )
        from app.services.order_lifecycle_guard import LifecycleDecision, LifecycleCode
        blocked = LifecycleDecision(
            code=LifecycleCode.MODIFIER_REQUIRED,
            blocking=True,
            response="I still need your sauce choice.",
        )
        with patch(
            "app.services.order_lifecycle_guard.can_checkout",
            return_value=blocked,
        ):
            val_result = validate_control_flow_resolution(
                resolution, ctx, "waiting_for_modifier"
            )
        assert val_result.is_valid is False
        assert "modifier_required" in val_result.reason.lower()

    # CF-25
    def test_confirming_order_yeah_do_it_validates(self):
        """confirming_order + 'yeah do it' → confirm_checkout passes validator."""
        ctx = _FakeContext()
        ctx.cart = _FakeCart(empty=False)

        resolution = _make_resolution(
            action=ControlFlowAction.CONFIRM_CHECKOUT, ok=True, confidence=0.92
        )
        from app.services.order_lifecycle_guard import LifecycleDecision, LifecycleCode
        ok_decision = LifecycleDecision(code=LifecycleCode.OK, blocking=False, response="")
        with patch(
            "app.services.order_lifecycle_guard.can_checkout",
            return_value=ok_decision,
        ):
            val_result = validate_control_flow_resolution(
                resolution, ctx, "confirming_order"
            )
        assert val_result.is_valid is True

    # CF-26
    def test_sms_permission_no_payment_link_validates(self):
        """waiting_for_pickup_sms_permission + 'no payment link' → deny_payment_link valid."""
        ctx = _FakeContext()
        resolution = _make_resolution(
            action=ControlFlowAction.DENY_PAYMENT_LINK,
            ok=True,
            confidence=0.91,
            payment_preference="pay_on_arrival",
        )
        val_result = validate_control_flow_resolution(
            resolution, ctx, "waiting_for_pickup_sms_permission"
        )
        assert val_result.is_valid is True

    # CF-27
    def test_active_order_make_it_delivery_validates(self):
        """active order + 'make it delivery' → change_order_type passes validator."""
        ctx = _FakeContext()
        ctx.order_submitted = False
        ctx.payment_link_sent = False
        resolution = _make_resolution(
            action=ControlFlowAction.CHANGE_ORDER_TYPE,
            ok=True,
            confidence=0.90,
            requested_order_type="delivery",
        )
        val_result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert val_result.is_valid is True

    # CF-28
    def test_ill_come_get_it_from_delivery_address_state_validates(self):
        """'I'll come get it' while in delivery address state → pickup change valid."""
        ctx = _FakeContext()
        ctx.order_submitted = False
        ctx.payment_link_sent = False
        resolution = _make_resolution(
            action=ControlFlowAction.CHANGE_ORDER_TYPE,
            ok=True,
            confidence=0.91,
            requested_order_type="pickup",
        )
        val_result = validate_control_flow_resolution(
            resolution, ctx, "waiting_for_delivery_address_collection"
        )
        # Validator passes — handler routes to pickup and exits address flow.
        assert val_result.is_valid is True

    # CF-29
    def test_yeah_do_it_in_idle_no_pending_action_is_clarify(self):
        """'yeah do it' in idle with no context resolves as clarify — not ok."""
        resolution = _make_resolution(
            action=ControlFlowAction.CLARIFY, ok=False, confidence=0.60,
        )
        ctx = _FakeContext()
        val_result = validate_control_flow_resolution(resolution, ctx, "idle")
        assert val_result.is_valid is True  # clarify is structurally valid
        assert resolution.ok is False        # but not applied

    # CF-30
    def test_gpt_failure_returns_fallback_and_does_not_crash(self):
        """GPT failure (exception in resolve) must return fallback, never raise."""
        cfg = MagicMock()
        cfg.model = "gpt-4o-mini"
        cfg.bucket_5_mode = "inline"
        cfg.bucket_6_mode = "inline"
        cfg.bucket_payment_mode = "inline"
        cfg.control_flow_timeout_ms = 700
        cfg.control_flow_min_confidence = 0.70

        mock_client = MagicMock()
        mock_client.call = AsyncMock(side_effect=RuntimeError("Connection refused"))

        ctx_builder = MagicMock()
        ctx_builder.build = MagicMock(return_value={
            "current_state": "confirming_order",
            "user_text": "yeah do it",
            "normalized_text": "yeah do it",
            "local_intent": "AFFIRM",
            "local_confidence": 0.55,
            "previous_turns": [],
            "previous_assistant_prompt": None,
        })

        resolver = ControlFlowResolver(
            gpt_client=mock_client,
            context_builder=ctx_builder,
            config=cfg,
        )
        # Must not raise
        result = asyncio.get_event_loop().run_until_complete(resolver.resolve(
            context=_FakeContext(),
            user_text="yeah do it",
            normalized_text="yeah do it",
            task_mode=TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
            state="confirming_order",
        ))
        assert result.ok is False
        assert result.action == ControlFlowAction.FALLBACK
        # No secrets / API key data in fallback result
        assert "api_key" not in (result.reason or "").lower()
        assert "api_key" not in str(result.metadata).lower()


# ─────────────────────────────────────────────────────────────────────────────
# CF-31 to CF-35  Config addition tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigAdditions:
    """CF-31 through CF-35: New SemanticRepairConfig fields."""

    def _make_config(self, **overrides):
        """Build a SemanticRepairConfig with defaults and optional overrides."""
        from app.config.semantic_repair import SemanticRepairConfig
        defaults = dict(
            phase=0,
            model="gpt-4o-mini",
            timeout_seconds=0.35,
        )
        defaults.update(overrides)
        return SemanticRepairConfig(**defaults)

    # CF-31
    def test_bucket_5_mode_default_disabled(self):
        cfg = self._make_config()
        assert cfg.bucket_5_mode == "disabled"

    # CF-32
    def test_bucket_6_mode_default_disabled(self):
        cfg = self._make_config()
        assert cfg.bucket_6_mode == "disabled"

    # CF-33
    def test_bucket_payment_mode_default_disabled(self):
        cfg = self._make_config()
        assert cfg.bucket_payment_mode == "disabled"

    # CF-34
    def test_control_flow_timeout_ms_default(self):
        cfg = self._make_config()
        assert cfg.control_flow_timeout_ms == 700

    # CF-35
    def test_control_flow_min_confidence_default(self):
        cfg = self._make_config()
        assert cfg.control_flow_min_confidence == 0.70

    def test_invalid_bucket_5_mode_raises(self):
        from app.config.semantic_repair import SemanticRepairConfig
        with pytest.raises(ValueError, match="bucket_5_mode"):
            SemanticRepairConfig(
                phase=0,
                model="gpt-4o-mini",
                timeout_seconds=0.35,
                bucket_5_mode="invalid",
            )

    def test_invalid_bucket_6_mode_raises(self):
        from app.config.semantic_repair import SemanticRepairConfig
        with pytest.raises(ValueError, match="bucket_6_mode"):
            SemanticRepairConfig(
                phase=0,
                model="gpt-4o-mini",
                timeout_seconds=0.35,
                bucket_6_mode="apply",  # "apply" not allowed, must be "inline"
            )

    def test_invalid_bucket_payment_mode_raises(self):
        from app.config.semantic_repair import SemanticRepairConfig
        with pytest.raises(ValueError, match="bucket_payment_mode"):
            SemanticRepairConfig(
                phase=0,
                model="gpt-4o-mini",
                timeout_seconds=0.35,
                bucket_payment_mode="yes",
            )

    def test_inline_modes_accepted(self):
        cfg = self._make_config(
            bucket_5_mode="inline",
            bucket_6_mode="shadow",
            bucket_payment_mode="inline",
        )
        assert cfg.bucket_5_mode == "inline"
        assert cfg.bucket_6_mode == "shadow"
        assert cfg.bucket_payment_mode == "inline"

    def test_env_override_bucket_5_mode(self):
        """Env var COMPASS_GPT_BUCKET_5_CHECKOUT_MODE overrides default."""
        import os
        from app.config.semantic_repair import get_semantic_repair_config
        # Clear the lru_cache before patching env
        get_semantic_repair_config.cache_clear()
        with patch.dict(os.environ, {"COMPASS_GPT_BUCKET_5_CHECKOUT_MODE": "shadow"}):
            get_semantic_repair_config.cache_clear()
            cfg = get_semantic_repair_config()
            assert cfg.bucket_5_mode == "shadow"
        get_semantic_repair_config.cache_clear()  # restore


# ─────────────────────────────────────────────────────────────────────────────
# Additional resolution parsing tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolutionParsing:
    """Resolver._parse_resolution handles legacy and new field formats."""

    def _parse(self, data: dict, task_mode: str = TASK_CHECKOUT_CONFIRMATION_RESOLUTION) -> ControlFlowResolution:
        resolver = ControlFlowResolver()
        return resolver._parse_resolution(data, task_mode, "test utterance")

    def test_action_field_used_directly(self):
        r = self._parse({"action": "confirm_checkout", "confidence": 0.90, "intent": "checkout"})
        assert r.action == ControlFlowAction.CONFIRM_CHECKOUT

    def test_legacy_affirm_mapped_to_confirm_checkout(self):
        r = self._parse({"decision": "affirm", "confidence": 0.88})
        assert r.action == ControlFlowAction.CONFIRM_CHECKOUT

    def test_legacy_deny_mapped_to_deny_checkout(self):
        r = self._parse({"decision": "deny", "confidence": 0.88})
        assert r.action == ControlFlowAction.DENY_CHECKOUT

    def test_legacy_sms_link_maps_to_confirm_payment_link(self):
        r = self._parse(
            {"action": "confirm_payment_link", "decision": "sms_link", "confidence": 0.90},
            task_mode=TASK_PAYMENT_PERMISSION_RESOLUTION,
        )
        assert r.action == ControlFlowAction.CONFIRM_PAYMENT_LINK

    def test_requested_order_type_captured(self):
        r = self._parse(
            {"action": "change_order_type", "requested_order_type": "delivery", "confidence": 0.90},
            task_mode=TASK_ORDER_TYPE_CHANGE,
        )
        assert r.requested_order_type == "delivery"

    def test_pickup_decision_captures_order_type(self):
        """Legacy GPT returning decision=pickup for initial selection."""
        r = self._parse(
            {"decision": "pickup", "confidence": 0.88},
            task_mode=TASK_PICKUP_DELIVERY_INITIAL,
        )
        assert r.requested_order_type == "pickup"

    def test_unknown_action_falls_back(self):
        r = self._parse({"action": "totally_unknown", "confidence": 0.80})
        assert r.action == ControlFlowAction.FALLBACK

    def test_clarify_action_ok_is_false(self):
        r = self._parse({"action": "clarify", "confidence": 0.50, "clarification_text": "Yes or no?"})
        assert r.ok is False
        assert r.clarification_text == "Yes or no?"

    def test_pay_on_arrival_action(self):
        r = self._parse(
            {"action": "pay_on_arrival", "payment_preference": "pay_on_arrival", "confidence": 0.88},
            task_mode=TASK_PAYMENT_PERMISSION_RESOLUTION,
        )
        assert r.action == ControlFlowAction.PAY_ON_ARRIVAL
        assert r.payment_preference == "pay_on_arrival"

    def test_json_parse_helper_strips_markdown(self):
        raw = '```json\n{"action": "confirm_checkout", "confidence": 0.9}\n```'
        result = _parse_json_response(raw)
        assert result["action"] == "confirm_checkout"

    def test_prompt_registry_has_correct_task_modes(self):
        registry = PromptRegistry()
        for mode in (
            TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
            TASK_PAYMENT_PERMISSION_RESOLUTION,
            TASK_PICKUP_DELIVERY_INITIAL,
            TASK_ORDER_TYPE_CHANGE,
        ):
            assert registry.is_known_task_mode(mode), f"{mode} not in registry"
            assert len(registry.get_system_prompt(mode)) > 20
            assert len(registry.get_task_instructions(mode)) > 20
            assert len(registry.get_output_contract(mode)) > 20
