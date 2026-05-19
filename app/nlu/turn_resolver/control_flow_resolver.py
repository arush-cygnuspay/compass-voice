# app/nlu/turn_resolver/control_flow_resolver.py
"""Priority-6 control-flow GPT resolver for Compass Voice.

Resolves contextually ambiguous checkout / payment / order-type phrases
using GPT as a secondary layer after deterministic NLU.

Covered task modes
------------------
checkout_confirmation_resolution  — "that's it", "yeah do it", "keep ordering"
payment_permission_resolution     — "no payment link", "send the link", "pay there"
pickup_delivery_initial           — "I'll come get it", "delivery", "pickup"
order_type_change                 — "make it delivery", "switch to pickup"

Safety invariants
-----------------
* GPT never directly mutates cart, order type, or payment state.
* All GPT results must pass validate_control_flow_resolution() before use.
* GPT failure always returns ControlFlowResolution(ok=False, action='fallback').
* No full menu, PII, or payment data is ever sent to GPT.
* Mode=shadow  → GPT is called and logged, but ok=False is returned.
* Mode=disabled → CONTROL_FLOW_NOT_CALLED sentinel is returned immediately.
* This module never raises into callers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from app.nlu.turn_resolver.gpt_safe_client import GptCallStatus, GptSafeClient
from app.nlu.turn_resolver.gpt_circuit_breaker import DEFAULT_CIRCUIT_BREAKER
from app.nlu.turn_resolver.prompt_registry import (
    TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
    TASK_PAYMENT_PERMISSION_RESOLUTION,
    TASK_PICKUP_DELIVERY_INITIAL,
    TASK_ORDER_TYPE_CHANGE,
    PromptRegistry,
)

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig
    from app.nlu.turn_resolver.gpt_context_builder import GptContextBuilder
    from app.state_machine.models.conversation_context import ConversationContext

_logger = logging.getLogger(__name__)

# Thread pool for sync → async bridge.
_SYNC_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="control_flow_gpt",
)

# Task modes handled by this resolver.
_CONTROL_FLOW_TASK_MODES: frozenset[str] = frozenset({
    TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
    TASK_PAYMENT_PERMISSION_RESOLUTION,
    TASK_PICKUP_DELIVERY_INITIAL,
    TASK_ORDER_TYPE_CHANGE,
})

# Valid action values from GPT.
_VALID_ACTIONS: frozenset[str] = frozenset({
    "confirm_checkout",
    "deny_checkout",
    "request_checkout",
    "confirm_payment_link",
    "deny_payment_link",
    "pay_on_arrival",
    "change_order_type",
    "select_initial_order_type",
    "cancel_order",
    "cancel_pending",
    "continue_ordering",
    "clarify",
    "fallback",
})

# Aliases for legacy/alternative GPT output values.
_ACTION_ALIASES: dict[str, str] = {
    # checkout
    "affirm": "confirm_checkout",
    "confirm": "confirm_checkout",
    "deny": "deny_checkout",
    "reject": "deny_checkout",
    # payment
    "sms_link": "confirm_payment_link",
    "send_link": "confirm_payment_link",
    "pay_on_arrival": "pay_on_arrival",
    # order type
    "pickup": "select_initial_order_type",
    "delivery": "select_initial_order_type",
    "no_change": "continue_ordering",
    # generic
    "no_match": "fallback",
    "clarify": "clarify",
    "fallback": "fallback",
    "cancel": "cancel_order",
    "cancel_pending": "cancel_pending",
    "continue_ordering": "continue_ordering",
    "keep_ordering": "continue_ordering",
}

# Valid order types and payment preferences.
_VALID_ORDER_TYPES: frozenset[str] = frozenset({"pickup", "delivery"})
_VALID_PAYMENT_PREFS: frozenset[str] = frozenset({"send_link", "pay_on_arrival"})


# ── Action constants ──────────────────────────────────────────────────────────


class ControlFlowAction:
    """String constants for ControlFlowResolution.action."""

    CONFIRM_CHECKOUT = "confirm_checkout"
    DENY_CHECKOUT = "deny_checkout"
    REQUEST_CHECKOUT = "request_checkout"
    CONFIRM_PAYMENT_LINK = "confirm_payment_link"
    DENY_PAYMENT_LINK = "deny_payment_link"
    PAY_ON_ARRIVAL = "pay_on_arrival"
    CHANGE_ORDER_TYPE = "change_order_type"
    SELECT_INITIAL_ORDER_TYPE = "select_initial_order_type"
    CANCEL_ORDER = "cancel_order"
    CANCEL_PENDING = "cancel_pending"
    CONTINUE_ORDERING = "continue_ordering"
    CLARIFY = "clarify"
    FALLBACK = "fallback"


# ── Resolution dataclass ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ControlFlowResolution:
    """Structured result from a control-flow GPT bucket call.

    ok=True means the action is safe to route to a deterministic handler.
    ok=False means fall through to local deterministic path.

    Fields
    ------
    ok:
        True when resolution is structurally valid and may be acted on.
        Always False in shadow mode and on GPT failure.
    action:
        One of ControlFlowAction.* constants.
    intent:
        Resolved NLU intent string (e.g. "checkout", "pay_on_arrival").
    confidence:
        GPT confidence score (0.0–1.0).
    requested_order_type:
        "pickup" | "delivery" | None — set for order-type actions.
    payment_preference:
        "send_link" | "pay_on_arrival" | None — set for payment actions.
    response_key_hint:
        Suggested response key for the handler (optional).
    clarification_text:
        GPT-generated clarification prompt (when action=clarify).
    reason:
        Short reason string from GPT or resolver.
    raw_gpt_status:
        GptCallStatus constant reflecting the call outcome.
    metadata:
        Extra diagnostic info (latency, model, shadow info, …).
    """

    ok: bool
    action: str
    intent: str = ""
    confidence: float = 0.0
    requested_order_type: str | None = None
    payment_preference: str | None = None
    response_key_hint: str | None = None
    clarification_text: str | None = None
    reason: str = ""
    raw_gpt_status: str | None = None
    metadata: dict = field(default_factory=dict, compare=False)


# Sentinel returned when the resolver was not called (mode=disabled).
CONTROL_FLOW_NOT_CALLED = ControlFlowResolution(
    ok=False,
    action=ControlFlowAction.FALLBACK,
    reason="not_called",
    raw_gpt_status=GptCallStatus.DISABLED,
)


# ── Resolver class ────────────────────────────────────────────────────────────


class ControlFlowResolver:
    """Control-flow GPT resolver for checkout / payment / order-type turns.

    Instantiate once per handler (or at service startup) and reuse.
    All constructor arguments are optional; defaults are used in production.
    Inject mocks in tests.
    """

    def __init__(
        self,
        gpt_client: GptSafeClient | None = None,
        context_builder: "GptContextBuilder | None" = None,
        prompt_registry: PromptRegistry | None = None,
        config: "SemanticRepairConfig | None" = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = gpt_client
        self._ctx_builder = context_builder
        self._registry = prompt_registry or PromptRegistry()
        self._config = config
        self._log = logger or _logger

    # ── Public sync API ───────────────────────────────────────────────────────

    def resolve_sync(
        self,
        *,
        context: "ConversationContext",
        user_text: str,
        normalized_text: str,
        task_mode: str,
        local_intent: str | None = None,
        local_confidence: float | None = None,
        local_candidates: list | tuple | None = None,
        local_slots: list | tuple | None = None,
        state: str,
    ) -> ControlFlowResolution:
        """Synchronous bridge around resolve(). Never raises.

        Runs the async resolve() in a dedicated thread with its own event loop.
        Safe to call from synchronous handler methods.
        """
        try:
            coro = self.resolve(
                context=context,
                user_text=user_text,
                normalized_text=normalized_text,
                task_mode=task_mode,
                local_intent=local_intent,
                local_confidence=local_confidence,
                local_candidates=local_candidates,
                local_slots=local_slots,
                state=state,
            )
            timeout_s = (self._get_timeout_ms() / 1000.0) + 1.0
            future = _SYNC_EXECUTOR.submit(_run_async_in_new_loop, coro)
            return future.result(timeout=timeout_s)
        except Exception as exc:
            self._log.warning(
                "control_flow_resolver_sync_error",
                extra={
                    "event": "control_flow_resolver_sync_error",
                    "error": str(exc)[:200],
                    "state": state,
                    "task_mode": task_mode,
                },
            )
            return ControlFlowResolution(
                ok=False,
                action=ControlFlowAction.FALLBACK,
                reason="sync_bridge_error",
                raw_gpt_status=GptCallStatus.UNKNOWN_ERROR,
            )

    # ── Async core ────────────────────────────────────────────────────────────

    async def resolve(
        self,
        *,
        context: "ConversationContext",
        user_text: str,
        normalized_text: str,
        task_mode: str,
        local_intent: str | None = None,
        local_confidence: float | None = None,
        local_candidates: list | tuple | None = None,
        local_slots: list | tuple | None = None,
        state: str,
    ) -> ControlFlowResolution:
        """Resolve a control-flow turn via GPT. Never raises."""
        try:
            return await self._resolve_inner(
                context=context,
                user_text=user_text,
                normalized_text=normalized_text,
                task_mode=task_mode,
                local_intent=local_intent,
                local_confidence=local_confidence,
                local_candidates=local_candidates,
                local_slots=local_slots,
                state=state,
            )
        except Exception as exc:
            self._log.warning(
                "control_flow_resolver_unexpected_error",
                extra={
                    "event": "control_flow_resolver_unexpected_error",
                    "error": str(exc)[:200],
                    "state": state,
                    "task_mode": task_mode,
                },
            )
            return ControlFlowResolution(
                ok=False,
                action=ControlFlowAction.FALLBACK,
                reason="unexpected_resolver_error",
                raw_gpt_status=GptCallStatus.UNKNOWN_ERROR,
            )

    async def _resolve_inner(
        self,
        *,
        context: "ConversationContext",
        user_text: str,
        normalized_text: str,
        task_mode: str,
        local_intent: str | None,
        local_confidence: float | None,
        local_candidates: list | tuple | None,
        local_slots: list | tuple | None,
        state: str,
    ) -> ControlFlowResolution:
        # ── Mode gate ─────────────────────────────────────────────────────────
        mode = self._get_mode_for_task(task_mode)
        if mode == "disabled":
            return CONTROL_FLOW_NOT_CALLED

        # ── Validate task mode ────────────────────────────────────────────────
        if task_mode not in _CONTROL_FLOW_TASK_MODES:
            return ControlFlowResolution(
                ok=False,
                action=ControlFlowAction.FALLBACK,
                reason=f"unsupported_task_mode:{task_mode}",
            )

        # ── Build context + messages ──────────────────────────────────────────
        ctx_builder = self._get_context_builder()
        ctx_packet = ctx_builder.build(
            context=context,
            user_text=user_text,
            normalized_text=normalized_text,
            local_intent=local_intent,
            local_confidence=local_confidence,
            local_candidates=local_candidates,
            local_slots=local_slots,
            task_mode=task_mode,
            state=state,
        )

        messages = self._build_messages(ctx_packet, task_mode, context)

        # ── GPT call ──────────────────────────────────────────────────────────
        cfg = self._get_config()
        model = cfg.model if cfg else "gpt-4o-mini"
        timeout_ms = self._get_timeout_ms()
        client = self._get_client()

        result = await client.call(
            task_mode=task_mode,
            messages=messages,
            model=model,
            timeout_ms=timeout_ms,
            parse_fn=_parse_json_response,
            enabled=True,
            budget_allowed=True,
        )

        # ── Structured logging ────────────────────────────────────────────────
        self._log.info(
            "control_flow_gpt_invoked",
            extra={
                "event": "control_flow_gpt_invoked",
                "control_flow_gpt_mode": mode,
                "control_flow_task_mode": task_mode,
                "control_flow_gpt_status": result.status,
                "control_flow_gpt_latency_ms": result.latency_ms,
                "state": state,
            },
        )

        # ── GPT failure → fallback ────────────────────────────────────────────
        if not result.ok or result.parsed is None:
            self._log.info(
                "control_flow_fallback_reason",
                extra={
                    "event": "control_flow_fallback_reason",
                    "control_flow_gpt_status": result.status,
                    "control_flow_trigger_reason": f"gpt_failed:{result.status}",
                    "state": state,
                    "task_mode": task_mode,
                },
            )
            return ControlFlowResolution(
                ok=False,
                action=ControlFlowAction.FALLBACK,
                reason=f"gpt_failed:{result.status}",
                raw_gpt_status=result.status,
                metadata={
                    "latency_ms": result.latency_ms,
                    "model": result.model,
                    "error": result.error_message,
                },
            )

        # ── Parse resolution ──────────────────────────────────────────────────
        resolution = self._parse_resolution(result.parsed, task_mode, user_text)

        # ── Shadow mode: log but do not apply ─────────────────────────────────
        if mode == "shadow":
            self._log.info(
                "control_flow_gpt_shadow",
                extra={
                    "event": "control_flow_gpt_shadow",
                    "shadow_action": resolution.action,
                    "shadow_intent": resolution.intent,
                    "shadow_order_type": resolution.requested_order_type,
                    "shadow_payment_pref": resolution.payment_preference,
                    "shadow_confidence": resolution.confidence,
                    "state": state,
                    "task_mode": task_mode,
                },
            )
            return ControlFlowResolution(
                ok=False,
                action=ControlFlowAction.FALLBACK,
                reason="shadow_mode_not_applied",
                raw_gpt_status=result.status,
                confidence=resolution.confidence,
                metadata={
                    "shadow_action": resolution.action,
                    "shadow_intent": resolution.intent,
                    "shadow_reason": resolution.reason,
                    "latency_ms": result.latency_ms,
                },
            )

        # ── Structured result logging ─────────────────────────────────────────
        self._log.info(
            "control_flow_gpt_resolved",
            extra={
                "event": "control_flow_gpt_resolved",
                "control_flow_gpt_action": resolution.action,
                "control_flow_gpt_intent": resolution.intent,
                "control_flow_requested_order_type": resolution.requested_order_type,
                "control_flow_payment_preference": resolution.payment_preference,
                "control_flow_gpt_confidence": resolution.confidence,
                "control_flow_applied": resolution.ok,
                "state": state,
                "task_mode": task_mode,
            },
        )

        return resolution

    # ── Message building ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        ctx_packet: dict,
        task_mode: str,
        context: Any,
    ) -> list[dict]:
        """Build GPT API messages from context packet and prompt registry."""
        system_prompt = self._registry.get_system_prompt(task_mode)
        task_instructions = self._registry.get_task_instructions(task_mode)
        output_contract = self._registry.get_output_contract(task_mode)

        user_content = (
            f"Task instructions:\n{task_instructions}\n\n"
            f"Output contract:\n{output_contract}\n\n"
            f"Current state: {ctx_packet.get('current_state', '')}\n"
            f"Customer utterance: {ctx_packet.get('user_text', '')}\n"
            f"Normalized utterance: {ctx_packet.get('normalized_text', '')}\n"
            f"Local intent: {ctx_packet.get('local_intent') or 'UNKNOWN'} "
            f"(confidence: {ctx_packet.get('local_confidence') or 0.0:.2f})\n"
        )

        # Order context.
        order_type = getattr(context, "order_type", None)
        if order_type:
            user_content += f"Current order type: {order_type}\n"

        # Cart summary (item count only — no PII).
        cart = getattr(context, "cart", None)
        if cart is not None:
            try:
                cart_empty = cart.is_empty()
                user_content += f"Cart empty: {cart_empty}\n"
            except Exception:
                pass

        # Pending item / queue status.
        pending = getattr(context, "pending_add_item", None)
        if pending is not None:
            item_name = getattr(pending, "item_name", "")
            if item_name:
                user_content += f"Pending item in progress: {item_name}\n"

        staged = getattr(context, "staged_item_queue", None)
        if staged and len(staged) > 0:
            user_content += f"Staged items waiting: {len(staged)}\n"

        # Previous conversation turns.
        prev_turns = ctx_packet.get("previous_turns") or []
        if prev_turns:
            user_content += f"\nRecent conversation turns:\n{_format_previous_turns(prev_turns)}\n"

        prev_prompt = ctx_packet.get("previous_assistant_prompt")
        if prev_prompt:
            user_content += f"\nPrevious bot message: {prev_prompt}\n"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_resolution(
        self,
        data: dict,
        task_mode: str,
        user_text: str,
    ) -> ControlFlowResolution:
        """Parse a GPT response dict into a ControlFlowResolution."""
        # Action — support both "action" and legacy "decision" field names.
        raw_action = str(
            data.get("action") or data.get("decision") or "fallback"
        ).strip().lower()

        # Resolve through alias map.
        action = _ACTION_ALIASES.get(raw_action, raw_action)

        # Handle task-specific default actions for order-type tasks when
        # GPT returns just "pickup" or "delivery" as the decision value.
        if action == "select_initial_order_type":
            if task_mode == TASK_ORDER_TYPE_CHANGE:
                action = ControlFlowAction.CHANGE_ORDER_TYPE

        # Unknown action → fallback.
        if action not in _VALID_ACTIONS:
            action = ControlFlowAction.FALLBACK

        # Confidence.
        confidence = 0.0
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            pass

        # Intent.
        intent = str(data.get("intent") or "").strip()[:100]

        # Reason / clarification / response key.
        reason = str(data.get("reason") or "").strip()[:200]
        clarification_text = str(data.get("clarification_text") or "").strip() or None
        response_key_hint = str(data.get("response_key_hint") or "").strip() or None

        # Requested order type.
        raw_order_type = str(
            data.get("requested_order_type") or data.get("order_type") or ""
        ).strip().lower()
        # For order-type tasks: if decision=pickup/delivery, capture it.
        if not raw_order_type and raw_action in ("pickup", "delivery"):
            raw_order_type = raw_action
        requested_order_type = raw_order_type if raw_order_type in _VALID_ORDER_TYPES else None

        # Payment preference.
        raw_pref = str(
            data.get("payment_preference") or data.get("decision") or ""
        ).strip().lower()
        # Map legacy decision values to payment preference.
        _PREF_MAP = {
            "sms_link": "send_link",
            "send_link": "send_link",
            "pay_on_arrival": "pay_on_arrival",
        }
        payment_preference = _PREF_MAP.get(raw_pref)
        # Also check explicit payment_preference key.
        if not payment_preference:
            raw_pref2 = str(data.get("payment_preference") or "").strip().lower()
            payment_preference = _PREF_MAP.get(raw_pref2)

        # ok=True only for actionable (non-control) actions.
        ok = action not in {
            ControlFlowAction.CLARIFY,
            ControlFlowAction.FALLBACK,
        }

        return ControlFlowResolution(
            ok=ok,
            action=action,
            intent=intent,
            confidence=confidence,
            requested_order_type=requested_order_type,
            payment_preference=payment_preference,
            response_key_hint=response_key_hint,
            clarification_text=clarification_text,
            reason=reason,
            raw_gpt_status=GptCallStatus.OK,
        )

    # ── Lazy initialisation helpers ───────────────────────────────────────────

    def _get_client(self) -> GptSafeClient:
        if self._client is None:
            underlying = _make_openai_callable()
            self._client = GptSafeClient(
                underlying_client=underlying,
                circuit_breaker=DEFAULT_CIRCUIT_BREAKER,
                config=self._get_config(),
            )
        return self._client

    def _get_context_builder(self) -> "GptContextBuilder":
        if self._ctx_builder is None:
            from app.nlu.turn_resolver.gpt_context_builder import GptContextBuilder
            self._ctx_builder = GptContextBuilder()
        return self._ctx_builder

    def _get_config(self) -> "SemanticRepairConfig":
        if self._config is None:
            from app.config.semantic_repair import get_semantic_repair_config
            self._config = get_semantic_repair_config()
        return self._config

    def _get_mode_for_task(self, task_mode: str) -> str:
        """Return the configured mode for the given task mode."""
        cfg = self._get_config()
        if task_mode == TASK_CHECKOUT_CONFIRMATION_RESOLUTION:
            return str(getattr(cfg, "bucket_5_mode", "disabled"))
        if task_mode == TASK_ORDER_TYPE_CHANGE:
            return str(getattr(cfg, "bucket_6_mode", "disabled"))
        if task_mode == TASK_PICKUP_DELIVERY_INITIAL:
            return str(getattr(cfg, "bucket_6_mode", "disabled"))
        if task_mode == TASK_PAYMENT_PERMISSION_RESOLUTION:
            return str(getattr(cfg, "bucket_payment_mode", "disabled"))
        return "disabled"

    def _get_timeout_ms(self) -> int:
        cfg = self._get_config()
        return int(getattr(cfg, "control_flow_timeout_ms", 700))

    def _get_min_confidence(self) -> float:
        cfg = self._get_config()
        return float(getattr(cfg, "control_flow_min_confidence", 0.70))


# ── Module-level helpers ──────────────────────────────────────────────────────


def _run_async_in_new_loop(coro: Any) -> ControlFlowResolution:
    """Run an async coroutine in a fresh event loop in the calling thread."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _parse_json_response(raw: str) -> dict:
    """Parse raw GPT text as JSON, stripping markdown code fences."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()
    return json.loads(text)


def _make_openai_callable() -> Callable | None:
    """Create an async OpenAI callable. Returns None if not configured."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        import openai

        async_client = openai.AsyncOpenAI(api_key=api_key)

        async def _call(messages: list[dict], model: str, timeout_s: float) -> str:
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=256,
                temperature=0.0,
                timeout=timeout_s,
            )
            return response.choices[0].message.content or ""

        return _call
    except Exception:
        return None


def _format_previous_turns(turns: list) -> str:
    """Format recent conversation turns as compact dialogue for the prompt."""
    if not turns:
        return "  (none)"
    lines: list[str] = []
    for turn in turns[-6:]:
        if isinstance(turn, dict):
            role = str(turn.get("role", ""))
            text = str(turn.get("text", ""))
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            role, text = str(turn[0]), str(turn[1])
        else:
            continue
        role_lower = role.lower()
        prefix = (
            "Bot"
            if "assistant" in role_lower or "bot" in role_lower
            else "Customer"
        )
        lines.append(f"  {prefix}: {text[:150]}")
    return "\n".join(lines)
