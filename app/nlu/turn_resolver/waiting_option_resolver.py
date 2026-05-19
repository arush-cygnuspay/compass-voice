# app/nlu/turn_resolver/waiting_option_resolver.py
"""Bucket-2 GPT waiting-state option resolver for Compass Voice.

Resolves ambiguous user responses in waiting states
(waiting_for_modifier, waiting_for_side, waiting_for_size, waiting_for_side_size)
using GPT as a secondary fallback after the deterministic fast path fails.

Safety invariants
-----------------
* GPT never directly mutates cart, context, or handler state.
* All GPT results must pass validate_waiting_option_resolution() before
  a handler may apply them.
* GPT failure always returns a WaitingOptionResolution with action=fallback.
* No full menu is ever sent — only the current group's options (≤12) plus
  recent turns (≤6) from turn_memory.
* This module never raises into callers.
* Mode=shadow → GPT is called and logged, but ok=False is returned so no
  application can occur.
* Mode=disabled → WAITING_OPTION_NOT_CALLED is returned immediately.
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
    TASK_MODIFIER_OPTION_RESOLUTION,
    TASK_SIDE_OPTION_RESOLUTION,
    TASK_SIZE_OPTION_RESOLUTION,
    PromptRegistry,
)

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig
    from app.nlu.turn_resolver.allowed_option_extractor import AllowedOptionExtractor
    from app.nlu.turn_resolver.gpt_context_builder import GptContextBuilder
    from app.state_machine.models.conversation_context import ConversationContext

_logger = logging.getLogger(__name__)

# Thread pool for synchronous → async bridge.
# 4 workers = 4 concurrent GPT calls; bounded to limit resource use.
_SYNC_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="waiting_opt_gpt",
)

# Map from normalised state → task mode constant
_WAITING_STATE_TO_TASK: dict[str, str] = {
    "waiting_for_modifier": TASK_MODIFIER_OPTION_RESOLUTION,
    "waiting_for_side": TASK_SIDE_OPTION_RESOLUTION,
    "waiting_for_side_size": TASK_SIDE_OPTION_RESOLUTION,
    "waiting_for_size": TASK_SIZE_OPTION_RESOLUTION,
}

_VALID_ACTIONS: frozenset[str] = frozenset({
    "select",
    "negate",
    "list_options",
    "skip",
    "cancel",
    "checkout_request",
    "change_order_type",
    "clarify",
    "fallback",
})


# ── Action constants ──────────────────────────────────────────────────────────


class WaitingOptionAction:
    """String constants for WaitingOptionResolution.action values."""

    SELECT = "select"
    NEGATE = "negate"
    LIST_OPTIONS = "list_options"
    SKIP = "skip"
    CANCEL = "cancel"
    CHECKOUT_REQUEST = "checkout_request"
    CHANGE_ORDER_TYPE = "change_order_type"
    CLARIFY = "clarify"
    FALLBACK = "fallback"


# ── Resolution dataclass ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class WaitingOptionResolution:
    """Structured result from bucket-2 GPT option resolution.

    ok=True and action='select' means GPT found a confident match that
    passed validation and may be applied by the handler.
    ok=False or action='fallback' means fall through to deterministic path.

    Fields
    ------
    ok:
        True when the resolution is safe to apply.
        Always False in shadow mode and on any GPT failure.
    action:
        One of WaitingOptionAction.* constants.
    selected_option_ids:
        GPT-returned IDs (modifier_id / item_id / variant_id) of the
        selected options.  Empty when action != 'select'.
    selected_option_names:
        GPT-returned display names of the selected options.
    selected_size:
        Size hint returned by GPT (e.g. "Small").
    selected_variant:
        Variant hint returned by GPT.
    negated_option_ids:
        Option IDs to be removed / rejected when action='negate'.
    confidence:
        GPT confidence score (0.0–1.0).
    reason:
        Short reason string from GPT or resolver.
    response_key_hint:
        Suggested response key for the handler (optional).
    clarification_text:
        GPT-generated clarification question (for action='clarify').
    raw_gpt_status:
        GptCallStatus constant reflecting the GPT call outcome.
    metadata:
        Extra diagnostic info (latency, model, shadow_action, …).
    """

    ok: bool
    action: str
    selected_option_ids: tuple[str, ...] = ()
    selected_option_names: tuple[str, ...] = ()
    selected_size: str | None = None
    selected_variant: str | None = None
    negated_option_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    reason: str = ""
    response_key_hint: str | None = None
    clarification_text: str | None = None
    raw_gpt_status: str | None = None
    metadata: dict = field(default_factory=dict, compare=False)


# Sentinel — returned when the resolver was not called (mode=disabled)
WAITING_OPTION_NOT_CALLED = WaitingOptionResolution(
    ok=False,
    action=WaitingOptionAction.FALLBACK,
    reason="not_called",
    raw_gpt_status=GptCallStatus.DISABLED,
)


# ── Resolver class ────────────────────────────────────────────────────────────


class WaitingOptionResolver:
    """Bucket-2 GPT resolver for waiting-state option selection.

    Instantiate once per handler (lazy) and reuse across turns.
    All constructor arguments are optional; defaults are used in production.
    Inject mocks in tests.
    """

    def __init__(
        self,
        gpt_client: GptSafeClient | None = None,
        context_builder: "GptContextBuilder | None" = None,
        prompt_registry: PromptRegistry | None = None,
        option_extractor: "AllowedOptionExtractor | None" = None,
        config: "SemanticRepairConfig | None" = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = gpt_client
        self._ctx_builder = context_builder
        self._registry = prompt_registry or PromptRegistry()
        self._option_extractor = option_extractor
        self._config = config
        self._log = logger or _logger

    # ── Public sync API (handler integration) ─────────────────────────────────

    def resolve_sync(
        self,
        *,
        context: "ConversationContext",
        user_text: str,
        normalized_text: str,
        local_intent: str | None,
        local_confidence: float | None,
        local_candidates: list | tuple | None,
        local_slots: list | tuple | None,
        state: str,
        deterministic_match_result: Any = None,
    ) -> WaitingOptionResolution:
        """Synchronous bridge around resolve(). Never raises.

        Runs the async resolve() in a dedicated thread with its own event loop.
        Safe to call from synchronous handler.handle() methods even when the
        process event loop is in use by the WebSocket transport layer.
        """
        try:
            coro = self.resolve(
                context=context,
                user_text=user_text,
                normalized_text=normalized_text,
                local_intent=local_intent,
                local_confidence=local_confidence,
                local_candidates=local_candidates,
                local_slots=local_slots,
                state=state,
                deterministic_match_result=deterministic_match_result,
            )
            timeout_s = (self._get_timeout_ms() / 1000.0) + 1.0
            future = _SYNC_EXECUTOR.submit(_run_async_in_new_loop, coro)
            return future.result(timeout=timeout_s)
        except Exception as exc:
            self._log.warning(
                "waiting_option_resolver_sync_error",
                extra={
                    "event": "waiting_option_resolver_sync_error",
                    "error": str(exc)[:200],
                    "state": state,
                },
            )
            return WaitingOptionResolution(
                ok=False,
                action=WaitingOptionAction.FALLBACK,
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
        local_intent: str | None,
        local_confidence: float | None,
        local_candidates: list | tuple | None,
        local_slots: list | tuple | None,
        state: str,
        deterministic_match_result: Any = None,
    ) -> WaitingOptionResolution:
        """Resolve a waiting-state user response via GPT. Never raises."""
        try:
            return await self._resolve_inner(
                context=context,
                user_text=user_text,
                normalized_text=normalized_text,
                local_intent=local_intent,
                local_confidence=local_confidence,
                local_candidates=local_candidates,
                local_slots=local_slots,
                state=state,
            )
        except Exception as exc:
            self._log.warning(
                "waiting_option_resolver_unexpected_error",
                extra={
                    "event": "waiting_option_resolver_unexpected_error",
                    "error": str(exc)[:200],
                    "state": state,
                },
            )
            return WaitingOptionResolution(
                ok=False,
                action=WaitingOptionAction.FALLBACK,
                reason="unexpected_resolver_error",
                raw_gpt_status=GptCallStatus.UNKNOWN_ERROR,
            )

    async def _resolve_inner(
        self,
        *,
        context: "ConversationContext",
        user_text: str,
        normalized_text: str,
        local_intent: str | None,
        local_confidence: float | None,
        local_candidates: list | tuple | None,
        local_slots: list | tuple | None,
        state: str,
    ) -> WaitingOptionResolution:
        # ── Mode gate ─────────────────────────────────────────────────────────
        mode = self._get_mode()
        if mode == "disabled":
            return WAITING_OPTION_NOT_CALLED

        # ── Task mode ─────────────────────────────────────────────────────────
        task_mode = _WAITING_STATE_TO_TASK.get((state or "").lower())
        if task_mode is None:
            return WaitingOptionResolution(
                ok=False,
                action=WaitingOptionAction.FALLBACK,
                reason="unsupported_state",
            )

        # ── Build context + messages ──────────────────────────────────────────
        option_extractor = self._get_option_extractor()
        allowed_options = option_extractor.extract(context, state)

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
            allowed_options=list(allowed_options),
        )

        messages = self._build_messages(ctx_packet, task_mode, allowed_options)

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
            "waiting_option_gpt_invoked",
            extra={
                "event": "waiting_option_gpt_invoked",
                "waiting_option_gpt_mode": mode,
                "waiting_option_gpt_task_mode": task_mode,
                "waiting_option_gpt_status": result.status,
                "waiting_option_gpt_latency_ms": result.latency_ms,
                "waiting_option_allowed_options_count": len(allowed_options),
                "waiting_option_previous_turn_count": len(
                    ctx_packet.get("previous_turns") or []
                ),
                "state": state,
            },
        )

        # ── GPT failure → fallback ────────────────────────────────────────────
        if not result.ok or result.parsed is None:
            return WaitingOptionResolution(
                ok=False,
                action=WaitingOptionAction.FALLBACK,
                reason=f"gpt_failed:{result.status}",
                raw_gpt_status=result.status,
                metadata={
                    "latency_ms": result.latency_ms,
                    "model": result.model,
                    "error": result.error_message,
                },
            )

        # ── Parse resolution ──────────────────────────────────────────────────
        resolution = self._parse_resolution_from_dict(result.parsed, allowed_options)

        # ── Shadow mode: log but do not apply ─────────────────────────────────
        if mode == "shadow":
            self._log.info(
                "waiting_option_gpt_shadow",
                extra={
                    "event": "waiting_option_gpt_shadow",
                    "shadow_action": resolution.action,
                    "shadow_selected_names": list(resolution.selected_option_names),
                    "shadow_confidence": resolution.confidence,
                    "state": state,
                },
            )
            return WaitingOptionResolution(
                ok=False,
                action=WaitingOptionAction.FALLBACK,
                reason="shadow_mode_not_applied",
                raw_gpt_status=result.status,
                confidence=resolution.confidence,
                selected_option_names=resolution.selected_option_names,
                selected_option_ids=resolution.selected_option_ids,
                metadata={
                    "shadow_action": resolution.action,
                    "shadow_reason": resolution.reason,
                    "latency_ms": result.latency_ms,
                },
            )

        # ── Structured result logging ─────────────────────────────────────────
        self._log.info(
            "waiting_option_gpt_resolved",
            extra={
                "event": "waiting_option_gpt_resolved",
                "waiting_option_gpt_action": resolution.action,
                "waiting_option_gpt_selected_options": list(resolution.selected_option_names),
                "waiting_option_gpt_negated_options": list(resolution.negated_option_ids),
                "waiting_option_gpt_confidence": resolution.confidence,
                "waiting_option_gpt_applied": resolution.ok,
                "waiting_option_gpt_validation_result": "pending",
                "state": state,
            },
        )

        return resolution

    # ── Message building ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        ctx_packet: dict,
        task_mode: str,
        allowed_options: tuple[dict, ...],
    ) -> list[dict]:
        """Build GPT API messages from context packet and prompt registry."""
        system_prompt = self._registry.get_system_prompt(task_mode)
        task_instructions = self._registry.get_task_instructions(task_mode)
        output_contract = self._registry.get_output_contract(task_mode)

        options_str = _format_allowed_options(allowed_options)

        user_content = (
            f"Task instructions:\n{task_instructions}\n\n"
            f"Output contract:\n{output_contract}\n\n"
            f"Current state: {ctx_packet.get('current_state', '')}\n"
            f"Customer utterance: {ctx_packet.get('user_text', '')}\n"
            f"Normalized utterance: {ctx_packet.get('normalized_text', '')}\n"
            f"Local intent: {ctx_packet.get('local_intent') or 'UNKNOWN'} "
            f"(confidence: {ctx_packet.get('local_confidence') or 0.0:.2f})\n"
            f"Allowed options:\n{options_str}\n"
        )

        prev_turns = ctx_packet.get("previous_turns") or []
        if prev_turns:
            user_content += f"\nRecent conversation turns:\n{_format_previous_turns(prev_turns)}\n"

        prev_prompt = ctx_packet.get("previous_assistant_prompt")
        if prev_prompt:
            user_content += f"\nPrevious bot message: {prev_prompt}\n"

        pending_group = ctx_packet.get("pending_group") or {}
        if pending_group:
            pg_name = pending_group.get("name", "")
            pg_min = pending_group.get("min_selector", 0)
            pg_max = pending_group.get("max_selector", 0)
            user_content += (
                f"\nCurrent group: {pg_name} "
                f"(min={pg_min}, max={pg_max})\n"
            )

        pending_item = ctx_packet.get("pending_item") or {}
        if pending_item:
            user_content += f"Pending item: {pending_item.get('name', '')}\n"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_resolution_from_dict(
        self,
        data: dict,
        allowed_options: tuple[dict, ...],
    ) -> WaitingOptionResolution:
        """Parse a GPT response dict into a WaitingOptionResolution.

        Supports two formats:
        New: {"action": "select", "selected_options": [{"id": ..., "name": ...}], ...}
        Legacy: {"decision": "select", "selected_option": "Name", ...}
        """
        # Normalise action field — "decision" is the legacy alias
        raw_action = str(
            data.get("action") or data.get("decision") or "fallback"
        ).strip().lower()

        # Map legacy decision values to action strings
        _LEGACY_MAP: dict[str, str] = {
            "select": "select",
            "select_option": "select",
            "no_match": "fallback",
            "list_options": "list_options",
            "skip": "skip",
            "clarify": "clarify",
            "cancel": "cancel",
            "checkout_request": "checkout_request",
            "change_order_type": "change_order_type",
            "negate": "negate",
        }
        action = _LEGACY_MAP.get(raw_action, raw_action)
        if action not in _VALID_ACTIONS:
            action = WaitingOptionAction.FALLBACK

        # Confidence
        confidence = 0.0
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            pass

        reason = str(data.get("reason") or "").strip()[:200]
        clarification_text = str(data.get("clarification_text") or "").strip() or None
        response_key_hint = str(data.get("response_key_hint") or "").strip() or None
        selected_size = (
            str(data.get("size_hint") or data.get("selected_size") or "").strip() or None
        )
        selected_variant = (
            str(data.get("variant") or data.get("selected_variant") or "").strip() or None
        )

        # ── Parse selected_options (new array format) ─────────────────────────
        selected_ids: list[str] = []
        selected_names: list[str] = []
        raw_opts = data.get("selected_options") or []
        if isinstance(raw_opts, list):
            for entry in raw_opts:
                if isinstance(entry, dict):
                    eid = str(entry.get("id") or "").strip()
                    ename = str(entry.get("name") or "").strip()
                    esize = str(entry.get("size") or "").strip()
                    evar = str(entry.get("variant") or "").strip()
                    if eid:
                        selected_ids.append(eid)
                    if ename:
                        selected_names.append(ename)
                    if esize and not selected_size:
                        selected_size = esize
                    if evar and not selected_variant:
                        selected_variant = evar
                elif isinstance(entry, str) and entry.strip():
                    selected_names.append(entry.strip())

        # ── Legacy: single selected_option string ─────────────────────────────
        if not selected_names and not selected_ids:
            legacy = str(data.get("selected_option") or "").strip()
            if legacy:
                selected_names.append(legacy)
                # Try to resolve the ID from allowed_options
                for opt in allowed_options:
                    if str(opt.get("name") or "").strip().lower() == legacy.lower():
                        oid = str(
                            opt.get("modifier_id")
                            or opt.get("item_id")
                            or opt.get("variant_id")
                            or ""
                        )
                        if oid:
                            selected_ids.append(oid)
                        break

        # ── Parse negated_options ─────────────────────────────────────────────
        negated_ids: list[str] = []
        raw_negs = data.get("negated_options") or []
        if isinstance(raw_negs, list):
            for entry in raw_negs:
                if isinstance(entry, dict):
                    eid = str(entry.get("id") or "").strip()
                    ename = str(entry.get("name") or "").strip()
                    if eid:
                        negated_ids.append(eid)
                    elif ename:
                        # Resolve by name
                        for opt in allowed_options:
                            if str(opt.get("name") or "").strip().lower() == ename.lower():
                                oid = str(
                                    opt.get("modifier_id")
                                    or opt.get("item_id")
                                    or opt.get("variant_id")
                                    or ""
                                )
                                if oid:
                                    negated_ids.append(oid)
                                break
                elif isinstance(entry, str) and entry.strip():
                    ename = entry.strip()
                    for opt in allowed_options:
                        if str(opt.get("name") or "").strip().lower() == ename.lower():
                            oid = str(
                                opt.get("modifier_id")
                                or opt.get("item_id")
                                or opt.get("variant_id")
                                or ""
                            )
                            if oid:
                                negated_ids.append(oid)
                            break

        # ok=True only for select/negate (control actions are not "ok" to apply)
        ok = action in {WaitingOptionAction.SELECT, WaitingOptionAction.NEGATE}

        return WaitingOptionResolution(
            ok=ok,
            action=action,
            selected_option_ids=tuple(selected_ids),
            selected_option_names=tuple(selected_names),
            selected_size=selected_size,
            selected_variant=selected_variant,
            negated_option_ids=tuple(negated_ids),
            confidence=confidence,
            reason=reason,
            response_key_hint=response_key_hint,
            clarification_text=clarification_text,
            raw_gpt_status=GptCallStatus.OK,
        )

    # ── Lazy initialisation helpers ───────────────────────────────────────────

    def _get_client(self) -> GptSafeClient:
        """Return (or create) the GptSafeClient with an OpenAI underlying callable."""
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

    def _get_option_extractor(self) -> "AllowedOptionExtractor":
        if self._option_extractor is None:
            from app.nlu.turn_resolver.allowed_option_extractor import AllowedOptionExtractor
            self._option_extractor = AllowedOptionExtractor()
        return self._option_extractor

    def _get_config(self) -> "SemanticRepairConfig":
        if self._config is None:
            from app.config.semantic_repair import get_semantic_repair_config
            self._config = get_semantic_repair_config()
        return self._config

    def _get_mode(self) -> str:
        """Return bucket_2_mode from config ('disabled'|'shadow'|'inline')."""
        cfg = self._get_config()
        return str(getattr(cfg, "bucket_2_mode", "disabled"))

    def _get_timeout_ms(self) -> int:
        """Return per-call timeout for bucket-2 GPT calls."""
        cfg = self._get_config()
        return int(getattr(cfg, "bucket_2_timeout_ms", 700))

    def _get_min_confidence(self) -> float:
        """Return minimum confidence threshold for bucket-2 apply."""
        cfg = self._get_config()
        return float(getattr(cfg, "bucket_2_min_confidence", 0.70))


# ── Module-level helpers ──────────────────────────────────────────────────────


def _run_async_in_new_loop(coro: Any) -> WaitingOptionResolution:
    """Run an async coroutine in a fresh event loop in the calling thread.

    Used by resolve_sync() to run the async GPT call in a dedicated thread
    without interfering with any existing event loop in the main thread.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _parse_json_response(raw: str) -> dict:
    """Parse raw GPT text as JSON, stripping markdown code fences if present."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()
    return json.loads(text)


def _make_openai_callable() -> Callable | None:
    """Create an async OpenAI API callable for GptSafeClient.

    Returns None if the API key is not set or openai is not installed.
    GptSafeClient treats None → API_KEY_MISSING (pre-call guard).
    """
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


def _format_allowed_options(allowed_options: tuple[dict, ...]) -> str:
    """Format allowed options as a compact numbered list for the GPT prompt."""
    if not allowed_options:
        return "  (no options available)"
    lines: list[str] = []
    for opt in allowed_options:
        idx = opt.get("index", "?")
        name = opt.get("name", "")
        aliases = opt.get("aliases") or opt.get("size_variants") or []
        line = f"  {idx}. {name}"
        if aliases:
            line += f" (also: {', '.join(str(a) for a in list(aliases)[:4])})"
        lines.append(line)
    return "\n".join(lines)


def _format_previous_turns(turns: list) -> str:
    """Format recent conversation turns as a compact dialogue for the prompt."""
    if not turns:
        return "  (none)"
    lines: list[str] = []
    for turn in turns[-6:]:  # honour the ≤6-turn context window
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
