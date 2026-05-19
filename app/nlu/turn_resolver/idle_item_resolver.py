# app/nlu/turn_resolver/idle_item_resolver.py
"""Bucket-0 GPT idle natural item resolver for Compass Voice.

Resolves bare menu-item phrases spoken in the idle state (e.g. "hamburger
with small coke", "large fries", "six piece wings") into structured item
plans that the existing AddItemHandler / deterministic add-item flow can
execute.

Safety invariants
-----------------
* GPT never directly mutates cart, context, or handler state.
* All GPT results must pass validate_idle_item_resolution() before any
  handler may apply them.
* GPT failure always returns an IdleItemResolution with ok=False.
* No full menu is ever sent — only bounded menu_candidates (≤12) plus
  recent turns (≤6) from turn_memory.
* This module never raises into callers.
* Mode=shadow → GPT is called and logged, but ok=False is returned so no
  application can occur.
* Mode=disabled → IDLE_ITEM_NOT_CALLED is returned immediately.

Integration
-----------
Called from final_turn_decision_resolver._call_bucket0_live() which
replaces the old _call_bucket0_stub placeholder.  The return value is
converted to a GptTurnResolution for the existing framework.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from app.nlu.turn_resolver.gpt_safe_client import GptCallStatus, GptSafeClient
from app.nlu.turn_resolver.gpt_circuit_breaker import DEFAULT_CIRCUIT_BREAKER
from app.nlu.turn_resolver.prompt_registry import (
    TASK_IDLE_ADD_ITEM_OR_MENU_QUERY,
    PromptRegistry,
)

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig
    from app.nlu.turn_resolver.menu_candidate_provider import MenuCandidateProvider

_logger = logging.getLogger(__name__)

# Thread pool for synchronous → async bridge (same pattern as waiting_option_resolver).
_SYNC_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="idle_item_gpt",
)

_VALID_DECISIONS: frozenset[str] = frozenset({
    "execute", "clarify", "reject", "fallback",
})

_VALID_INTENTS: frozenset[str] = frozenset({
    "add_item", "ask_item_info", "ask_menu",
    "checkout", "cancel", "unknown",
})

_MAX_MENU_CANDIDATES: int = 12
_MAX_PREV_TURNS: int = 6


# ── Decision constants ────────────────────────────────────────────────────────


class IdleItemDecision:
    """String constants for IdleItemResolution.decision values."""
    EXECUTE = "execute"
    CLARIFY = "clarify"
    REJECT = "reject"
    FALLBACK = "fallback"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IdleResolvedModifier:
    """One modifier from a GPT-proposed item plan."""
    modifier_id: str | None
    name: str
    operation: str = "add"   # add | remove | extra | light


@dataclass(frozen=True)
class IdleResolvedSide:
    """One side/drink from a GPT-proposed item plan."""
    item_id: str | None
    name: str
    size: str | None = None
    variant: str | None = None
    quantity: int = 1


@dataclass(frozen=True)
class IdleResolvedItem:
    """One item entry from a GPT idle-state resolution.

    Parsed and validated before any handler may use it.  Never applied
    to the cart directly from this module.
    """
    item_id: str | None
    item_name: str
    quantity: int = 1
    size_name: str | None = None
    size_id: str | None = None
    variant_name: str | None = None
    variant_id: str | None = None
    sides: tuple[IdleResolvedSide, ...] = ()
    modifiers: tuple[IdleResolvedModifier, ...] = ()
    raw_span: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class IdleItemResolution:
    """Full result of one bucket-0 idle item resolution attempt.

    ok=True and decision='execute' means GPT found confident item(s) that
    passed validation and may be routed to AddItemHandler.
    ok=False or decision='fallback' means use the local deterministic path.
    """
    ok: bool
    intent: str
    confidence: float
    item_plan: tuple[IdleResolvedItem, ...]
    unresolved_spans: tuple[str, ...]
    decision: str
    reason: str
    raw_gpt_status: str | None = None
    metadata: dict = field(default_factory=dict, compare=False)


# Sentinel — returned when the resolver was not called (mode=disabled)
IDLE_ITEM_NOT_CALLED = IdleItemResolution(
    ok=False,
    intent="unknown",
    confidence=0.0,
    item_plan=(),
    unresolved_spans=(),
    decision=IdleItemDecision.FALLBACK,
    reason="not_called",
    raw_gpt_status=GptCallStatus.DISABLED,
)


# ── Conversion to GptTurnResolution ──────────────────────────────────────────


def to_gpt_turn_resolution(resolution: IdleItemResolution) -> "Any":
    """Convert IdleItemResolution → GptTurnResolution for the existing framework.

    This is the adapter between the new resolver result and the existing
    FinalTurnDecisionResolver framework (schemas.py types).
    """
    from app.nlu.turn_resolver.schemas import (
        GptTurnResolution,
        ResolvedItemPlan,
        ResolvedModifierPlan,
        ResolvedSidePlan,
    )
    from app.nlu.turn_resolver.bucket_policy import BUCKET_IDLE_ITEM

    # Map internal decision to GptTurnResolution decision vocabulary
    _DECISION_MAP: dict[str, str] = {
        "execute": "add_items",
        "clarify": "clarify",
        "reject": "no_match",
        "fallback": "skipped",
    }
    gpt_decision = _DECISION_MAP.get(resolution.decision, "skipped")

    if not resolution.ok or resolution.decision != IdleItemDecision.EXECUTE:
        return GptTurnResolution(
            bucket=BUCKET_IDLE_ITEM,
            decision=gpt_decision,
            intent=resolution.intent,
            confidence=resolution.confidence,
            reason_code=resolution.reason,
            gpt_called=(resolution.raw_gpt_status != GptCallStatus.DISABLED),
            skipped_reason=resolution.reason if gpt_decision == "skipped" else None,
        )

    # Convert item plan
    items = tuple(
        ResolvedItemPlan(
            item_name=it.item_name,
            quantity=it.quantity,
            size=it.size_name,
            variant=it.variant_name,
            sides=tuple(
                ResolvedSidePlan(name=s.name, quantity=s.quantity, size=s.size)
                for s in it.sides
            ),
            modifiers=tuple(
                ResolvedModifierPlan(name=m.name, operation=m.operation)
                for m in it.modifiers
            ),
        )
        for it in resolution.item_plan
    )

    return GptTurnResolution(
        bucket=BUCKET_IDLE_ITEM,
        decision="add_items",
        intent=resolution.intent,
        items=items,
        confidence=resolution.confidence,
        reason_code=resolution.reason,
        gpt_called=True,
        safe_to_apply=False,  # always set by validators.py, never by resolver
    )


# ── Resolver class ────────────────────────────────────────────────────────────


class IdleItemResolver:
    """Bucket-0 GPT resolver for idle-state natural item phrases.

    Instantiate once per process (lazy singleton in integration) and reuse
    across turns.  All constructor arguments are optional; defaults are used
    in production.  Inject mocks in tests.
    """

    def __init__(
        self,
        gpt_client: GptSafeClient | None = None,
        prompt_registry: PromptRegistry | None = None,
        menu_candidate_provider: "MenuCandidateProvider | None" = None,
        config: "SemanticRepairConfig | None" = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = gpt_client
        self._registry = prompt_registry or PromptRegistry()
        self._menu_provider = menu_candidate_provider
        self._config = config
        self._log = logger or _logger

    # ── Public sync API (integration entry point) ─────────────────────────────

    def resolve_sync(
        self,
        *,
        user_text: str,
        state: str,
        local_intent: str | None,
        local_confidence: float | None,
        local_slots: list | tuple | None,
        previous_turns: tuple | list | None = None,
        menu_candidates: tuple[dict, ...] | None = None,
        timeout_ms: int | None = None,
        config: "SemanticRepairConfig | None" = None,
    ) -> IdleItemResolution:
        """Synchronous bridge around resolve(). Never raises.

        Runs the async resolve() in a dedicated thread with its own event loop.
        Safe to call from synchronous handler.handle() or final_turn_decision_resolver.
        """
        cfg = config or self._config
        try:
            coro = self.resolve(
                user_text=user_text,
                state=state,
                local_intent=local_intent,
                local_confidence=local_confidence,
                local_slots=local_slots,
                previous_turns=previous_turns,
                menu_candidates=menu_candidates,
                timeout_ms=timeout_ms,
                config=cfg,
            )
            effective_timeout_ms = timeout_ms or self._get_timeout_ms(cfg)
            timeout_s = (effective_timeout_ms / 1000.0) + 1.0
            future = _SYNC_EXECUTOR.submit(_run_async_in_new_loop, coro)
            return future.result(timeout=timeout_s)
        except Exception as exc:
            self._log.warning(
                "idle_item_resolver_sync_error",
                extra={
                    "event": "idle_item_resolver_sync_error",
                    "error": str(exc)[:200],
                    "state": state,
                },
            )
            return IdleItemResolution(
                ok=False,
                intent="unknown",
                confidence=0.0,
                item_plan=(),
                unresolved_spans=(),
                decision=IdleItemDecision.FALLBACK,
                reason="sync_bridge_error",
                raw_gpt_status=GptCallStatus.UNKNOWN_ERROR,
            )

    # ── Async core ────────────────────────────────────────────────────────────

    async def resolve(
        self,
        *,
        user_text: str,
        state: str,
        local_intent: str | None,
        local_confidence: float | None,
        local_slots: list | tuple | None,
        previous_turns: tuple | list | None = None,
        menu_candidates: tuple[dict, ...] | None = None,
        timeout_ms: int | None = None,
        config: "SemanticRepairConfig | None" = None,
    ) -> IdleItemResolution:
        """Resolve an idle-state utterance via GPT. Never raises."""
        try:
            return await self._resolve_inner(
                user_text=user_text,
                state=state,
                local_intent=local_intent,
                local_confidence=local_confidence,
                local_slots=local_slots,
                previous_turns=previous_turns,
                menu_candidates=menu_candidates,
                timeout_ms=timeout_ms,
                config=config,
            )
        except Exception as exc:
            self._log.warning(
                "idle_item_resolver_unexpected_error",
                extra={
                    "event": "idle_item_resolver_unexpected_error",
                    "error": str(exc)[:200],
                    "state": state,
                },
            )
            return IdleItemResolution(
                ok=False,
                intent="unknown",
                confidence=0.0,
                item_plan=(),
                unresolved_spans=(),
                decision=IdleItemDecision.FALLBACK,
                reason="unexpected_resolver_error",
                raw_gpt_status=GptCallStatus.UNKNOWN_ERROR,
            )

    async def _resolve_inner(
        self,
        *,
        user_text: str,
        state: str,
        local_intent: str | None,
        local_confidence: float | None,
        local_slots: list | tuple | None,
        previous_turns: tuple | list | None,
        menu_candidates: tuple[dict, ...] | None,
        timeout_ms: int | None,
        config: "SemanticRepairConfig | None",
    ) -> IdleItemResolution:
        # ── Mode gate ─────────────────────────────────────────────────────────
        mode = self._get_mode(config)
        if mode == "disabled":
            return IDLE_ITEM_NOT_CALLED

        # ── Only operate in idle state ─────────────────────────────────────────
        if (state or "").lower().strip() != "idle":
            return IdleItemResolution(
                ok=False,
                intent="unknown",
                confidence=0.0,
                item_plan=(),
                unresolved_spans=(),
                decision=IdleItemDecision.FALLBACK,
                reason="unsupported_state",
            )

        # ── Compute menu candidates if not provided ────────────────────────────
        if menu_candidates is None:
            menu_candidates = self._get_menu_candidates(user_text, config)

        # ── Build GPT messages ────────────────────────────────────────────────
        messages = self._build_messages(
            user_text=user_text,
            local_intent=local_intent,
            local_confidence=local_confidence,
            local_slots=local_slots,
            previous_turns=previous_turns or (),
            menu_candidates=menu_candidates,
        )

        # ── GPT call ──────────────────────────────────────────────────────────
        cfg = config or self._config
        model = getattr(cfg, "model", "gpt-4o-mini") if cfg else "gpt-4o-mini"
        effective_timeout_ms = timeout_ms or self._get_timeout_ms(cfg)
        client = self._get_client()

        result = await client.call(
            task_mode=TASK_IDLE_ADD_ITEM_OR_MENU_QUERY,
            messages=messages,
            model=model,
            timeout_ms=effective_timeout_ms,
            parse_fn=_parse_json_response,
            enabled=True,
            budget_allowed=True,
        )

        # ── Structured logging ────────────────────────────────────────────────
        self._log.info(
            "idle_item_resolver_invoked",
            extra={
                "event": "idle_item_resolver_invoked",
                "idle_item_resolver_mode": mode,
                "idle_item_gpt_status": result.status,
                "idle_item_menu_candidates_count": len(menu_candidates),
                "idle_item_previous_turn_count": len(list(previous_turns or ())),
                "state": state,
            },
        )

        # ── GPT failure → fallback ────────────────────────────────────────────
        if not result.ok or result.parsed is None:
            return IdleItemResolution(
                ok=False,
                intent="unknown",
                confidence=0.0,
                item_plan=(),
                unresolved_spans=(),
                decision=IdleItemDecision.FALLBACK,
                reason=f"gpt_failed:{result.status}",
                raw_gpt_status=result.status,
                metadata={
                    "latency_ms": result.latency_ms,
                    "model": result.model,
                    "error": result.error_message,
                },
            )

        # ── Parse resolution ──────────────────────────────────────────────────
        resolution = self._parse_resolution_from_dict(result.parsed)

        # ── Shadow mode: log but do not apply ─────────────────────────────────
        if mode == "shadow":
            self._log.info(
                "idle_item_gpt_shadow",
                extra={
                    "event": "idle_item_gpt_shadow",
                    "shadow_decision": resolution.decision,
                    "shadow_intent": resolution.intent,
                    "shadow_items_count": len(resolution.item_plan),
                    "shadow_confidence": resolution.confidence,
                    "state": state,
                },
            )
            return IdleItemResolution(
                ok=False,
                intent=resolution.intent,
                confidence=resolution.confidence,
                item_plan=(),
                unresolved_spans=(),
                decision=IdleItemDecision.FALLBACK,
                reason="shadow_mode_not_applied",
                raw_gpt_status=result.status,
                metadata={
                    "shadow_decision": resolution.decision,
                    "shadow_items": [it.item_name for it in resolution.item_plan],
                    "latency_ms": result.latency_ms,
                },
            )

        # ── Validate before returning ok=True ─────────────────────────────────
        from app.nlu.turn_resolver.idle_item_validator import (
            validate_idle_item_resolution,
        )
        min_conf = float(getattr(cfg, "bucket_0_min_confidence", 0.70) if cfg else 0.70)
        validation = validate_idle_item_resolution(
            resolution,
            menu_candidates,
            None,
            min_confidence=min_conf,
        )

        # ── Structured result logging ─────────────────────────────────────────
        self._log.info(
            "idle_item_gpt_resolved",
            extra={
                "event": "idle_item_gpt_resolved",
                "idle_item_gpt_decision": resolution.decision,
                "idle_item_gpt_intent": resolution.intent,
                "idle_item_gpt_items": [it.item_name for it in resolution.item_plan],
                "idle_item_unresolved_spans": list(resolution.unresolved_spans),
                "idle_item_gpt_confidence": resolution.confidence,
                "idle_item_validation_result": validation.reason,
                "idle_item_applied": resolution.ok and validation.is_valid,
                "state": state,
            },
        )

        if not validation.is_valid:
            return IdleItemResolution(
                ok=False,
                intent=resolution.intent,
                confidence=resolution.confidence,
                item_plan=(),
                unresolved_spans=resolution.unresolved_spans,
                decision=IdleItemDecision.FALLBACK,
                reason=f"validation_failed:{validation.reason}",
                raw_gpt_status=result.status,
                metadata={"validation_block": validation.block_reason},
            )

        # Pass through the validated resolution with ok=True
        return IdleItemResolution(
            ok=resolution.ok,
            intent=resolution.intent,
            confidence=resolution.confidence,
            item_plan=resolution.item_plan,
            unresolved_spans=resolution.unresolved_spans,
            decision=resolution.decision,
            reason=resolution.reason,
            raw_gpt_status=result.status,
            metadata={"latency_ms": result.latency_ms, "model": result.model},
        )

    # ── Message building ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        *,
        user_text: str,
        local_intent: str | None,
        local_confidence: float | None,
        local_slots: list | tuple | None,
        previous_turns: tuple | list,
        menu_candidates: tuple[dict, ...],
    ) -> list[dict]:
        """Build GPT API messages for the idle item resolution task."""
        task_mode = TASK_IDLE_ADD_ITEM_OR_MENU_QUERY
        system_prompt = self._registry.get_system_prompt(task_mode)
        task_instructions = self._registry.get_task_instructions(task_mode)
        output_contract = self._registry.get_output_contract(task_mode)

        candidates_str = _format_menu_candidates(menu_candidates)
        turns_str = _format_previous_turns(list(previous_turns))

        user_content = (
            f"Task instructions:\n{task_instructions}\n\n"
            f"Output contract:\n{output_contract}\n\n"
            f"Customer utterance: {user_text}\n"
            f"Local intent: {local_intent or 'UNKNOWN'} "
            f"(confidence: {float(local_confidence or 0.0):.2f})\n"
        )

        if local_slots:
            slots_repr = ", ".join(
                f"{s.get('n', '?')}={s.get('v', '?')}" if isinstance(s, dict)
                else str(s)
                for s in list(local_slots)[:10]
            )
            user_content += f"Local slots: {slots_repr}\n"

        user_content += f"\nMenu candidates:\n{candidates_str}\n"

        if turns_str:
            user_content += f"\nRecent conversation:\n{turns_str}\n"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_resolution_from_dict(self, data: dict) -> IdleItemResolution:
        """Parse a GPT response dict into an IdleItemResolution."""
        try:
            decision = str(data.get("decision") or "fallback").strip().lower()
            if decision not in _VALID_DECISIONS:
                decision = IdleItemDecision.FALLBACK

            intent = str(data.get("intent") or "unknown").strip().lower()
            if intent not in _VALID_INTENTS:
                intent = "unknown"

            confidence = 0.0
            try:
                confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            except (TypeError, ValueError):
                pass

            reason = str(data.get("reason") or "").strip()[:200]

            # Parse items
            raw_items = data.get("items") or []
            item_plan = self._parse_items(raw_items)

            # Parse unresolved spans
            raw_unresolved = data.get("unresolved_spans") or []
            unresolved_spans = tuple(
                str(s).strip() for s in raw_unresolved
                if s and str(s).strip()
            )

            ok = (decision == IdleItemDecision.EXECUTE and bool(item_plan))

            return IdleItemResolution(
                ok=ok,
                intent=intent,
                confidence=confidence,
                item_plan=item_plan,
                unresolved_spans=unresolved_spans,
                decision=decision,
                reason=reason,
            )
        except Exception as exc:
            return IdleItemResolution(
                ok=False,
                intent="unknown",
                confidence=0.0,
                item_plan=(),
                unresolved_spans=(),
                decision=IdleItemDecision.FALLBACK,
                reason=f"parse_error:{str(exc)[:100]}",
            )

    def _parse_items(self, raw_items: list) -> tuple[IdleResolvedItem, ...]:
        """Parse GPT items list into IdleResolvedItem tuple."""
        if not isinstance(raw_items, list):
            return ()
        parsed: list[IdleResolvedItem] = []
        for entry in raw_items[:10]:  # cap at 10 items
            if not isinstance(entry, dict):
                continue
            item = self._parse_one_item(entry)
            if item:
                parsed.append(item)
        return tuple(parsed)

    def _parse_one_item(self, entry: dict) -> IdleResolvedItem | None:
        """Parse one item dict entry."""
        item_name = str(entry.get("item_name") or "").strip()
        if not item_name:
            return None

        item_id = str(entry.get("item_id") or "").strip() or None
        quantity = 1
        try:
            quantity = max(1, min(20, int(entry.get("quantity") or 1)))
        except (TypeError, ValueError):
            pass

        size_name = str(entry.get("size") or "").strip() or None
        variant_name = str(entry.get("variant") or "").strip() or None
        raw_span = str(entry.get("raw_span") or "").strip()
        item_confidence = 0.0
        try:
            item_confidence = max(0.0, min(1.0, float(entry.get("confidence", 0.0))))
        except (TypeError, ValueError):
            pass

        # Parse sides
        sides = self._parse_sides(entry.get("sides") or [])

        # Parse modifiers
        modifiers = self._parse_modifiers(entry.get("modifiers") or [])

        return IdleResolvedItem(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            size_name=size_name,
            variant_name=variant_name,
            sides=sides,
            modifiers=modifiers,
            raw_span=raw_span,
            confidence=item_confidence,
        )

    def _parse_sides(self, raw_sides: list) -> tuple[IdleResolvedSide, ...]:
        sides: list[IdleResolvedSide] = []
        for s in raw_sides[:6]:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or "").strip()
            if not name:
                continue
            sides.append(IdleResolvedSide(
                item_id=str(s.get("item_id") or "").strip() or None,
                name=name,
                size=str(s.get("size") or "").strip() or None,
                variant=str(s.get("variant") or "").strip() or None,
                quantity=max(1, int(s.get("quantity") or 1)),
            ))
        return tuple(sides)

    def _parse_modifiers(self, raw_mods: list) -> tuple[IdleResolvedModifier, ...]:
        mods: list[IdleResolvedModifier] = []
        for m in raw_mods[:10]:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "").strip()
            if not name:
                continue
            operation = str(m.get("operation") or "add").strip().lower()
            if operation not in {"add", "remove", "extra", "light"}:
                operation = "add"
            mods.append(IdleResolvedModifier(
                modifier_id=str(m.get("modifier_id") or "").strip() or None,
                name=name,
                operation=operation,
            ))
        return tuple(mods)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_mode(self, config: "SemanticRepairConfig | None") -> str:
        cfg = config or self._config
        if cfg is None:
            return "disabled"
        return str(getattr(cfg, "bucket_0_mode", "disabled") or "disabled")

    def _get_timeout_ms(self, config: "SemanticRepairConfig | None") -> int:
        cfg = config or self._config
        if cfg is None:
            return 700
        return int(
            getattr(cfg, "bucket_0_timeout_ms", None)
            or getattr(cfg, "bucket_timeout_ms", 700)
            or 700
        )

    def _get_client(self) -> GptSafeClient:
        if self._client is None:
            openai_fn = _make_openai_callable()
            self._client = GptSafeClient(
                underlying_client=openai_fn,
                circuit_breaker=DEFAULT_CIRCUIT_BREAKER,
            )
        return self._client

    def _get_menu_candidates(
        self,
        user_text: str,
        config: "SemanticRepairConfig | None",
    ) -> tuple[dict, ...]:
        """Compute menu candidates from user_text using provider if available."""
        if not self._menu_provider:
            return ()
        cfg = config or self._config
        limit = int(getattr(cfg, "idle_item_menu_candidate_limit", _MAX_MENU_CANDIDATES) if cfg else _MAX_MENU_CANDIDATES)
        try:
            return self._menu_provider.get_candidates(user_text, limit=limit)
        except Exception:
            return ()


# ── Module-level helpers ──────────────────────────────────────────────────────


def _run_async_in_new_loop(coro) -> IdleItemResolution:
    """Run *coro* in a brand-new event loop (worker thread pattern)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_openai_callable() -> Callable | None:
    """Return an async callable wrapping the OpenAI client, or None if no API key."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

        async def _call(messages: list[dict], model: str, timeout_ms: int) -> str:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                timeout=timeout_ms / 1000.0,
            )
            return resp.choices[0].message.content or ""

        return _call
    except Exception:
        return None


def _parse_json_response(raw: str) -> dict:
    """Parse a JSON string, stripping markdown code fences if present."""
    if not raw:
        raise ValueError("empty GPT response")
    text = raw.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    return json.loads(text)


def _format_menu_candidates(candidates: tuple[dict, ...]) -> str:
    """Format candidates as a numbered list for the GPT prompt."""
    if not candidates:
        return "(no candidates provided)"
    lines: list[str] = []
    for i, c in enumerate(candidates[:_MAX_MENU_CANDIDATES]):
        name = str(c.get("name") or "")
        item_id = str(c.get("item_id") or "")
        aliases = c.get("aliases") or []
        sizes = c.get("available_sizes") or []
        variants = c.get("available_variants") or []

        parts = [f"{i + 1}. {name}"]
        if item_id:
            parts.append(f"(id={item_id})")
        if aliases:
            parts.append(f"aliases: {', '.join(aliases[:3])}")
        if sizes:
            parts.append(f"sizes: {', '.join(sizes[:4])}")
        if variants:
            parts.append(f"variants: {', '.join(variants[:4])}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _format_previous_turns(turns: list) -> str:
    """Format previous turns as Bot/Customer dialogue for the GPT prompt."""
    if not turns:
        return ""
    lines: list[str] = []
    for turn in turns[-_MAX_PREV_TURNS:]:
        if isinstance(turn, dict):
            role = str(turn.get("role") or "").lower()
            text = str(turn.get("text") or "").strip()
        elif isinstance(turn, (tuple, list)) and len(turn) >= 2:
            role, text = str(turn[0]).lower(), str(turn[1]).strip()
        else:
            continue
        if not text:
            continue
        label = "Bot" if role in {"assistant", "bot"} else "Customer"
        lines.append(f"{label}: {text}")
    return "\n".join(lines) if lines else ""
