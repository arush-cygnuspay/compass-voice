# app/nlu/semantic_repair/add_item_service.py
"""GPT ADD_ITEM extractor service (shadow-only).

Shadow-only contract
--------------------
* result is parsed and logged; it is NEVER applied to cart, state, or response.
* GPT timeout / error / missing API key NEVER stops the local deterministic flow.
* API key is read from the environment at call time; it is NEVER stored or logged.
* Prompt contains no full menu, no full cart JSON, no full Intent enum, no PII.

TODO (Phase 2): Deduplicate JSONL writes — currently, TurnEngine's main JSONL
  path (gpt_repair_jsonl_logger) and the ADD_ITEM extractor path both write
  separate records to the JSONL log for the same turn.  Consolidate into a
  single merged record per turn so downstream training pipelines see one row.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from app.nlu.semantic_repair.add_item_extractor import (
    ADD_ITEM_NOT_CALLED,
    AddItemEligibilityGate,
    AddItemPayloadBuilder,
    GptAddItemPlan,
)
from app.nlu.semantic_repair.add_item_output_parser import parse_add_item_output
from app.nlu.semantic_repair.daily_budget import GptDailyBudget

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig
    from app.nlu.intent_resolution.intent_result import IntentResult
    from app.session.session import Session
    from app.state_machine.models.conversation_state import ConversationState


class AddItemExtractorService:
    """Shadow-mode GPT service that extracts structured add-item plans.

    Usage
    -----
    The service is instantiated once (in TurnEngine.__init__) and reused
    across turns.  The OpenAI client is created lazily on first use.

    run() is the main entry point.  It always returns a GptAddItemPlan —
    never raises.
    """

    def __init__(self, config: "SemanticRepairConfig | None" = None) -> None:
        from app.config.semantic_repair import get_semantic_repair_config

        self._config: SemanticRepairConfig = config or get_semantic_repair_config()
        self._eligibility_gate = AddItemEligibilityGate()
        self._payload_builder = AddItemPayloadBuilder()
        self._client: Any = None  # lazy-initialised OpenAI client
        self._daily_budget = GptDailyBudget(limit=self._config.daily_budget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        session: "Session",
        nlu: Any,
        intent_result: "IntentResult",
        state: "ConversationState",
        intent_candidates: Any = None,
        gpt_shadow_decision: str | None = None,
        gpt_shadow_repaired_intent: str | None = None,
    ) -> GptAddItemPlan:
        """Run the ADD_ITEM extractor for the current turn.

        Returns a GptAddItemPlan regardless of outcome.  Never raises.
        """
        cfg = self._config
        normalized_text = (
            getattr(nlu, "normalized_text", "") or ""
        ).strip()

        # ── 1. Eligibility check ─────────────────────────────────────────
        eligible, reason = self._eligibility_gate.check(
            intent_result=intent_result,
            state=state,
            normalized_text=normalized_text,
            gpt_shadow_decision=gpt_shadow_decision,
            gpt_shadow_repaired_intent=gpt_shadow_repaired_intent,
            config=cfg,
        )
        if not eligible:
            return GptAddItemPlan(
                decision="no_repair",
                eligible=False,
                skipped_reason=reason,
            )

        # ── 2. Daily budget ──────────────────────────────────────────────
        if not self._daily_budget.try_consume():
            return GptAddItemPlan(
                decision="no_repair",
                eligible=True,
                skipped_reason="daily_budget_exceeded",
            )

        # ── 3. API key check ─────────────────────────────────────────────
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return GptAddItemPlan(
                decision="no_repair",
                eligible=True,
                skipped_reason="missing_api_key",
            )

        # ── 4. Build payload ─────────────────────────────────────────────
        ctx = getattr(session, "conversation_context", None)
        current_item = (
            getattr(ctx, "current_item_name", "") or ""
        ) if ctx else ""
        prompt_field = (
            getattr(ctx, "current_prompt_field", "") or ""
        ) if ctx else ""
        choices_tuple: tuple[str, ...] = (
            getattr(ctx, "available_choices_values", ()) or ()
        ) if ctx else ()

        # Cart items: compact name list only (no prices, no raw JSON)
        cart_item_names: list[str] = []
        cart = getattr(session, "cart", None)
        if cart is not None:
            try:
                for ci in cart.get_items():
                    name = getattr(ci, "name", None) or str(ci)
                    if name:
                        cart_item_names.append(str(name))
            except Exception:
                pass

        # Previous turns (bot/user pairs from session context, last 3)
        previous_turns: list[tuple[str, str]] = []
        try:
            _get_mem = getattr(ctx, "get_turn_memory", None)
            history: list[Any] = list(_get_mem(3)) if callable(_get_mem) else []
            for turn in history:
                if isinstance(turn, (list, tuple)) and len(turn) == 2:
                    previous_turns.append((str(turn[0]), str(turn[1])))
        except Exception:
            pass

        # Local top-K intent candidates
        top_k_intents: list[dict[str, Any]] = []
        for c in (intent_candidates or getattr(nlu, "intent_candidates", ()) or ()):
            try:
                top_k_intents.append({
                    "intent": c.canonical_intent,
                    "conf": round(float(c.confidence), 4),
                })
            except Exception:
                pass

        # Local slots
        local_slots: list[dict[str, Any]] = []
        for sv in (getattr(nlu, "slots", ()) or ()):
            try:
                local_slots.append({
                    "n": getattr(sv, "name", ""),
                    "v": str(getattr(sv, "value", "")),
                })
            except Exception:
                pass

        # Required missing slots from FSM state
        required_missing: list[str] = []
        try:
            from app.nlu.semantic_repair.repair_service import _STATE_REQUIRED_SLOTS
            required_missing = list(
                _STATE_REQUIRED_SLOTS.get(state.value, ())
            )
        except Exception:
            pass

        messages = self._payload_builder.build_messages(
            state=state,
            normalized_text=normalized_text,
            current_item=current_item,
            prompt_field=prompt_field,
            local_intent=getattr(
                getattr(nlu, "effective_intent", None), "value", ""
            ) or "",
            local_confidence=float(
                getattr(nlu, "intent_confidence", 0.0) or 0.0
            ),
            top_k_intents=top_k_intents,
            local_slots=local_slots,
            choices=list(choices_tuple),
            required_missing=required_missing,
            cart_item_names=cart_item_names,
            previous_turns=previous_turns,
        )

        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        # ── 5. GPT call ──────────────────────────────────────────────────
        client = self._get_client()
        if client is None:
            return GptAddItemPlan(
                decision="no_repair",
                eligible=True,
                skipped_reason="missing_api_key",
            )

        timeout_s = cfg.add_item_timeout_ms / 1000.0
        model = cfg.model
        t_start = time.perf_counter()

        raw_response: str = ""
        completion_chars: int = 0
        timed_out: bool = False
        call_error: str | None = None

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=240,
                temperature=0.0,
                timeout=timeout_s,
            )
            raw_response = response.choices[0].message.content or ""
            completion_chars = len(raw_response)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            exc_name = type(exc).__name__.lower()
            timed_out = "timeout" in exc_name or "timedout" in exc_name
            call_error = f"{type(exc).__name__}: {exc}"[:200]

            # Retry once for transient 429 / 5xx
            if any(k in exc_name for k in ("ratelimit", "servererror", "apierror", "429", "500", "503")):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=240,
                        temperature=0.0,
                        timeout=timeout_s,
                    )
                    raw_response = response.choices[0].message.content or ""
                    completion_chars = len(raw_response)
                    call_error = None
                    timed_out = False
                except Exception:
                    pass

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if call_error and not raw_response:
            return GptAddItemPlan(
                decision="no_repair",
                eligible=True,
                timeout=timed_out,
                parse_error=call_error,
                latency_ms=latency_ms,
                total_ms=latency_ms,
                prompt_chars=prompt_chars,
                model=model,
            )

        # ── 6. Parse output ──────────────────────────────────────────────
        plan = parse_add_item_output(
            raw=raw_response,
            utterance_text=normalized_text,
            choices=choices_tuple,
            cart_item_names=tuple(cart_item_names),
            latency_ms=latency_ms,
            max_items=cfg.add_item_max_items_per_turn,
        )

        # Overlay timing / metadata not available inside the parser
        return GptAddItemPlan(
            decision=plan.decision,
            intent=plan.intent,
            items=plan.items,
            global_slots=plan.global_slots,
            missing=plan.missing,
            fallback_type=plan.fallback_type,
            confidence=plan.confidence,
            reason=plan.reason,
            parse_error=plan.parse_error,
            latency_ms=latency_ms,
            total_ms=latency_ms,
            timeout=timed_out,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars,
            model=model,
            eligible=True,
            skipped_reason=plan.skipped_reason,
            parse_notes=plan.parse_notes,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily create the OpenAI client.  Returns None if key missing."""
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None
        try:
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        except Exception:
            return None
        return self._client
