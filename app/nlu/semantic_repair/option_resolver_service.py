# app/nlu/semantic_repair/option_resolver_service.py
"""Phase 3 GPT Option Resolver Service.

Calls OpenAI to resolve modifier option names when local deterministic
matching fails in WAITING_FOR_MODIFIER (and future option states).

Shadow-only and inline contracts
---------------------------------
* SHADOW mode  — GPT is called; result is returned for logging only.
  safe_to_apply is always False in shadow mode.
* INLINE mode  — GPT is called; validator marks safe_to_apply=True only
  when all selected names exist in the group and confidence >= threshold.
* The handler decides whether to apply based on safe_to_apply and route_mode.

Safety invariants
-----------------
* API key is read from the environment at call time; never stored or logged.
* Payload contains only the current modifier group's choices — no full menu.
* Cart, intent enum, prices, PII are never included.
* A GPT error/timeout NEVER raises — always returns OptionResolverResult.
* The result is NEVER applied by this service; the handler owns application.
"""
from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

from app.nlu.semantic_repair.daily_budget import GptDailyBudget
from app.nlu.semantic_repair.option_context_builder import GptOptionContextBuilder
from app.nlu.semantic_repair.option_resolver_result import (
    OPTION_RESOLVER_NOT_CALLED,
    OptionResolverResult,
)
from app.nlu.semantic_repair.option_routing_policy import GptRoutingPolicy, OptionRouteMode
from app.nlu.semantic_repair.option_selection_validator import GptOptionSelectionValidator

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig
    from app.state_machine.models.pending_item_models import (
        ModifierSelection,
        PendingModifierGroup,
    )

# Maximum tokens allowed in the option resolver response (short by design).
_MAX_TOKENS = 120

# Allowed decision values in the GPT response.
_VALID_DECISIONS = {"select_option", "no_match"}

# Allowed reason codes (GPT may produce others; unknown codes are kept as-is).
_VALID_REASON_CODES = {"exact_match", "phonetic_match", "fuzzy_match", "no_match"}


class GptOptionResolverService:
    """Inline/shadow GPT service for resolving modifier option names.

    Instantiate once (in WaitingForModifierHandler.__init__) and reuse
    across turns.  The OpenAI client is created lazily on first use.

    run() always returns an OptionResolverResult — it never raises.
    """

    def __init__(self, config: "SemanticRepairConfig | None" = None) -> None:
        from app.config.semantic_repair import get_semantic_repair_config

        self._config: SemanticRepairConfig = config or get_semantic_repair_config()
        self._routing_policy = GptRoutingPolicy()
        self._context_builder = GptOptionContextBuilder()
        self._validator = GptOptionSelectionValidator()
        self._daily_budget = GptDailyBudget(limit=self._config.daily_budget)
        self._client: Any = None  # lazy-initialised OpenAI client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        user_text: str,
        item_name: str,
        group: "PendingModifierGroup",
        existing_selections: list["ModifierSelection"],
        local_resolved: bool,
        repeat_count: int = 0,
        previous_turns: list[tuple[str, str]] | None = None,
        last_response_key: str | None = None,
        local_slots: list[dict] | None = None,
        top_intents: list[dict] | None = None,
        has_correction_signal: bool = False,
    ) -> OptionResolverResult:
        """Attempt to resolve a modifier option using GPT.

        Parameters
        ----------
        user_text:
            Normalized customer utterance for this turn.
        item_name:
            The name of the item currently being assembled.
        group:
            The PendingModifierGroup whose choices are the allowed options.
        existing_selections:
            Already-selected ModifierSelections in this group.
        local_resolved:
            True when ModifierGroupResolver.resolve() already found selections.
            When True, routing policy returns NO_GPT immediately.
        repeat_count:
            Number of consecutive failed reprompts on this modifier field.
        previous_turns:
            Recent bot/user turn pairs for context (capped inside builder).
        last_response_key:
            Last bot response key label sent to the customer (optional context).
        local_slots:
            Local NLU slot values from the current turn (optional context).
        top_intents:
            Top-K local NLU intent candidates (optional context).
        has_correction_signal:
            True when user_text starts with a self-correction phrase such as
            "actually", "I mean", "instead", etc.  Escalates to INLINE_GPT
            even for short text when mode == "inline".

        Returns
        -------
        OptionResolverResult
            Never raises; errors are captured in the result.
        """
        cfg = self._config

        # ── 1. Routing policy ─────────────────────────────────────────────
        choice_names = GptOptionContextBuilder.extract_choice_names(group)
        route = self._routing_policy.decide(
            config=cfg,
            local_resolved=local_resolved,
            user_text=user_text,
            options_exist=bool(choice_names),
            repeat_count=repeat_count,
            has_correction=has_correction_signal,
        )

        if route == OptionRouteMode.NO_GPT:
            return OptionResolverResult(
                decision="skipped",
                skipped_reason="routing_policy_no_gpt",
                route_mode=route.value,
            )

        if not choice_names:
            return OptionResolverResult(
                decision="skipped",
                skipped_reason="no_choices",
                route_mode=route.value,
            )

        # ── 2. Daily budget ───────────────────────────────────────────────
        if not self._daily_budget.try_consume():
            return OptionResolverResult(
                decision="skipped",
                skipped_reason="daily_budget_exceeded",
                route_mode=route.value,
            )

        # ── 3. API key ────────────────────────────────────────────────────
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return OptionResolverResult(
                decision="skipped",
                skipped_reason="missing_api_key",
                route_mode=route.value,
            )

        # ── 4. Build payload ──────────────────────────────────────────────
        already_selected_names = [sel.name for sel in existing_selections if sel.name]
        messages = self._context_builder.build_messages(
            user_text=user_text,
            item_name=item_name,
            group_name=group.name or "",
            choice_names=choice_names,
            already_selected_names=already_selected_names or None,
            previous_turns=previous_turns,
            last_response_key=last_response_key,
            local_slots=local_slots,
            top_intents=top_intents,
        )
        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        # ── 5. GPT call ───────────────────────────────────────────────────
        client = self._get_client()
        if client is None:
            return OptionResolverResult(
                decision="skipped",
                skipped_reason="missing_api_key",
                route_mode=route.value,
            )

        timeout_ms = int(getattr(cfg, "option_resolver_timeout_ms", 1200))
        timeout_s = timeout_ms / 1000.0
        model = cfg.model
        t_start = time.perf_counter()

        raw_response = ""
        completion_chars = 0
        timed_out = False
        call_error: str | None = None

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=_MAX_TOKENS,
                temperature=0.0,
                timeout=timeout_s,
            )
            raw_response = response.choices[0].message.content or ""
            completion_chars = len(raw_response)
        except Exception as exc:
            exc_name = type(exc).__name__.lower()
            timed_out = "timeout" in exc_name or "timedout" in exc_name
            call_error = f"{type(exc).__name__}: {exc}"[:200]

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if call_error and not raw_response:
            return OptionResolverResult(
                decision="error",
                reason_code="timeout" if timed_out else "api_error",
                parse_error=call_error,
                gpt_called=True,
                latency_ms=latency_ms,
                route_mode=route.value,
                prompt_chars=prompt_chars,
                model=model,
            )

        # ── 6. Parse GPT response ─────────────────────────────────────────
        result = self._parse_response(
            raw=raw_response,
            route=route,
            latency_ms=latency_ms,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars,
            model=model,
        )

        # ── 7. Validate selections against the group ──────────────────────
        min_conf = float(getattr(cfg, "option_resolver_min_confidence", 0.75))
        result = self._validator.validate(
            result=result,
            group=group,
            min_confidence=min_conf,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        *,
        raw: str,
        route: OptionRouteMode,
        latency_ms: float,
        prompt_chars: int,
        completion_chars: int,
        model: str,
    ) -> OptionResolverResult:
        """Parse the raw GPT response into an OptionResolverResult."""
        raw = (raw or "").strip()

        # Strip markdown code fences if present.
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            return OptionResolverResult(
                decision="error",
                reason_code="parse_error",
                parse_error=f"JSONDecodeError: {exc}"[:200],
                gpt_called=True,
                latency_ms=latency_ms,
                route_mode=route.value,
                prompt_chars=prompt_chars,
                completion_chars=completion_chars,
                model=model,
            )

        if not isinstance(data, dict):
            return OptionResolverResult(
                decision="error",
                reason_code="parse_error",
                parse_error="Response is not a JSON object",
                gpt_called=True,
                latency_ms=latency_ms,
                route_mode=route.value,
                prompt_chars=prompt_chars,
                completion_chars=completion_chars,
                model=model,
            )

        decision = str(data.get("decision", "no_match")).strip()
        if decision not in _VALID_DECISIONS:
            decision = "no_match"

        raw_names = data.get("selected_names", [])
        selected_names: tuple[str, ...] = ()
        if isinstance(raw_names, list):
            selected_names = tuple(
                str(n).strip() for n in raw_names if n and str(n).strip()
            )

        confidence = 0.0
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            pass

        raw_reason = str(data.get("reason_code", "")).strip()
        reason_code = raw_reason if raw_reason else "no_match"

        # Shadow mode result is never safe to apply — validator will also
        # enforce this, but set it explicitly here for clarity.
        safe_to_apply = False

        return OptionResolverResult(
            decision=decision,
            selected_names=selected_names,
            confidence=confidence,
            reason_code=reason_code,
            safe_to_apply=safe_to_apply,
            gpt_called=True,
            latency_ms=latency_ms,
            route_mode=route.value,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars,
            model=model,
        )

    def _get_client(self) -> Any:
        """Lazily create and return the OpenAI client. Returns None on failure."""
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
