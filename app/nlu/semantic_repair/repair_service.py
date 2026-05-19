# app/nlu/semantic_repair/repair_service.py
# TODO(priority-2-migration): Migrate call_gpt_for_shadow() to use GptSafeClient
#   (app/nlu/turn_resolver/gpt_safe_client.py) so circuit-breaker and structured
#   failure results are applied consistently across all GPT call sites.
"""RepairPolicy eligibility check and GptRepairService shadow-mode caller.

Phase / call_mode contract
--------------------------
call_mode=None (legacy)  → behavior driven by ``phase``:
  phase < 2  → no GPT call; GPT_NOT_CALLED returned.
  phase >= 2 → GPT called when RepairPolicy is eligible; result logged, never applied.

call_mode="disabled"     → no GPT call regardless of phase.
call_mode="eligible_only"→ GPT called when RepairPolicy is eligible; result logged,
                           never applied.  Requires OPENAI_API_KEY + daily budget.
call_mode="all_shadow"   → returns (analysis, GPT_NOT_CALLED) immediately so
                           TurnEngine can dispatch the actual GPT call to a
                           background thread after the response is built.
call_mode="all_apply_safe"→ stub only; treated as disabled in Step 1.

In all modes, intent_result, slots, state, cart, and response are NEVER mutated.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from app.config.semantic_repair import SemanticRepairConfig, get_semantic_repair_config
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_result import IntentCandidate, NLUResult, SlotValue
from app.nlu.semantic_repair.daily_budget import GptDailyBudget
from app.nlu.semantic_repair.gpt_repair_result import GPT_NOT_CALLED, GptRepairResult
from app.nlu.semantic_repair.output_parser import parse_output
from app.nlu.semantic_repair.prompt_builder import build_messages, get_candidates
from app.state_machine.models.conversation_state import ConversationState

if TYPE_CHECKING:
    from app.session.session import Session


# FSM states where turn should be treated as "completing something we're waiting for"
_WAITING_STATES = {
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
}

# States where GPT should never be called (terminal / transfer / error)
_SKIP_STATES = {
    ConversationState.COMPLETED,
    ConversationState.TRANSFERRING_TO_HUMAN_AGENT,
}
# String values for states that should be skipped in all_shadow mode
# (includes error_recovery which may or may not be in ConversationState depending on build)
_ALL_SHADOW_SKIP_STATE_VALUES: frozenset[str] = frozenset({
    "completed",
    "transferring_to_human_agent",
    "error_recovery",
})

# Cancel/done/remove signal words that indicate the user wants to exit a waiting state
_EXIT_WAITING_PHRASES = {
    "cancel",
    "never mind",
    "nevermind",
    "remove",
    "done",
    "finish",
    "that's all",
    "thats all",
    "checkout",
    "stop",
}

# Slot required by each waiting state — used to detect when local NLU missed it
_STATE_REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    ConversationState.WAITING_FOR_MODIFIER.value: ("MODIFIER",),
    ConversationState.WAITING_FOR_SIDE.value: ("SIDE",),
    ConversationState.WAITING_FOR_SIZE.value: ("SIZE",),
    ConversationState.WAITING_FOR_SIDE_SIZE.value: ("SIZE",),
    ConversationState.WAITING_FOR_QUANTITY.value: ("QUANTITY",),
}

# Minimum gap between top-1 and top-2 confidence: below this triggers GPT
_CONFIDENCE_GAP_THRESHOLD = 0.20


@dataclass(frozen=True)
class LocalTurnAnalysis:
    """Eligibility snapshot built by RepairPolicy.check().

    Stamped onto every TurnEvent so Phase 0 also produces useful data.
    """

    # Core eligibility
    gpt_repair_eligible: bool
    reason: str
    candidate_count: int
    candidates: frozenset[str]

    # Intent context at check time
    intent_effective: str = ""
    intent_confidence: float = 0.0
    intent_candidates: tuple[IntentCandidate, ...] = ()

    # FSM / turn context
    state_before: str = ""
    customer_text: str = ""
    normalized_text: str = ""

    # Slots extracted by local NLU
    slots: tuple[SlotValue, ...] = ()

    # Curated candidate sets from prompt_builder
    candidate_repair_intents: frozenset[str] = field(default_factory=frozenset)
    candidate_control_kinds: frozenset[str] = field(default_factory=frozenset)

    # Cart / session context (safe summary only — no PII)
    cart_summary: dict[str, Any] | None = None
    repeat_count: int = 0
    last_response_key: str = ""
    engine_elapsed_ms: float = 0.0

    # Why GPT was skipped when eligible=False
    skipped_reason: str = ""

    # Required slots that are missing given current FSM state
    required_missing: tuple[str, ...] = ()

    # Available choice labels for the current side/modifier group
    choices: tuple[str, ...] = ()

    # Last N (role, text) turns for GPT context — capped to 3 before passing to GPT
    previous_turns: tuple[tuple[str, str], ...] = ()



class RepairPolicy:
    """Decides whether a turn is a GPT repair candidate.

    Eligibility triggers (any one → eligible):
    - intent is UNKNOWN and text is long enough
    - top1-top2 confidence gap < _CONFIDENCE_GAP_THRESHOLD
    - in a waiting state + UNKNOWN intent
    - user says a cancel/done/remove phrase while waiting
    - last_response_key was "intent_not_allowed" or session.fallback_count >= 2

    Skip conditions (any one → ineligible):
    - terminal / transfer states
    - text too short
    - trivial candidate set (< 2 candidates)
    - high-confidence accepted route (>= 0.85 and intent is known)
    """

    _MIN_TEXT_LEN: int = 3
    _MIN_CANDIDATE_COUNT: int = 2
    _HIGH_CONF_THRESHOLD: float = 0.85

    def check(
        self,
        *,
        nlu: NLUResult,
        intent_result: IntentResult,
        state: ConversationState,
        session: "Session | None" = None,
    ) -> LocalTurnAnalysis:
        """Return a LocalTurnAnalysis describing eligibility."""
        candidates = get_candidates(state.value)

        # Build shared context fields
        last_response_key = ""
        repeat_count = 0
        cart_summary: dict[str, Any] | None = None

        if session is not None:
            last_response_key = session.last_response_key or ""
            repeat_count = getattr(session, "fallback_count", 0)
            try:
                cart_items = session.cart.get_items() if session.cart else []
                cart_summary = {
                    "count": len(cart_items),
                    "items": [getattr(item, "name", str(item)) for item in cart_items],
                }
            except Exception:
                cart_summary = None

        # Extract choices and conversation history from session context
        choices: tuple[str, ...] = ()
        previous_turns: tuple[tuple[str, str], ...] = ()

        if session is not None:
            _ctx = getattr(session, "conversation_context", None)
            if _ctx is not None:
                choices = tuple(getattr(_ctx, "available_choices_values", ()) or ())
                _get_mem = getattr(_ctx, "get_turn_memory", None)
                if callable(_get_mem):
                    previous_turns = _ctx.get_turn_memory(3)

        # Infer required-but-missing slots from FSM state and local extraction
        _extracted_slot_names = frozenset(
            (getattr(sv, "name", "") or "").upper()
            for sv in (getattr(nlu, "slots", ()) or ())
            if getattr(sv, "name", "")
        )
        _state_required = _STATE_REQUIRED_SLOTS.get(state.value, ())
        required_missing = tuple(s for s in _state_required if s not in _extracted_slot_names)

        def _make(eligible: bool, reason: str, skipped: str = "") -> LocalTurnAnalysis:
            return LocalTurnAnalysis(
                gpt_repair_eligible=eligible,
                reason=reason,
                candidate_count=len(candidates) if eligible else 0,
                candidates=candidates if eligible else frozenset(),
                intent_effective=intent_result.intent.value,
                intent_confidence=getattr(nlu, "intent_confidence", 0.0),
                intent_candidates=getattr(nlu, "intent_candidates", ()),
                state_before=state.value,
                customer_text=getattr(nlu, "raw_text", ""),
                normalized_text=getattr(nlu, "normalized_text", ""),
                slots=tuple(getattr(nlu, "slots", ()) or ()),
                candidate_repair_intents=candidates,
                candidate_control_kinds=frozenset({"cancel", "confirm", "deny"}),
                cart_summary=cart_summary,
                repeat_count=repeat_count,
                last_response_key=last_response_key,
                skipped_reason=skipped,
                required_missing=required_missing,
                choices=choices,
                previous_turns=previous_turns,
            )

        # --- Skip conditions (checked first) ---

        if state in _SKIP_STATES:
            return _make(False, "terminal_state", skipped="terminal_state")

        text = (getattr(nlu, "normalized_text", "") or "").strip()
        if len(text) < self._MIN_TEXT_LEN:
            return _make(False, "text_too_short", skipped="text_too_short")

        if len(candidates) < self._MIN_CANDIDATE_COUNT:
            return _make(False, "trivial_candidate_set", skipped="trivial_candidate_set")

        # High-confidence known intent → skip (low benefit, don't waste budget)
        intent_conf = getattr(nlu, "intent_confidence", 0.0) or 0.0
        if (
            intent_result.intent != Intent.UNKNOWN
            and intent_conf >= self._HIGH_CONF_THRESHOLD
        ):
            return _make(False, "intent_known", skipped="high_confidence")

        # --- Eligibility triggers ---

        # 1. Unknown intent with sufficient text
        if intent_result.intent == Intent.UNKNOWN:
            return _make(True, "unknown_intent_with_text")

        # 2. Known intent but low confidence gap between top-1 and top-2
        intent_candidates = tuple(getattr(nlu, "intent_candidates", ()) or ())
        if len(intent_candidates) >= 2:
            gap = intent_candidates[0].confidence - intent_candidates[1].confidence
            if gap < _CONFIDENCE_GAP_THRESHOLD:
                return _make(True, "low_confidence_gap")

        # 3. User says cancel/exit while in a waiting state
        if state in _WAITING_STATES:
            for phrase in _EXIT_WAITING_PHRASES:
                if phrase in text:
                    return _make(True, "exit_phrase_in_waiting_state")

        # 4. Previous turn was intent_not_allowed or high fallback count
        if last_response_key == "intent_not_allowed":
            return _make(True, "previous_intent_not_allowed")

        if repeat_count >= 2:
            return _make(True, "high_fallback_count")

        # 5. Required slot missing for the current waiting state
        if required_missing:
            return _make(True, "required_slot_missing")

        # 6. Known intent but it's a low-confidence case not caught above
        if intent_result.intent != Intent.UNKNOWN:
            return _make(False, "intent_known", skipped="intent_known")

        return _make(True, "unknown_intent_with_text")


class GptRepairService:
    """Shadow-mode GPT repair caller.

    Construction is cheap (no network, no model load).  The OpenAI client is
    created lazily on the first eligible call.

    The API key is read from ``OPENAI_API_KEY`` at call time — it is never
    stored on ``self``, never logged, never serialised.
    """

    def __init__(self, config: SemanticRepairConfig | None = None) -> None:
        self._config = config or get_semantic_repair_config()
        self._policy = RepairPolicy()
        self._client = None  # lazy-initialised openai.OpenAI instance
        self._daily_budget = GptDailyBudget(self._config.daily_budget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        nlu: NLUResult,
        intent_result: IntentResult,
        state: ConversationState,
        session: "Session | None" = None,
        engine_elapsed_ms: float = 0.0,
    ) -> tuple[LocalTurnAnalysis, GptRepairResult]:
        """Run policy evaluation + optional GPT call.

        This method NEVER mutates intent_result, nlu, state, cart, or response.
        """
        analysis = self._policy.check(
            nlu=nlu,
            intent_result=intent_result,
            state=state,
            session=session,
        )

        # Phase 2 call-mode routing — no Phase 3 execution policy.
        effective_mode = self._config.effective_call_mode
        if effective_mode == "all_shadow":
            # TurnEngine dispatches the GPT call to a background thread.
            return analysis, GPT_NOT_CALLED

        if effective_mode not in ("eligible_only",):
            return analysis, GPT_NOT_CALLED

        if not analysis.gpt_repair_eligible:
            return analysis, GPT_NOT_CALLED

        # SLO budget (Phase 2 feature — skip if engine already used its budget).
        slo_ms = self._config.slo_budget_ms
        if slo_ms > 0 and engine_elapsed_ms > slo_ms:
            return replace(analysis, skipped_reason="slo_budget_exceeded"), GPT_NOT_CALLED

        if not self._daily_budget.try_consume():
            return replace(analysis, skipped_reason="daily_budget_exceeded"), GPT_NOT_CALLED

        return analysis, self._call_gpt(
            nlu=nlu,
            analysis=analysis,
            state=state,
            session=session,
        )

    def call_gpt_for_shadow(


        self,
        *,
        nlu: NLUResult,
        analysis: LocalTurnAnalysis,
        state: ConversationState,
        prompt_field: str = "",
        item_name: str = "",
        timeout_seconds: float | None = None,
    ) -> GptRepairResult:
        """Execute a GPT call for all_shadow background tasks.

        Accepts an already-built LocalTurnAnalysis (captured synchronously before
        the response was returned) so no live session access is needed.

        Returns GPT_NOT_CALLED on budget exceeded, missing key, or any error.
        This method NEVER mutates any external state.
        """
        # Final budget check (consumed in background thread)
        if not self._daily_budget.try_consume():
            return GptRepairResult(
                decision="no_repair",
                skipped_reason="daily_budget_exceeded",
                applied=False,
            )

        # Use shadow-specific timeout if provided
        effective_timeout: float
        if timeout_seconds is not None:
            effective_timeout = timeout_seconds
        else:
            effective_timeout = self._config.shadow_timeout_ms / 1000.0

        return self._call_gpt(
            nlu=nlu,
            analysis=analysis,
            state=state,
            session=None,
            _prompt_field_override=prompt_field,
            _item_name_override=item_name,
            _timeout_override=effective_timeout,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_gpt(
        self,
        *,
        nlu: NLUResult,
        analysis: LocalTurnAnalysis,
        state: ConversationState,
        session: "Session | None" = None,
        _prompt_field_override: str | None = None,
        _item_name_override: str | None = None,
        _timeout_override: float | None = None,
    ) -> GptRepairResult:
        t_total = time.perf_counter()

        # Build prompt
        t0 = time.perf_counter()
        current_prompt_field = _prompt_field_override if _prompt_field_override is not None else ""
        current_item_name = _item_name_override if _item_name_override is not None else ""
        if session is not None and _prompt_field_override is None:
            ctx = getattr(session, "conversation_context", None)
            if ctx is not None:
                current_prompt_field = getattr(ctx, "current_prompt_field", "") or ""
                current_item_name = getattr(ctx, "current_item_name", "") or ""

        # Use candidates from RepairPolicy (Phase 2 only — no Phase 3 routing override)
        candidates = analysis.candidates
        if not candidates:
            candidates = get_candidates(state.value)

        messages = build_messages(
            utterance=nlu.normalized_text or "",
            state_name=state.value,
            candidates=candidates,
            current_prompt_field=current_prompt_field,
            current_item_name=current_item_name,
            intent_candidates=analysis.intent_candidates,
            cart_summary=analysis.cart_summary,
            repeat_count=analysis.repeat_count,
            slots=analysis.slots,
            selected_intent=analysis.intent_effective,
            selected_confidence=analysis.intent_confidence,
            choices=analysis.choices,
            required_missing=analysis.required_missing,
            previous_turns=analysis.previous_turns,
            prompt_bucket=None,
        )
        payload_build_ms = (time.perf_counter() - t0) * 1000.0
        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        # Determine effective timeout
        timeout_sec = _timeout_override if _timeout_override is not None else self._config.timeout_seconds

        # Check API key before starting the request timer.
        client = self._get_client()
        if client is None:
            total_ms = (time.perf_counter() - t_total) * 1000.0
            return GptRepairResult(
                decision="no_repair",
                skipped_reason="missing_api_key",
                applied=False,
                payload_build_ms=payload_build_ms,
                prompt_chars=prompt_chars,
                model=self._config.model,
                total_ms=total_ms,
            )

        t_req = time.perf_counter()
        raw: str | None = None
        last_exc: Exception | None = None

        for attempt in range(2):  # attempt 0 = first try; attempt 1 = one retry
            try:
                response = client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    timeout=timeout_sec,
                    max_tokens=256,
                    temperature=0.0,
                )
                raw = (response.choices[0].message.content or "").strip()
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                exc_name = type(exc).__name__
                is_transient = (
                    "RateLimitError" in exc_name
                    or "InternalServerError" in exc_name
                    or "APIStatusError" in exc_name
                )
                if not is_transient or attempt >= 1:
                    break

        request_ms = (time.perf_counter() - t_req) * 1000.0

        if last_exc is not None:
            total_ms = (time.perf_counter() - t_total) * 1000.0
            exc_name = type(last_exc).__name__
            is_timeout = "Timeout" in exc_name or "timeout" in exc_name.lower()
            exc_str = str(last_exc)
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key and api_key in exc_str:
                exc_str = exc_str.replace(api_key, "[REDACTED]")
            return GptRepairResult(
                decision="no_repair",
                timeout=is_timeout,
                parse_error=None if is_timeout else f"call_error:{exc_name}",
                latency_ms=request_ms,
                payload_build_ms=payload_build_ms,
                request_ms=request_ms,
                total_ms=total_ms,
                prompt_chars=prompt_chars,
                model=self._config.model,
                applied=False,
            )

        completion_chars = len(raw) if raw else 0

        t_parse = time.perf_counter()
        result = parse_output(
            raw=raw or "",
            candidates=candidates,
            nlu=nlu,
            latency_ms=request_ms,
        )
        parse_ms = (time.perf_counter() - t_parse) * 1000.0
        total_ms = (time.perf_counter() - t_total) * 1000.0

        return GptRepairResult(
            decision=result.decision,
            repaired_intent=result.repaired_intent,
            repaired_control_intent=result.repaired_control_intent,
            slot_corrections=result.slot_corrections,
            slot_corrections_list=result.slot_corrections_list,
            confidence=result.confidence,
            reason=result.reason,
            latency_ms=result.latency_ms,
            timeout=False,
            parse_error=result.parse_error,
            applied=False,
            payload_build_ms=payload_build_ms,
            request_ms=request_ms,
            parse_ms=parse_ms,
            total_ms=total_ms,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars,
            model=self._config.model,
            fallback_type=result.fallback_type,
            missing_slots=result.missing_slots,
            items=result.items,
        )

    def _get_client(self):
        """Return the shared openai.OpenAI client, or None if API key is absent.

        The API key is read here at call time — never stored as an attribute,
        never logged.  Returns None (instead of raising) when the key is missing
        so callers can emit a clean no_repair result.
        """
        if self._client is not None:
            return self._client
        # Check key first — avoids importing openai when the key is absent.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None  # Caller emits no_repair with skipped_reason="missing_api_key"
        try:
            import openai  # noqa: PLC0415  # lazy import — openai is optional
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed. "
                "Add openai>=1.0.0 to requirements and reinstall."
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)
        return self._client
