# app/nlu/turn_resolver/final_turn_decision_resolver.py
"""Final turn decision resolver — chooses between local NLU and GPT for a turn.

Architecture
------------
``resolve()`` is the top-level entry point.  It:
  1. Calls ``pick_bucket()`` to classify the turn.
  2. Checks the circuit breaker — skips GPT if provider is unhealthy.
  3. If a bucket applies and mode allows, delegates to the corresponding
     existing GPT service (or builds a stub result for Bucket 0).
  4. Calls ``validate_gpt_result()`` to check the GPT output.
  5. Returns a ``FinalTurnDecision`` with ``source="gpt"`` only when all
     gates pass and the mode is "inline".  Shadow mode always returns
     ``source="local"`` (``apply_gpt=False``).

Safety contract
---------------
  * GPT never mutates cart, state, or customer-facing response directly.
  * ``apply_gpt=True`` only when: bucket found, mode=="inline", validation
    passes, and ``safe_to_apply=True`` from validators.
  * In shadow mode the GPT result is logged only; source remains "local".
  * If GPT times out / errors / circuit is open, the local path is used.
  * ``FinalTurnDecision.gpt_status`` always reflects why GPT was skipped /
    failed so callers can build appropriate fallback responses.
  * Never raises — all exceptions are caught and logged; local path wins.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Collection, Sequence

from app.nlu.turn_resolver.bucket_policy import (
    BUCKET_IDLE_ITEM,
    BUCKET_MULTI_ITEM,
    BUCKET_OPTION,
    pick_bucket,
)
from app.nlu.turn_resolver.context_builder import GptTurnContextPacket, build_context_packet
from app.nlu.turn_resolver.gpt_circuit_breaker import DEFAULT_CIRCUIT_BREAKER, GptCircuitBreaker
from app.nlu.turn_resolver.gpt_safe_client import GptCallStatus
from app.nlu.turn_resolver.schemas import GPT_TURN_RESOLUTION_SKIPPED, GptTurnResolution
from app.nlu.turn_resolver.validators import validate_gpt_result

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig
    from app.nlu.intent_resolution.intent import Intent
    from app.nlu.nlu_result import NLUResult, SlotValue
    from app.state_machine.models.conversation_context import ConversationContext
    from app.state_machine.models.conversation_state import ConversationState

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinalTurnDecision:
    """Outcome of one turn-resolution cycle.

    Fields
    ------
    source:
        "local"  — local deterministic NLU is used (default / safe)
        "gpt"    — GPT output was validated and is safe to apply
    bucket:
        Which GPT bucket was triggered (None when source=="local" and no
        bucket was attempted, or when GPT was shadowed).
    local_intent:
        Effective intent from local NLU.  Always valid.
    local_slots:
        Slots from local NLU.  Always valid.
    local_confidence:
        Intent confidence from local NLU.
    gpt_result:
        The GptTurnResolution produced (None when no GPT call was made).
    apply_gpt:
        True only when source=="gpt" and validation passed in inline mode.
        False in shadow mode even when GPT resolved something useful.
    reason:
        Short code explaining the decision (for logging/diagnostics).

    Failure isolation fields (always present — defaults to safe values)
    -------------------------------------------------------------------
    gpt_status:
        ``GptCallStatus`` constant for why GPT failed, or "ok" / "not_called".
    local_is_safe:
        True when the local NLU result is safe to execute without GPT assist.
    gpt_circuit_open:
        True when the circuit breaker was open and GPT was skipped.
    fallback_used:
        True when a state-specific clarification was issued instead of the
        local or GPT path.
    """

    source: str
    bucket: str | None
    local_intent: "Intent"
    local_slots: "tuple[SlotValue, ...]"
    local_confidence: float
    gpt_result: GptTurnResolution | None
    apply_gpt: bool
    reason: str
    # Failure isolation metadata
    gpt_status: str = GptCallStatus.OK
    local_is_safe: bool = True
    gpt_circuit_open: bool = False
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------


def _make_local_decision(
    local_intent: "Intent",
    local_slots: "tuple[SlotValue, ...]",
    local_confidence: float,
    *,
    bucket: str | None = None,
    gpt_result: GptTurnResolution | None = None,
    reason: str = "local_path",
    gpt_status: str = GptCallStatus.OK,
    local_is_safe: bool = True,
    gpt_circuit_open: bool = False,
    fallback_used: bool = False,
) -> FinalTurnDecision:
    return FinalTurnDecision(
        source="local",
        bucket=bucket,
        local_intent=local_intent,
        local_slots=local_slots,
        local_confidence=local_confidence,
        gpt_result=gpt_result,
        apply_gpt=False,
        reason=reason,
        gpt_status=gpt_status,
        local_is_safe=local_is_safe,
        gpt_circuit_open=gpt_circuit_open,
        fallback_used=fallback_used,
    )


# Sentinel returned when no bucket was found and no GPT call was attempted.
def _local_no_bucket(local_nlu: "NLUResult") -> FinalTurnDecision:
    return _make_local_decision(
        local_nlu.effective_intent,
        local_nlu.slots,
        local_nlu.intent_confidence,
        reason="no_bucket",
        gpt_status=GptCallStatus.DISABLED,
    )


# Public sentinel for callers that need a "no GPT attempted" decision object
# without a full NLUResult (e.g., in tests or no-op paths).
FINAL_DECISION_LOCAL = FinalTurnDecision(
    source="local",
    bucket=None,
    local_intent=None,  # type: ignore[arg-type]
    local_slots=(),
    local_confidence=0.0,
    gpt_result=None,
    apply_gpt=False,
    reason="no_bucket",
    gpt_status=GptCallStatus.DISABLED,
)


# ---------------------------------------------------------------------------
# Stub GPT result for Bucket 0 (not yet backed by a real GPT service)
# ---------------------------------------------------------------------------


def _call_bucket0_stub(
    packet: GptTurnContextPacket,
    timeout_ms: int,
) -> GptTurnResolution:
    """Stub Bucket 0 call — returns skipped until the service is implemented.

    Replace this with a real GPT service call when Bucket 0 is backed by
    a production GPT service.  The stub ensures the framework is wired
    correctly and tests can exercise the full decision path without a
    live GPT call.
    """
    return GptTurnResolution(
        bucket=BUCKET_IDLE_ITEM,
        decision="skipped",
        gpt_called=False,
        skipped_reason="bucket0_not_yet_implemented",
    )


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------


def _bucket_mode(config: "SemanticRepairConfig", bucket: str) -> str:
    """Return the mode string for the given bucket from config."""
    attr_map = {
        BUCKET_IDLE_ITEM: "bucket_0_mode",
        BUCKET_OPTION: "bucket_2_mode",
        BUCKET_MULTI_ITEM: "bucket_3_mode",
    }
    attr = attr_map.get(bucket, "")
    return str(getattr(config, attr, "disabled") or "disabled")


def _bucket_timeout_ms(config: "SemanticRepairConfig") -> int:
    return int(getattr(config, "bucket_timeout_ms", 1200) or 1200)


# ---------------------------------------------------------------------------
# GPT dispatch per bucket
# ---------------------------------------------------------------------------


def _call_gpt_for_bucket(
    bucket: str,
    packet: GptTurnContextPacket,
    config: "SemanticRepairConfig",
) -> GptTurnResolution:
    """Call the appropriate GPT service for the given bucket.

    Delegates to the existing specialized services:
      Bucket 0 → stub (to be replaced with a real Bucket 0 service)
      Bucket 2 → GptOptionResolverService (via option_resolver_service)
      Bucket 3 → GptAddItemPlannerService (via add_item_planner_service)

    Always returns a GptTurnResolution — never raises.
    """
    timeout_ms = _bucket_timeout_ms(config)

    try:
        if bucket == BUCKET_IDLE_ITEM:
            return _call_bucket0_stub(packet, timeout_ms)

        if bucket == BUCKET_OPTION:
            # Bucket 2 delegates to GptOptionResolverService.
            # The result is adapted to GptTurnResolution.
            from app.nlu.semantic_repair.option_resolver_service import (
                GptOptionResolverService,
            )
            svc = GptOptionResolverService()
            opt_result = svc.run(
                user_text=packet.user_text,
                item_name="",  # not available in bucket context; service handles gracefully
                group=None,  # type: ignore[arg-type]  # caller must pass group via choices
                existing_selections=[],
                local_resolved=False,
                repeat_count=0,
                previous_turns=list(packet.previous_turns),
            )
            return GptTurnResolution(
                bucket=BUCKET_OPTION,
                decision=opt_result.decision,
                control_intent=opt_result.control_intent,
                selected_option_names=opt_result.selected_names,
                confidence=opt_result.confidence,
                reason_code=opt_result.reason_code,
                safe_to_apply=False,  # validator sets this
                latency_ms=opt_result.latency_ms,
                gpt_called=opt_result.gpt_called,
                skipped_reason=opt_result.skipped_reason,
                parse_error=opt_result.parse_error,
                prompt_chars=opt_result.prompt_chars,
                completion_chars=opt_result.completion_chars,
                model=opt_result.model,
            )

        if bucket == BUCKET_MULTI_ITEM:
            # Bucket 3 delegates to GptAddItemPlannerService.
            from app.nlu.semantic_repair.add_item_planner_service import (
                GptAddItemPlannerService,
            )
            svc3 = GptAddItemPlannerService()
            plan_result = svc3.run(
                user_text=packet.user_text,
                local_intent=packet.local_intent,
                local_confidence=packet.local_confidence,
                local_slots=list(packet.local_slots),
                item_candidates=[],
            )
            # Adapt items from PlannerGptItem to ResolvedItemPlan
            from app.nlu.turn_resolver.schemas import ResolvedItemPlan, ResolvedModifierPlan, ResolvedSidePlan
            resolved_items = tuple(
                ResolvedItemPlan(
                    item_name=it.item_name,
                    quantity=it.quantity or 1,
                    size=it.size,
                    variant=it.variant,
                    sides=tuple(
                        ResolvedSidePlan(name=s.name, quantity=s.quantity, size=s.size)
                        for s in it.sides
                    ),
                    modifiers=tuple(
                        ResolvedModifierPlan(name=m.name, operation=m.operation)
                        for m in it.modifiers
                    ),
                )
                for it in plan_result.items
            )
            return GptTurnResolution(
                bucket=BUCKET_MULTI_ITEM,
                decision=plan_result.decision,
                items=resolved_items,
                confidence=plan_result.confidence,
                reason_code=plan_result.reason_code,
                safe_to_apply=False,
                latency_ms=plan_result.latency_ms,
                gpt_called=plan_result.gpt_called,
                skipped_reason=plan_result.skipped_reason,
                parse_error=plan_result.parse_error,
                prompt_chars=plan_result.prompt_chars,
                completion_chars=plan_result.completion_chars,
                model=plan_result.model,
            )

    except Exception as exc:
        _logger.warning("gpt_turn_resolver_bucket_error: bucket=%s err=%s", bucket, exc)
        return GptTurnResolution(
            bucket=bucket,
            decision="error",
            gpt_called=False,
            parse_error=str(exc)[:200],
        )

    return GPT_TURN_RESOLUTION_SKIPPED


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def _log_resolver_result(
    bucket: str | None,
    gpt_result: GptTurnResolution | None,
    decision: FinalTurnDecision,
    elapsed_ms: float,
) -> None:
    _logger.info(
        "gpt_turn_resolver_result",
        extra={
            "event": "gpt_turn_resolver_invoked",
            "gpt_bucket": bucket,
            "gpt_decision": gpt_result.decision if gpt_result else "none",
            "gpt_called": gpt_result.gpt_called if gpt_result else False,
            "gpt_confidence": gpt_result.confidence if gpt_result else None,
            "gpt_safe_to_apply": gpt_result.safe_to_apply if gpt_result else False,
            "gpt_status": decision.gpt_status,
            "gpt_failure_type": decision.gpt_status if decision.gpt_status != GptCallStatus.OK else None,
            "gpt_latency_ms": gpt_result.latency_ms if gpt_result else None,
            "gpt_timeout_ms": None,  # populated by caller if known
            "gpt_circuit_open": decision.gpt_circuit_open,
            "gpt_fallback_used": decision.fallback_used,
            "fallback_source": "gpt" if decision.apply_gpt else ("local" if decision.local_is_safe else "state_clarification"),
            "local_used_after_gpt_failure": (
                decision.source == "local"
                and decision.gpt_status not in {GptCallStatus.OK, GptCallStatus.DISABLED}
            ),
            "local_safe_after_gpt_failure": decision.local_is_safe,
            "unsafe_local_slots_blocked": not decision.local_is_safe and not decision.apply_gpt,
            "final_source": decision.source,
            "apply_gpt": decision.apply_gpt,
            "repair_type": bucket or "none",
            "resolver_reason": decision.reason,
            "resolver_elapsed_ms": round(elapsed_ms, 1),
        },
    )


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------


def resolve(
    state: "ConversationState",
    local_nlu: "NLUResult",
    context: "ConversationContext",
    config: "SemanticRepairConfig",
    *,
    option_match_failed: bool = False,
    reprompt_count: int = 0,
    choices: Sequence[str] = (),
    cart_item_names: Sequence[str] = (),
    allowed_intents: Sequence[str] = (),
    known_item_names: Sequence[str] = (),
    circuit_breaker: GptCircuitBreaker | None = None,
) -> FinalTurnDecision:
    """Choose the final intent/slot source for the current turn.

    Parameters
    ----------
    state:
        Current conversation state.
    local_nlu:
        NLU result from the local deterministic model.
    context:
        Current conversation context (for prompt field, turn memory).
    config:
        Semantic repair config (for mode flags and timeouts).
    option_match_failed:
        True when the calling waiting handler's local matcher found nothing.
        Required for Bucket 2 eligibility.
    reprompt_count:
        Number of previous reprompts for the current field.
    choices:
        Option names from the current modifier/side group (Bucket 2).
    cart_item_names:
        Item names from the current cart (Bucket 0/3 context).
    allowed_intents:
        Valid intent strings for this state (Bucket 0 intent gate).
    known_item_names:
        Known menu item names for Bucket 3 menu-match validation.
    circuit_breaker:
        Override for the module-level DEFAULT_CIRCUIT_BREAKER (for tests).

    Returns
    -------
    ``FinalTurnDecision`` — always valid, never raises.
    Callers must check ``gpt_status``, ``local_is_safe``, and ``fallback_used``
    to determine if a state-specific clarification response is needed.
    """
    from app.nlu.turn_resolver.gpt_fallback_policy import is_local_result_safe

    t0 = time.monotonic()
    breaker = circuit_breaker or DEFAULT_CIRCUIT_BREAKER

    try:
        bucket = pick_bucket(
            state=state,
            local_intent=local_nlu.effective_intent,
            local_confidence=local_nlu.intent_confidence,
            local_slots=local_nlu.slots,
            user_text=local_nlu.normalized_text,
            option_match_failed=option_match_failed,
            reprompt_count=reprompt_count,
            config=config,
        )

        # Assess local result safety regardless of whether GPT is called —
        # callers need this to decide whether to use local slots or clarify.
        _local_safe, _unsafe_reason = is_local_result_safe(
            state=state,
            local_intent_value=local_nlu.effective_intent.value,
            local_slots=local_nlu.slots,
            local_confidence=local_nlu.intent_confidence,
        )

        if bucket is None:
            return _local_no_bucket(local_nlu)

        mode = _bucket_mode(config, bucket)
        if mode == "disabled":
            return _make_local_decision(
                local_nlu.effective_intent,
                local_nlu.slots,
                local_nlu.intent_confidence,
                bucket=bucket,
                reason="bucket_mode_disabled",
                gpt_status=GptCallStatus.DISABLED,
                local_is_safe=_local_safe,
            )

        model = getattr(config, "model", None)
        circuit_key = breaker.circuit_key(model, bucket)

        # ── Circuit breaker check ────────────────────────────────────────────
        if breaker.is_open(circuit_key):
            elapsed_ms = (time.monotonic() - t0) * 1000
            decision = _make_local_decision(
                local_nlu.effective_intent,
                local_nlu.slots,
                local_nlu.intent_confidence,
                bucket=bucket,
                reason="circuit_open",
                gpt_status=GptCallStatus.CIRCUIT_OPEN,
                local_is_safe=_local_safe,
                gpt_circuit_open=True,
            )
            _log_resolver_result(bucket, None, decision, elapsed_ms)
            return decision

        # Build compact context packet (PII-free)
        packet = build_context_packet(
            bucket=bucket,
            user_text=local_nlu.normalized_text,
            state=state,
            context=context,
            local_nlu=local_nlu,
            choices=choices,
            cart_item_names=cart_item_names,
            allowed_intents=allowed_intents,
        )

        # Call the appropriate GPT service
        gpt_result = _call_gpt_for_bucket(bucket, packet, config)

        # Classify GPT result status
        gpt_status = _classify_gpt_result_status(gpt_result)

        # Update circuit breaker based on result
        if gpt_status in {
            GptCallStatus.TIMEOUT,
            GptCallStatus.RATE_LIMITED,
            GptCallStatus.PROVIDER_ERROR,
            GptCallStatus.NETWORK_ERROR,
        }:
            breaker.record_failure(circuit_key)
        elif gpt_result.gpt_called and gpt_status == GptCallStatus.OK:
            breaker.record_success(circuit_key)

        # Handle GPT failure: classify and fall back
        if gpt_status not in {GptCallStatus.OK}:
            elapsed_ms = (time.monotonic() - t0) * 1000
            decision = _make_local_decision(
                local_nlu.effective_intent,
                local_nlu.slots,
                local_nlu.intent_confidence,
                bucket=bucket,
                gpt_result=gpt_result,
                reason=f"gpt_failed:{gpt_status}",
                gpt_status=gpt_status,
                local_is_safe=_local_safe,
            )
            _log_resolver_result(bucket, gpt_result, decision, elapsed_ms)
            return decision

        # Validate — always required before any apply
        min_confidence = float(getattr(config, "option_resolver_min_confidence", 0.75))
        validation = validate_gpt_result(
            bucket=bucket,
            gpt_result=gpt_result,
            allowed_intents=allowed_intents,
            choice_names=choices,
            known_item_names=known_item_names,
            min_confidence=min_confidence,
        )

        # In shadow mode: log but never apply
        if mode == "shadow":
            elapsed_ms = (time.monotonic() - t0) * 1000
            decision = _make_local_decision(
                local_nlu.effective_intent,
                local_nlu.slots,
                local_nlu.intent_confidence,
                bucket=bucket,
                gpt_result=gpt_result,
                reason="shadow_mode_no_apply",
                gpt_status=gpt_status,
                local_is_safe=_local_safe,
            )
            _log_resolver_result(bucket, gpt_result, decision, elapsed_ms)
            return decision

        # Inline mode: apply only when validation passes
        if mode == "inline" and validation.is_safe:
            gpt_result = GptTurnResolution(
                bucket=gpt_result.bucket,
                decision=gpt_result.decision,
                intent=gpt_result.intent,
                control_intent=gpt_result.control_intent,
                items=gpt_result.items,
                selected_option_names=gpt_result.selected_option_names,
                confidence=gpt_result.confidence,
                reason_code=gpt_result.reason_code,
                safe_to_apply=True,
                latency_ms=gpt_result.latency_ms,
                gpt_called=gpt_result.gpt_called,
                skipped_reason=gpt_result.skipped_reason,
                parse_error=gpt_result.parse_error,
                prompt_chars=gpt_result.prompt_chars,
                completion_chars=gpt_result.completion_chars,
                model=gpt_result.model,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            decision = FinalTurnDecision(
                source="gpt",
                bucket=bucket,
                local_intent=local_nlu.effective_intent,
                local_slots=local_nlu.slots,
                local_confidence=local_nlu.intent_confidence,
                gpt_result=gpt_result,
                apply_gpt=True,
                reason="inline_validated",
                gpt_status=GptCallStatus.OK,
                local_is_safe=_local_safe,
            )
            _log_resolver_result(bucket, gpt_result, decision, elapsed_ms)
            return decision

        # Inline but validation failed — fall back to local
        val_status = GptCallStatus.SCHEMA_VALIDATION_FAILED if not validation.is_safe else gpt_status
        elapsed_ms = (time.monotonic() - t0) * 1000
        decision = _make_local_decision(
            local_nlu.effective_intent,
            local_nlu.slots,
            local_nlu.intent_confidence,
            bucket=bucket,
            gpt_result=gpt_result,
            reason=f"validation_failed:{validation.reject_reason}",
            gpt_status=val_status,
            local_is_safe=_local_safe,
        )
        _log_resolver_result(bucket, gpt_result, decision, elapsed_ms)
        return decision

    except Exception as exc:
        _logger.warning("gpt_turn_resolver_error: %s", exc)
        elapsed_ms = (time.monotonic() - t0) * 1000
        decision = _make_local_decision(
            local_nlu.effective_intent,
            local_nlu.slots,
            local_nlu.intent_confidence,
            reason=f"resolver_exception:{type(exc).__name__}",
            gpt_status=GptCallStatus.UNKNOWN_ERROR,
        )
        _log_resolver_result(None, None, decision, elapsed_ms)
        return decision


def _classify_gpt_result_status(gpt_result: GptTurnResolution) -> str:
    """Map a GptTurnResolution to a GptCallStatus constant."""
    if not gpt_result.gpt_called:
        # Was not called — check skipped_reason before defaulting to DISABLED
        skipped = (gpt_result.skipped_reason or "").lower()
        if "missing_api_key" in skipped or "api_key" in skipped:
            return GptCallStatus.API_KEY_MISSING
        if "budget" in skipped:
            return GptCallStatus.BUDGET_EXCEEDED
        return GptCallStatus.DISABLED

    if gpt_result.decision == "error":
        # Distinguish parse vs provider error from parse_error content
        err = (gpt_result.parse_error or "").lower()
        if "timeout" in err or "timed" in err:
            return GptCallStatus.TIMEOUT
        if "rate" in err or "429" in err:
            return GptCallStatus.RATE_LIMITED
        if "500" in err or "provider" in err or "internal" in err:
            return GptCallStatus.PROVIDER_ERROR
        if "json" in err or "parse" in err or "decode" in err:
            return GptCallStatus.INVALID_JSON
        return GptCallStatus.UNKNOWN_ERROR

    if gpt_result.decision in {"skipped"}:
        if "missing_api_key" in (gpt_result.skipped_reason or ""):
            return GptCallStatus.API_KEY_MISSING
        if "budget" in (gpt_result.skipped_reason or ""):
            return GptCallStatus.BUDGET_EXCEEDED
        return GptCallStatus.DISABLED

    return GptCallStatus.OK
