# app/nlu/semantic_repair/gpt_log_record_builder.py
"""Build structured dicts from a TurnEvent (or raw components) for GPT logging.

Two builders are provided:

build_gpt_repair_csv_row(event)
    Returns a flat dict keyed to GptRepairCsvLogger.HEADERS for CSV logging.

build_gpt_repair_log_record(event)
    Returns a nested JSON-safe dict from *event* (kept for reference / tooling).

build_gpt_shadow_jsonl_record(analysis, result, nlu, ...)
    Returns the full JSONL record for all_shadow and eligible_only JSONL logging.
    Uses raw components (not TurnEvent) so it works from background threads.

Safety contract — the record must never contain:
  - the raw prompt sent to GPT
  - the full menu (item catalogue)
  - raw cart JSON (only a safe summary)
  - API keys or payment links (sanitized by the logger)
  - phone numbers, e-mail addresses, or payment links in text fields
"""
from __future__ import annotations

import json as _json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.diagnostics.turn_event import TurnEvent
    from app.nlu.nlu_result import NLUResult
    from app.nlu.semantic_repair.gpt_repair_result import GptRepairResult
    from app.nlu.semantic_repair.repair_service import LocalTurnAnalysis
    from app.state_machine.models.conversation_state import ConversationState


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _items_to_list(items: Any) -> list[dict[str, Any]]:
    """Serialise GptRepairItem tuple to a list of plain dicts."""
    if not items:
        return []
    out = []
    for it in items:
        out.append({
            "item": getattr(it, "item", ""),
            "quantity": getattr(it, "quantity", 1),
            "size": getattr(it, "size", None),
            "variant": getattr(it, "variant", None),
            "sides": list(getattr(it, "sides", ())),
            "modifiers": list(getattr(it, "modifiers", ())),
            "missing": list(getattr(it, "missing", ())),
        })
    return out


# ---------------------------------------------------------------------------
# Flat CSV row builder (existing — unchanged interface)
# ---------------------------------------------------------------------------


def build_gpt_repair_csv_row(event: "TurnEvent") -> dict[str, Any]:
    """Return a flat dict keyed to GptRepairCsvLogger.HEADERS for CSV logging."""
    return {
        "timestamp_utc": _utc_iso(),
        "session_id": event.session_id,
        "turn_index": event.turn_index,
        "state_before": event.state_before,
        "response_key": event.response_key,
        # Customer utterance
        "user_text": event.user_text,
        "normalized_text": event.normalized_text,
        # Local model snapshot
        "local_intent": event.local_intent_before_gpt,
        "local_sub_intent": event.local_sub_intent_before_gpt,
        "local_confidence": event.local_intent_confidence_before_gpt,
        "local_candidates_json": event.local_intent_candidates_json,
        "local_slots_json": event.local_slots_before_gpt,
        "local_route_allowed": event.local_route_allowed,
        "local_route_reject_reason": event.local_route_reject_reason,
        # GPT eligibility
        "gpt_repair_eligible": event.gpt_repair_eligible,
        "gpt_eligible_reason": event.gpt_repair_eligible_reason,
        "gpt_candidate_count": event.gpt_candidate_count,
        "gpt_skipped_reason": event.gpt_skipped_reason,
        "gpt_phase": event.gpt_phase,
        # GPT call
        "gpt_called": event.gpt_called,
        "gpt_decision": event.gpt_decision,
        "gpt_selected_intent": event.gpt_selected_intent,
        "gpt_selected_control_intent": event.gpt_selected_control_intent,
        "gpt_slot_corrections_json": event.gpt_slot_corrections_json,
        "gpt_confidence": event.gpt_confidence,
        "gpt_reason": event.gpt_reason,
        "gpt_latency_ms": event.gpt_latency_ms,
        "gpt_total_ms": event.gpt_total_ms,
        "gpt_timeout": event.gpt_timeout,
        "gpt_parse_error": event.gpt_parse_error,
        "gpt_model": event.gpt_model,
        "gpt_prompt_chars": event.gpt_prompt_chars,
        "gpt_completion_chars": event.gpt_completion_chars,
        # Final block
        "gpt_applied": event.gpt_applied,
        "gpt_apply_reason": event.gpt_apply_reason,
        "final_intent_after_gpt": event.final_intent_after_gpt,
        "final_response_key": event.final_response_key,
        "training_candidate": event.training_candidate,
        # Fallback classification
        "gpt_fallback_type": event.gpt_fallback_type,
        "fallback_response_key": event.fallback_response_key,
        # ADD_ITEM extractor (Phase 1 shadow)
        "add_item_extractor_called": event.add_item_extractor_called,
        "add_item_eligible": event.add_item_eligible,
        "add_item_skipped_reason": event.add_item_skipped_reason,
        "add_item_decision": event.add_item_decision,
        "add_item_confidence": event.add_item_confidence,
        "add_item_items_json": event.add_item_items_json,
        "add_item_items_count": event.add_item_items_count,
        "add_item_global_slots_json": event.add_item_global_slots_json,
        "add_item_latency_ms": event.add_item_latency_ms,
        "add_item_total_ms": event.add_item_total_ms,
        "add_item_prompt_chars": event.add_item_prompt_chars,
        "add_item_completion_chars": event.add_item_completion_chars,
        "add_item_timeout": event.add_item_timeout,
        "add_item_parse_error": event.add_item_parse_error,
        "add_item_parse_notes_json": event.add_item_parse_notes_json,
        "add_item_reason": event.add_item_reason,
        "add_item_model": event.add_item_model,
        # ADD_ITEM validator (Phase 2 shadow)
        "add_item_validated_items_json": event.add_item_validated_items_json,
        "add_item_validated_items_count": event.add_item_validated_items_count,
        "add_item_rejected_items_json": event.add_item_rejected_items_json,
        "add_item_validation_warnings_json": event.add_item_validation_warnings_json,
        "add_item_validator_ms": event.add_item_validator_ms,
        "add_item_has_blocking_warnings": event.add_item_has_blocking_warnings,
    }


# ---------------------------------------------------------------------------
# Nested log record builder (existing — kept for reference / tooling)
# ---------------------------------------------------------------------------


def build_gpt_repair_log_record(event: "TurnEvent") -> dict[str, Any]:
    """Return a nested JSON-safe dict from *event* (kept for reference/tooling)."""
    return {
        "timestamp_utc": _utc_iso(),
        "session_id": event.session_id,
        "turn_index": event.turn_index,
        "local": {
            "intent": event.local_intent_before_gpt,
            "sub_intent": event.local_sub_intent_before_gpt,
            "confidence": event.local_intent_confidence_before_gpt,
            "candidates_json": event.local_intent_candidates_json,
            "slots_json": event.local_slots_before_gpt,
            "route_allowed": event.local_route_allowed,
            "route_reject_reason": event.local_route_reject_reason,
        },
        "allowed": {
            "repair_eligible": event.gpt_repair_eligible,
            "eligible_reason": event.gpt_repair_eligible_reason,
            "candidate_count": event.gpt_candidate_count,
            "skipped_reason": event.gpt_skipped_reason,
            "phase": event.gpt_phase,
        },
        "context": {
            "state_before": event.state_before,
            "response_key": event.response_key,
            "turn_index": event.turn_index,
        },
        "gpt": {
            "called": event.gpt_called,
            "decision": event.gpt_decision,
            "selected_intent": event.gpt_selected_intent,
            "selected_control_intent": event.gpt_selected_control_intent,
            "slot_corrections_json": event.gpt_slot_corrections_json,
            "confidence": event.gpt_confidence,
            "reason": event.gpt_reason,
            "latency_ms": event.gpt_latency_ms,
            "total_ms": event.gpt_total_ms,
            "timeout": event.gpt_timeout,
            "parse_error": event.gpt_parse_error,
            "model": event.gpt_model,
            "prompt_chars": event.gpt_prompt_chars,
            "completion_chars": event.gpt_completion_chars,
            "fallback_type": event.gpt_fallback_type,
        },
        "final": {
            "applied": event.gpt_applied,
            "apply_reason": event.gpt_apply_reason,
            "intent_after_gpt": event.final_intent_after_gpt,
            "response_key": event.final_response_key,
            "training_candidate": event.training_candidate,
            "fallback_response_key": event.fallback_response_key,
        },
        "add_item": {
            "called": event.add_item_extractor_called,
            "eligible": event.add_item_eligible,
            "skipped_reason": event.add_item_skipped_reason,
            "decision": event.add_item_decision,
            "confidence": event.add_item_confidence,
            "items_json": event.add_item_items_json,
            "items_count": event.add_item_items_count,
            "global_slots_json": event.add_item_global_slots_json,
            "latency_ms": event.add_item_latency_ms,
            "total_ms": event.add_item_total_ms,
            "prompt_chars": event.add_item_prompt_chars,
            "completion_chars": event.add_item_completion_chars,
            "timeout": event.add_item_timeout,
            "parse_error": event.add_item_parse_error,
            "parse_notes_json": event.add_item_parse_notes_json,
            "reason": event.add_item_reason,
            "model": event.add_item_model,
            # Phase 2: local menu validator results (shadow only)
            "validated_items_json": event.add_item_validated_items_json,
            "validated_items_count": event.add_item_validated_items_count,
            "rejected_items_json": event.add_item_rejected_items_json,
            "validation_warnings_json": event.add_item_validation_warnings_json,
            "validator_ms": event.add_item_validator_ms,
            "has_blocking_warnings": event.add_item_has_blocking_warnings,
        },
    }


# ---------------------------------------------------------------------------
# Full JSONL shadow record builder (new — source of truth for GPT training data)
# ---------------------------------------------------------------------------


def build_gpt_shadow_jsonl_record(
    *,
    analysis: "LocalTurnAnalysis",
    result: "GptRepairResult",
    nlu: "NLUResult",
    state_before: str,
    session_id: str,
    turn_index: int,
    response_key: str,
    final_response_key: str | None = None,
    call_mode: str = "",
    phase: int = 0,
) -> dict[str, Any]:
    """Build the full JSONL record from raw components.

    Suitable for both inline (eligible_only) and background (all_shadow) logging
    since it does not depend on a TurnEvent object.

    Safety invariants:
    - full_prompt is never included.
    - full_menu is never included.
    - full_cart raw JSON is never included (compact summary only).
    - API key is never included.
    - PII in text fields is sanitised by GptRepairJsonlLogger.log_turn().
    """
    from app.nlu.semantic_repair.gpt_repair_result import GPT_NOT_CALLED

    _gpt_called = (result is not None and result is not GPT_NOT_CALLED
                   and result.decision != "no_repair" or (
                       result is not None and result is not GPT_NOT_CALLED
                       and result.total_ms is not None
                   ))
    # More precise: GPT was called if total_ms is set (proxy for actual call)
    _gpt_called = (
        result is not None
        and result is not GPT_NOT_CALLED
        and result.total_ms is not None
    )

    # Top-k intent candidates (safe list, no PII)
    candidates_list: list[dict[str, Any]] = []
    for c in (getattr(analysis, "intent_candidates", ()) or ()):
        try:
            candidates_list.append({
                "intent": c.canonical_intent,
                "confidence": round(float(c.confidence), 4),
            })
        except Exception:
            pass

    # Slots extracted by local NLU
    slots_list: list[dict[str, Any]] = []
    for sv in (getattr(analysis, "slots", ()) or ()):
        try:
            slots_list.append({
                "name": getattr(sv, "name", ""),
                "value": str(getattr(sv, "value", "")),
            })
        except Exception:
            pass

    # Compact cart summary (safe: name list + count only, no prices/PII)
    cart_summary = getattr(analysis, "cart_summary", None)

    # Previous turns (capped to 3)
    prev_turns: list[dict[str, str]] = []
    for role, text in (getattr(analysis, "previous_turns", ()) or ()):
        prev_turns.append({"role": role, "text": text})

    # Choices for current waiting state
    choices = list(getattr(analysis, "choices", ()) or ())

    # Allowed intents / control / slots from candidates
    allowed_intents = sorted(getattr(analysis, "candidate_repair_intents", ()) or ())
    allowed_control = sorted(getattr(analysis, "candidate_control_kinds", ()) or ())
    # Slot corrections as list
    sc_list: list[dict[str, Any]] = []
    if result and result.slot_corrections_list:
        for sc in result.slot_corrections_list:
            sc_list.append({
                "slot_name": sc.slot_name,
                "old_value": sc.old_value,
                "new_value": sc.new_value,
                "operation": sc.operation,
            })

    # items[] (parse + log only)
    items_list = _items_to_list(getattr(result, "items", ()) if result else ())

    # missing slots from GPT
    missing_slots = list(getattr(result, "missing_slots", ()) if result else ())

    return {
        "timestamp_utc": _utc_iso(),
        "session_id": session_id,
        "turn_index": turn_index,
        "call_mode": call_mode,
        "phase": phase,
        # Context
        "state_before": state_before,
        "normalized_text": getattr(nlu, "normalized_text", "") or "",
        "response_key": response_key,
        # Local model snapshot
        "local": {
            "intent": getattr(analysis, "intent_effective", None),
            "sub_intent": None,  # not stored in LocalTurnAnalysis
            "confidence": getattr(analysis, "intent_confidence", None),
            "top_intents": candidates_list,
            "slots": slots_list,
            "route_allowed": getattr(analysis, "gpt_repair_eligible", None),
        },
        # Allowed candidates (curated per-state sets; never the full enum)
        "allowed": {
            "intents": allowed_intents,
            "control": allowed_control,
            "slots": list(getattr(analysis, "required_missing", ()) or ()),
            "choices": choices,
            "repair_eligible": getattr(analysis, "gpt_repair_eligible", False),
            "eligible_reason": getattr(analysis, "reason", None),
            "skipped_reason": getattr(analysis, "skipped_reason", None),
        },
        # Compact cart (no prices, no PII, no raw JSON)
        "compact_cart": cart_summary,
        # Previous conversation turns (capped to 3)
        "previous_turns": prev_turns,
        # GPT output
        "gpt": {
            "called": _gpt_called,
            "decision": getattr(result, "decision", None) if result else None,
            "intent": getattr(result, "repaired_intent", None) if result else None,
            "control": getattr(result, "repaired_control_intent", None) if result else None,
            "slots": sc_list,
            "items": items_list,
            "missing": missing_slots,
            "fallback_type": getattr(result, "fallback_type", "none") if result else "none",
            "confidence": getattr(result, "confidence", None) if result else None,
            "latency_ms": getattr(result, "latency_ms", None) if result else None,
            "total_ms": getattr(result, "total_ms", None) if result else None,
            "timeout": getattr(result, "timeout", False) if result else False,
            "parse_error": getattr(result, "parse_error", None) if result else None,
            "model": getattr(result, "model", None) if result else None,
            "prompt_chars": getattr(result, "prompt_chars", None) if result else None,
            "completion_chars": getattr(result, "completion_chars", None) if result else None,
        },
        # Final — invariant: final == local in shadow mode (never applied)
        "final": {
            "intent_after_gpt": getattr(analysis, "intent_effective", None),
            "response_key": final_response_key or response_key,
            "gpt_applied": False,
            "training_candidate": False,
        },
    }
