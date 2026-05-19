# app/logging/turn_event_schema.py
"""Canonical TurnEvent JSONL schema — version 1.

Every committed customer turn produces exactly one canonical record written to
``logs/current/turn_events.jsonl``.  The record is the single source of truth
for debugging, replay, and future NLU model training.

Structure (top-level sections)
-------------------------------
schema_version   — semver string, bumped on breaking changes.
timestamp_utc    — ISO-8601 UTC timestamp.
ids              — session / call / store identifiers.
turn             — FSM context at the time of the turn.
asr              — raw / cleaned / normalised ASR text.
local_nlu        — local model intent + slot output (before any repair).
smart_planner    — SmartTurnPlanner decision (null when not invoked).
gpt_repair       — GptRepairService decision (null when not invoked).
final_decision   — which system was used and what changed (derived).
validation       — plan-validator outcome (from add_item shadow pipeline).
cart             — before/after cart hash + item-level diff.
response         — response_key and rendered text.
latency          — engine timing breakdown in milliseconds.
training         — training-candidate classification (derived).
errors           — non-fatal errors encountered during this turn.

Safety invariants (must always hold)
--------------------------------------
* No raw API keys.
* No full menu (item catalogue).
* No raw cart JSON — only compact name lists and hashes.
* PII in text fields (phone, e-mail, payment links) is redacted by TurnEventLogger.
* ISO-8601 timestamps are never redacted.

Extension notes
---------------
* ``smart_planner`` fields are stubbed (all null / false) until SmartTurnPlanner
  results are wired into the diagnostics TurnEvent dataclass.
* ``cart.diff`` is stubbed (empty list) until cart snapshots are captured.
"""
from __future__ import annotations

import hashlib
import json as _json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.diagnostics.turn_event import TurnEvent
    from app.logging.final_decision_resolver import FinalDecision
    from app.logging.training_candidate_classifier import TrainingClassification


SCHEMA_VERSION: str = "1"


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_canonical_record(
    event: "TurnEvent",
    *,
    final_decision: "FinalDecision",
    training: "TrainingClassification",
    # Optional extras not yet in TurnEvent but available at call sites.
    call_sid: str = "",
    stream_sid: str = "",
    store_id: str = "",
    company_id: str = "",
    previous_assistant_text: str = "",
    cart_before_hash: str | None = None,
    cart_after_hash: str | None = None,
    cart_diff: list[dict[str, Any]] | None = None,
    spoken_text: str = "",
    tts_chunks: int = 0,
    validation_ok: bool = True,
    validation_validator: str | None = None,
    validation_errors: list[str] | None = None,
    validation_warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON record dict for *event*.

    The returned dict is JSON-serialisable via the custom ``_json_default``
    encoder in ``turn_event_logger.py``.  Callers should not call
    ``json.dumps`` directly; the logger handles encoding safely.

    Parameters
    ----------
    event:
        The internal TurnEvent assembled by TurnEngine._record_turn_event.
    final_decision:
        Pre-computed by FinalDecisionResolver.resolve_final_decision(event).
    training:
        Pre-computed by TrainingCandidateClassifier.classify_training_candidate.
    call_sid / stream_sid / store_id / company_id:
        Telephony / restaurant identifiers not yet on TurnEvent.
    previous_assistant_text:
        Last spoken bot text (for context in training records).
    cart_before_hash / cart_after_hash / cart_diff:
        Cart snapshot — optional until the cart-diff pipeline is wired.
    spoken_text:
        The actual TTS text sent to the caller (may differ from internal text).
    tts_chunks:
        Number of TTS audio chunks streamed.
    validation_ok / validation_validator / validation_errors / validation_warnings:
        Plan-validator outcome from the add_item shadow pipeline.
    errors:
        Non-fatal errors collected during this turn (logged but not raised).
    """
    try:
        return _build(
            event=event,
            final_decision=final_decision,
            training=training,
            call_sid=call_sid,
            stream_sid=stream_sid,
            store_id=store_id,
            company_id=company_id,
            previous_assistant_text=previous_assistant_text,
            cart_before_hash=cart_before_hash,
            cart_after_hash=cart_after_hash,
            cart_diff=cart_diff or [],
            spoken_text=spoken_text,
            tts_chunks=tts_chunks,
            validation_ok=validation_ok,
            validation_validator=validation_validator,
            validation_errors=validation_errors or [],
            validation_warnings=validation_warnings or [],
            errors=errors or [],
        )
    except Exception as exc:
        # Return a minimal safe record rather than crashing the turn
        return {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": _utc_now(),
            "ids": {"session_id": str(getattr(event, "session_id", ""))},
            "turn": {"turn_index": int(getattr(event, "turn_index", 0))},
            "errors": [f"canonical_record_build_error: {type(exc).__name__}: {exc}"],
            "_build_failed": True,
        }


# ---------------------------------------------------------------------------
# Private builder
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: Any, default: Any = None) -> Any:
    """Return *value* or *default* when value is falsy (None / empty string)."""
    return value if value is not None else default


def _slots_list(slots_json: str | None) -> list[dict[str, Any]]:
    """Parse a slots JSON string into a list of dicts."""
    if not slots_json:
        return []
    try:
        parsed = _json.loads(slots_json)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def _candidates_list(candidates_json: str | None) -> list[dict[str, Any]]:
    if not candidates_json:
        return []
    try:
        parsed = _json.loads(candidates_json)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def _build(
    *,
    event: "TurnEvent",
    final_decision: "FinalDecision",
    training: "TrainingClassification",
    call_sid: str,
    stream_sid: str,
    store_id: str,
    company_id: str,
    previous_assistant_text: str,
    cart_before_hash: str | None,
    cart_after_hash: str | None,
    cart_diff: list[dict[str, Any]],
    spoken_text: str,
    tts_chunks: int,
    validation_ok: bool,
    validation_validator: str | None,
    validation_errors: list[str],
    validation_warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    # ── IDs ──────────────────────────────────────────────────────────────
    session_id = str(getattr(event, "session_id", "") or "")

    # ── Normalised text pieces ────────────────────────────────────────────
    raw_text = str(getattr(event, "raw_user_text", "") or getattr(event, "user_text", "") or "")
    cleaned_text = str(getattr(event, "user_text", "") or "")
    normalized_text = str(getattr(event, "normalized_text", "") or "")

    # ── Local NLU snapshot ────────────────────────────────────────────────
    local_candidates = _candidates_list(getattr(event, "local_intent_candidates_json", None))
    local_slots = _slots_list(getattr(event, "local_slots_before_gpt", None))

    # ── GPT repair ───────────────────────────────────────────────────────
    gpt_slot_corrections = _slots_list(getattr(event, "gpt_slot_corrections_json", None))
    gpt_called = bool(getattr(event, "gpt_called", False))

    # ── Validation (add_item shadow pipeline) ────────────────────────────
    # If add_item validator ran and had blocking warnings, update validation result.
    if getattr(event, "add_item_has_blocking_warnings", False):
        validation_ok = False
        validation_validator = "add_item_menu_validator"
        _warnings_json = getattr(event, "add_item_validation_warnings_json", None)
        if _warnings_json and not validation_warnings:
            try:
                _w = _json.loads(_warnings_json)
                if isinstance(_w, list):
                    validation_warnings = [
                        f"{w.get('code','?')}: {w.get('detail','')}"
                        for w in _w
                        if isinstance(w, dict)
                    ]
            except Exception:
                pass

    # ── Latency ──────────────────────────────────────────────────────────
    latency: dict[str, Any] = {
        "preprocess_ms": getattr(event, "preprocess_ms", None),
        "nlu_ms": getattr(event, "nlu_ms", None),
        "flow_ms": getattr(event, "flow_ms", None),
        "route_ms": getattr(event, "route_ms", None),
        "handler_ms": getattr(event, "handler_ms", None),
        "total_ms": getattr(event, "total_ms", None),
        "gpt_payload_build_ms": getattr(event, "gpt_payload_build_ms", None),
        "gpt_request_ms": getattr(event, "gpt_request_ms", None),
        "gpt_parse_ms": getattr(event, "gpt_parse_ms", None),
        "gpt_total_ms": getattr(event, "gpt_total_ms", None),
        "add_item_latency_ms": getattr(event, "add_item_latency_ms", None),
        "add_item_total_ms": getattr(event, "add_item_total_ms", None),
        "add_item_validator_ms": getattr(event, "add_item_validator_ms", None),
    }

    # ── Build record ─────────────────────────────────────────────────────
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": _utc_now(),

        # ── Identifiers ──────────────────────────────────────────────────
        "ids": {
            "session_id": session_id,
            "call_sid": call_sid,
            "stream_sid": stream_sid,
            "store_id": store_id,
            "company_id": company_id,
        },

        # ── Turn context ─────────────────────────────────────────────────
        "turn": {
            "turn_index": int(getattr(event, "turn_index", 0)),
            "state_before": str(getattr(event, "state_before", "") or ""),
            "state_after": str(getattr(event, "state_after", "") or ""),
            "pending_action": str(getattr(event, "pending_action", "") or ""),
            "current_prompt_field": str(getattr(event, "current_prompt_field", "") or ""),
            "current_item_id": str(getattr(event, "current_item_id", "") or ""),
            "current_item_name": str(getattr(event, "current_item_name", "") or ""),
            "previous_assistant_text": previous_assistant_text,
            "reprompt_count": int(getattr(event, "reprompt_count", 0) or 0),
            "reprompt_field": str(getattr(event, "reprompt_field", "") or ""),
            "reprompt_escalated": bool(getattr(event, "reprompt_escalated", False)),
            "fallback_triggered": bool(getattr(event, "fallback_triggered", False)),
            "fallback_count": int(getattr(event, "fallback_count", 0) or 0),
            "user_repeated": bool(getattr(event, "user_repeated", False)),
        },

        # ── ASR ──────────────────────────────────────────────────────────
        "asr": {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "normalized_text": normalized_text,
            "confidence": None,     # populated when ASR confidence is available
            "alternatives": [],     # populated when ASR n-best list is available
        },

        # ── Local NLU (before any repair) ────────────────────────────────
        "local_nlu": {
            "intent_main": str(getattr(event, "pred_main_intent", "") or ""),
            "intent_sub_intent": str(getattr(event, "pred_sub_intent", "") or ""),
            "intent_effective": str(
                getattr(event, "local_intent_before_gpt", None)
                or getattr(event, "pred_intent", "") or ""
            ),
            "confidence": getattr(event, "local_intent_confidence_before_gpt", None)
                or getattr(event, "pred_intent_confidence", None),
            "all_candidates": local_candidates,
            "slot_model_ran": bool(getattr(event, "slot_model_ran", False)),
            "slots": local_slots,
            "route_allowed": getattr(event, "local_route_allowed", None),
            "route_reject_reason": getattr(event, "local_route_reject_reason", None),
        },

        # ── SmartTurnPlanner ─────────────────────────────────────────────
        # Stubbed until SmartTurnPlanner results are wired into TurnEvent.
        # See: app/services/smart_turn_planner.py, app/services/smart_turn_policy.py
        "smart_planner": {
            "enabled": False,
            "invoked": False,
            "task_mode": None,
            "trigger_reason": None,
            "context_keys": [],
            "allowed_options_count": 0,
            "menu_candidates_count": 0,
            "model": None,
            "decision": None,
            "selected_intent": None,
            "selected_control_intent": None,
            "parsed_plan": None,
            "parse_error": None,
            "timeout": False,
            "latency_ms": None,
            "applied": False,
            "apply_reason": None,
            "fallback_type": None,
        },

        # ── GPT repair ───────────────────────────────────────────────────
        "gpt_repair": {
            "eligible": bool(getattr(event, "gpt_repair_eligible", False)),
            "eligible_reason": getattr(event, "gpt_repair_eligible_reason", None),
            "called": gpt_called,
            "phase": int(getattr(event, "gpt_phase", 0) or 0),
            "decision": getattr(event, "gpt_decision", None),
            "selected_intent": getattr(event, "gpt_selected_intent", None),
            "selected_control_intent": getattr(event, "gpt_selected_control_intent", None),
            "slot_corrections": gpt_slot_corrections,
            "confidence": getattr(event, "gpt_confidence", None),
            "reason": getattr(event, "gpt_reason", None),
            "latency_ms": getattr(event, "gpt_latency_ms", None),
            "total_ms": getattr(event, "gpt_total_ms", None),
            "timeout": bool(getattr(event, "gpt_timeout", False)),
            "parse_error": getattr(event, "gpt_parse_error", None),
            "model": getattr(event, "gpt_model", None),
            "prompt_chars": getattr(event, "gpt_prompt_chars", None),
            "completion_chars": getattr(event, "gpt_completion_chars", None),
            "applied": bool(getattr(event, "gpt_applied", False)),
            "apply_reason": getattr(event, "gpt_apply_reason", None),
            "fallback_type": str(getattr(event, "gpt_fallback_type", "none") or "none"),
        },

        # ── ADD_ITEM extractor (shadow pipeline) ─────────────────────────
        "add_item_extractor": {
            "called": bool(getattr(event, "add_item_extractor_called", False)),
            "eligible": bool(getattr(event, "add_item_eligible", False)),
            "skipped_reason": getattr(event, "add_item_skipped_reason", None),
            "decision": getattr(event, "add_item_decision", None),
            "confidence": getattr(event, "add_item_confidence", None),
            "items_count": getattr(event, "add_item_items_count", None),
            "timeout": bool(getattr(event, "add_item_timeout", False)),
            "parse_error": getattr(event, "add_item_parse_error", None),
            "model": getattr(event, "add_item_model", None),
        },

        # ── Final decision (derived) ──────────────────────────────────────
        "final_decision": {
            "final_intent": final_decision.final_intent,
            "final_source": final_decision.final_source,
            "repair_applied": final_decision.repair_applied,
            "repair_type": final_decision.repair_type,
            "intent_changed": final_decision.intent_changed,
            "slots_changed": final_decision.slots_changed,
            "response_key": final_decision.response_key,
            "decision_reason": final_decision.decision_reason,
        },

        # ── Validation (add_item shadow pipeline) ────────────────────────
        "validation": {
            "ok": validation_ok,
            "validator": validation_validator,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "validated_items_count": getattr(event, "add_item_validated_items_count", None),
            "has_blocking_warnings": bool(getattr(event, "add_item_has_blocking_warnings", False)),
        },

        # ── Cart ─────────────────────────────────────────────────────────
        # cart_before_hash and cart_after_hash are optional until the
        # cart-snapshot pipeline is wired.
        "cart": {
            "cart_before_hash": cart_before_hash,
            "cart_after_hash": cart_after_hash,
            "diff": cart_diff,
        },

        # ── Response ─────────────────────────────────────────────────────
        "response": {
            "response_key": str(getattr(event, "response_key", "") or ""),
            "internal_text": str(getattr(event, "response_text", "") or ""),
            "spoken_text": spoken_text,
            "tts_chunks": tts_chunks,
        },

        # ── Latency ──────────────────────────────────────────────────────
        "latency": latency,

        # ── Training (derived) ────────────────────────────────────────────
        "training": {
            "candidate": training.candidate,
            "candidate_reason": training.candidate_reasons,
            "label_status": training.label_status,
            "gold_intent": None,
            "gold_slots": [],
            "gold_action": None,
            "needs_human_review": training.needs_human_review,
        },

        # ── Non-fatal errors ─────────────────────────────────────────────
        "errors": errors,
    }
