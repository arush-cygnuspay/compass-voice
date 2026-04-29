# app/diagnostics/turn_event.py
"""Immutable record of everything that happened during one conversation turn.

TurnEngine assembles a TurnEvent at every exit point and passes it to
TurnDiagnostics.record().  Backends (CSV, JSONL, …) consume it and write
whatever subset of fields they care about.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TurnEvent:
    # ── Session / turn metadata ────────────────────────────────────────────
    session_id: str
    turn_index: int

    # ── State before / after ──────────────────────────────────────────────
    state_before: str
    state_after: str
    next_state: str

    # ── Context snapshot (captured at turn start) ─────────────────────────
    pending_action: str
    current_prompt_field: str
    current_item_id: str
    current_item_name: str

    # ── Input text ────────────────────────────────────────────────────────
    raw_user_text: str
    user_text: str
    normalized_text: str

    # ── NLU predictions ───────────────────────────────────────────────────
    pred_main_intent: str
    pred_sub_intent: str
    pred_intent: str
    pred_intent_confidence: float | None
    slot_model_ran: bool
    slots: tuple[Any, ...]

    # ── Response ──────────────────────────────────────────────────────────
    response_key: str
    response_text: str
    command: dict[str, Any] | None

    # ── Slot interpretation ───────────────────────────────────────────────
    normalized_values: dict[str, Any]
    missing_required_fields: tuple[str, ...]

    # ── Reprompt tracking ─────────────────────────────────────────────────
    reprompt_field: str
    reprompt_count: int
    reprompt_escalated: bool
    reprompt_escalation_count: int

    # ── Fallback tracking ─────────────────────────────────────────────────
    fallback_triggered: bool
    fallback_reason: str
    fallback_count: int

    # ── Failure counters ──────────────────────────────────────────────────
    slot_extraction_failed: bool
    slot_extraction_failure_count: int
    invalid_modifier: bool
    invalid_modifier_count: int

    # ── Repeat detection ──────────────────────────────────────────────────
    user_repeated: bool
    repeated_user_turn_count: int

    # ── Timing (None when the path skips NLU / flow evaluation) ──────────
    preprocess_ms: float | None = None
    nlu_ms: float | None = None
    flow_ms: float | None = None
    route_ms: float | None = None
    handler_ms: float | None = None
    total_ms: float | None = None
