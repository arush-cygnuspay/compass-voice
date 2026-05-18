"""Immutable record of everything that happened during one conversation turn.

TurnEngine assembles a TurnEvent at every exit point and passes it to
TurnDiagnostics.record(). Backends (CSV, JSONL, etc.) consume it and write
whatever subset of fields they care about.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TurnEvent:
    session_id: str
    turn_index: int

    state_before: str
    state_after: str
    next_state: str

    pending_action: str
    current_prompt_field: str
    current_item_id: str
    current_item_name: str

    raw_user_text: str
    user_text: str
    normalized_text: str

    pred_main_intent: str
    pred_sub_intent: str
    pred_intent: str
    pred_intent_confidence: float | None
    slot_model_ran: bool
    slots: tuple[Any, ...]

    response_key: str
    response_text: str
    command: dict[str, Any] | None

    normalized_values: dict[str, Any]
    missing_required_fields: tuple[str, ...]

    reprompt_field: str
    reprompt_count: int
    reprompt_escalated: bool
    reprompt_escalation_count: int

    fallback_triggered: bool
    fallback_reason: str
    fallback_count: int

    slot_extraction_failed: bool
    slot_extraction_failure_count: int
    invalid_modifier: bool
    invalid_modifier_count: int

    user_repeated: bool
    repeated_user_turn_count: int

    preprocess_ms: float | None = None
    nlu_ms: float | None = None
    flow_ms: float | None = None
    route_ms: float | None = None
    handler_ms: float | None = None
    total_ms: float | None = None

    raw_slots: tuple[Any, ...] | None = None
    effective_slots: tuple[Any, ...] | None = None
    active_resolution_scope: str | None = None
    resolved_entity_type: str | None = None
    resolved_entity_id: str | None = None
    route_reason: str | None = None
    coercion_reason: str | None = None

    local_intent_before_gpt: str | None = None
    local_sub_intent_before_gpt: str | None = None
    local_intent_confidence_before_gpt: float | None = None
    local_intent_candidates_json: str | None = None
    local_slots_before_gpt: str | None = None
    local_route_allowed: bool | None = None
    local_route_reject_reason: str | None = None

    gpt_repair_eligible: bool = False
    gpt_repair_eligible_reason: str | None = None
    gpt_repair_reason: str | None = None
    gpt_candidate_count: int | None = None
    gpt_skipped_reason: str | None = None
    gpt_phase: int = 0
    gpt_called: bool = False
    gpt_payload_build_ms: float | None = None
    gpt_request_ms: float | None = None
    gpt_parse_ms: float | None = None
    gpt_total_ms: float | None = None
    gpt_prompt_chars: int | None = None
    gpt_completion_chars: int | None = None
    gpt_model: str | None = None

    gpt_decision: str | None = None
    gpt_selected_intent: str | None = None
    gpt_selected_control_intent: str | None = None
    gpt_slot_corrections_json: str | None = None
    gpt_confidence: float | None = None
    gpt_reason: str | None = None
    gpt_latency_ms: float | None = None
    gpt_timeout: bool = False
    gpt_parse_error: str | None = None

    gpt_applied: bool = False
    gpt_apply_reason: str | None = None
    final_intent_after_gpt: str | None = None
    final_slots_after_gpt: str | None = None
    final_response_key: str | None = None
    training_candidate: bool = False

    gpt_fallback_type: str = "none"
    fallback_response_key: str | None = None

    add_item_extractor_called: bool = False
    add_item_eligible: bool = False
    add_item_skipped_reason: str | None = None
    add_item_decision: str | None = None
    add_item_confidence: float | None = None
    add_item_items_json: str | None = None
    add_item_items_count: int | None = None
    add_item_global_slots_json: str | None = None
    add_item_latency_ms: float | None = None
    add_item_total_ms: float | None = None
    add_item_prompt_chars: int | None = None
    add_item_completion_chars: int | None = None
    add_item_timeout: bool = False
    add_item_parse_error: str | None = None
    add_item_parse_notes_json: str | None = None
    add_item_reason: str | None = None
    add_item_model: str | None = None

    add_item_validated_items_json: str | None = None
    add_item_validated_items_count: int | None = None
    add_item_rejected_items_json: str | None = None
    add_item_validation_warnings_json: str | None = None
    add_item_validator_ms: float | None = None
    add_item_has_blocking_warnings: bool = False
