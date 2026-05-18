# app/nlu/semantic_repair/gpt_repair_result.py
"""Immutable result of one GPT shadow-mode repair attempt."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlotCorrection:
    """One element of the structured slot_corrections list from GPT."""

    slot_name: str
    old_value: str | None
    new_value: str | None
    operation: str  # "add" | "replace" | "remove"


@dataclass(frozen=True, slots=True)
class GptRepairItem:
    """One item entry from the optional items[] array in a GPT response.

    Parsed and logged for training data; never applied to cart in this PR.

    Constraints applied at parse time:
      - item must be non-empty (entries with empty item are dropped)
      - quantity is clamped to [1, 99]
      - sides and modifiers are tuples (order preserved)
      - missing is a tuple of missing slot names
    """

    item: str
    quantity: int = 1
    size: str | None = None
    variant: str | None = None
    sides: tuple[str, ...] = ()
    modifiers: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GptRepairResult:
    """Outcome of a single GPT repair call (or a no-op when skipped).

    In phase 2 / all_shadow ``applied`` is always ``False`` — the result is
    logged but never used to change intent_result, slots, state, cart, or
    response.

    Decision values:
      "no_repair"                  – GPT found no improvement
      "repair" / "repair_intent"   – intent should change (legacy / new)
      "repair_slots"               – slot corrections only
      "repair_intent_and_slots"    – both intent and slot corrections
      "ask_clarifying_question"    – GPT suggests prompting the user
    """

    decision: str
    repaired_intent: str | None = None
    repaired_control_intent: str | None = None
    # dict form kept for backward compat with existing tests
    slot_corrections: dict | None = None
    # structured list form from new schema
    slot_corrections_list: tuple[SlotCorrection, ...] | None = None
    confidence: float | None = None
    reason: str | None = None
    latency_ms: float | None = None
    timeout: bool = False
    parse_error: str | None = None
    applied: bool = False

    # Timing breakdown (new)
    payload_build_ms: float | None = None
    request_ms: float | None = None
    parse_ms: float | None = None
    total_ms: float | None = None

    # Token / char counts
    prompt_chars: int | None = None
    completion_chars: int | None = None
    model: str | None = None

    # Reason the GPT call was skipped (eligible=False)
    skipped_reason: str | None = None

    # Fallback classification (decision="fallback" only)
    # Values: "none" | "off_topic" | "restaurant_question" | "user_frustrated"
    #         | "request_human" | "unclear" | "unsupported_request" | "back_to_order"
    fallback_type: str = "none"

    # Required slots GPT says are missing from the utterance (decision="missing_info")
    missing_slots: tuple[str, ...] = ()

    # Optional multi-item array from GPT (parse + log only; never applied to cart)
    items: tuple[GptRepairItem, ...] = ()


# Sentinel returned when the call is skipped (phase < 2 or not eligible).
GPT_NOT_CALLED = GptRepairResult(decision="no_repair")
