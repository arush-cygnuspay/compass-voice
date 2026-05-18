# app/nlu/semantic_repair/add_item_planner_output_parser.py
"""Parse GPT output into Phase 4 AddItemPlannerResult items.

This parser handles the Phase 4 planner schema, which is distinct from the
Phase 1/2 extractor schema:
  - decisions: "add_items" | "clarify" | "no_repair" | "unclear"
  - modifier operations: "add" | "remove" | "extra" | "light"
  - unresolved[] entity array
  - candidate_item_id per item

Parsing contract
----------------
* Never raises — all errors return a safe "no_repair" result state.
* Hallucinated item names (not found in utterance or candidates) are dropped.
* Hallucinated modifier/side names (not found in utterance or candidates) are dropped.
* Quantity must be a positive integer; invalid values default to 1.
* Operations not in the allowed set default to "add".
* Items are capped to max_items; overflow is noted in parse_notes.
"""
from __future__ import annotations

import json
from typing import Any

from app.nlu.semantic_repair.add_item_planner_result import (
    PlannerGptItem,
    PlannerGptModifier,
    PlannerGptSide,
    PlannerUnresolved,
)

# Valid Phase 4 decision values
_VALID_DECISIONS: frozenset[str] = frozenset({
    "add_items", "clarify", "no_repair", "unclear",
})

# Valid modifier operations (Phase 4 extends Phase 1/2 with extra + light)
_VALID_OPERATIONS: frozenset[str] = frozenset({
    "add", "remove", "extra", "light",
})

# Valid unresolved reason codes
_VALID_UNRESOLVED_REASONS: frozenset[str] = frozenset({
    "not_on_menu", "ambiguous", "belongs_to_unknown_group", "unsupported",
})

# Maximum string length for name fields (sanity clamp)
_MAX_STR_LEN: int = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(s: Any, max_len: int = _MAX_STR_LEN) -> str | None:
    if not isinstance(s, str):
        return None
    stripped = s.strip()[:max_len]
    return stripped or None


def _safe_int(v: Any, default: int = 1) -> int:
    try:
        result = int(v)
        return result if result >= 1 else default
    except (TypeError, ValueError):
        return default


def _safe_operation(op: Any) -> str:
    return op if isinstance(op, str) and op in _VALID_OPERATIONS else "add"


def _in_utterance_or_candidates(
    value: str,
    *,
    utterance_lower: str,
    candidate_names: set[str],
    candidate_option_names: set[str],
) -> bool:
    """Return True if value appears in utterance or in candidate option names."""
    v = value.strip().lower()
    if not v:
        return False
    if v in utterance_lower:
        return True
    if v in candidate_names:
        return True
    if v in candidate_option_names:
        return True
    return False


# ---------------------------------------------------------------------------
# Sub-parsers
# ---------------------------------------------------------------------------


def _parse_modifier(raw: Any, *, utterance_lower: str, candidate_option_names: set[str]) -> PlannerGptModifier | None:
    if not isinstance(raw, dict):
        return None
    name = _safe_str(raw.get("name"))
    if not name:
        return None
    # Drop hallucinated modifier names
    name_lower = name.lower()
    if name_lower not in utterance_lower and name_lower not in candidate_option_names:
        return None
    return PlannerGptModifier(
        name=name,
        operation=_safe_operation(raw.get("operation")),
        quantity=_safe_int(raw.get("quantity")),
    )


def _parse_side(raw: Any, *, utterance_lower: str, candidate_option_names: set[str]) -> PlannerGptSide | None:
    if not isinstance(raw, dict):
        return None
    name = _safe_str(raw.get("name"))
    if not name:
        return None
    # Drop hallucinated side names
    name_lower = name.lower()
    if name_lower not in utterance_lower and name_lower not in candidate_option_names:
        return None
    size = _safe_str(raw.get("size"))
    return PlannerGptSide(
        name=name,
        quantity=_safe_int(raw.get("quantity")),
        size=size,
    )


def _parse_item(
    raw: Any,
    *,
    utterance_lower: str,
    candidate_names: set[str],
    candidate_option_names: set[str],
) -> PlannerGptItem | None:
    if not isinstance(raw, dict):
        return None

    item_name = _safe_str(raw.get("item_name") or raw.get("name") or "")
    if not item_name:
        return None

    # Drop hallucinated item names (not in utterance AND not a known candidate)
    item_lower = item_name.lower()
    if item_lower not in utterance_lower and item_lower not in candidate_names:
        return None

    candidate_item_id = _safe_str(raw.get("candidate_item_id"))
    quantity = _safe_int(raw.get("quantity"))
    size = _safe_str(raw.get("size"))
    variant = _safe_str(raw.get("variant"))
    special_instructions = _safe_str(raw.get("special_instructions"))

    modifiers: list[PlannerGptModifier] = []
    for m in (raw.get("modifiers") or []):
        parsed = _parse_modifier(m, utterance_lower=utterance_lower, candidate_option_names=candidate_option_names)
        if parsed is not None:
            modifiers.append(parsed)

    sides: list[PlannerGptSide] = []
    for s in (raw.get("sides") or []):
        parsed_side = _parse_side(s, utterance_lower=utterance_lower, candidate_option_names=candidate_option_names)
        if parsed_side is not None:
            sides.append(parsed_side)

    return PlannerGptItem(
        item_name=item_name,
        candidate_item_id=candidate_item_id,
        quantity=quantity,
        size=size,
        variant=variant,
        modifiers=tuple(modifiers),
        sides=tuple(sides),
        special_instructions=special_instructions,
    )


def _parse_unresolved(raw: Any) -> PlannerUnresolved | None:
    if not isinstance(raw, dict):
        return None
    text = _safe_str(raw.get("text"))
    if not text:
        return None
    reason_raw = raw.get("reason", "unsupported")
    reason = reason_raw if reason_raw in _VALID_UNRESOLVED_REASONS else "unsupported"
    return PlannerUnresolved(text=text, reason=reason)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_planner_output(
    raw: str,
    *,
    utterance_text: str,
    candidate_names: set[str] | None = None,
    candidate_option_names: set[str] | None = None,
    max_items: int = 8,
) -> tuple[str, tuple[PlannerGptItem, ...], tuple[PlannerUnresolved, ...], float | None, str | None, str | None]:
    """Parse Phase 4 GPT planner output.

    Returns
    -------
    (decision, items, unresolved, confidence, reason_code, parse_error)

    Never raises — all errors return ("no_repair", (), (), None, None, error_str).
    """
    utterance_lower = (utterance_text or "").lower().strip()
    cand_names = {n.lower() for n in (candidate_names or set())}
    cand_options = {n.lower() for n in (candidate_option_names or set())}

    # Strip markdown code fences if present
    stripped = (raw or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    try:
        data: Any = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        return "no_repair", (), (), None, None, f"json_decode:{exc}"[:200]

    if not isinstance(data, dict):
        return "no_repair", (), (), None, None, "json_not_object"

    # Decision
    decision_raw = data.get("decision", "no_repair")
    decision = decision_raw if decision_raw in _VALID_DECISIONS else "no_repair"

    # Items
    raw_items = data.get("items") or []
    if not isinstance(raw_items, list):
        raw_items = []

    parsed_items: list[PlannerGptItem] = []
    parse_notes: list[str] = []
    for entry in raw_items:
        item = _parse_item(
            entry,
            utterance_lower=utterance_lower,
            candidate_names=cand_names,
            candidate_option_names=cand_options,
        )
        if item is not None:
            parsed_items.append(item)

    if len(parsed_items) > max_items:
        overflow = len(parsed_items) - max_items
        parsed_items = parsed_items[:max_items]
        parse_notes.append(f"items_truncated:{overflow}")

    # Unresolved
    raw_unresolved = data.get("unresolved") or []
    parsed_unresolved: list[PlannerUnresolved] = []
    if isinstance(raw_unresolved, list):
        for entry in raw_unresolved:
            u = _parse_unresolved(entry)
            if u is not None:
                parsed_unresolved.append(u)

    # Confidence
    confidence: float | None = None
    try:
        conf_raw = data.get("confidence")
        if conf_raw is not None:
            confidence = max(0.0, min(1.0, float(conf_raw)))
    except (TypeError, ValueError):
        pass

    # Reason code
    reason_code = _safe_str(data.get("reason_code"))

    parse_error = ("; ".join(parse_notes)) if parse_notes else None

    return (
        decision,
        tuple(parsed_items),
        tuple(parsed_unresolved),
        confidence,
        reason_code,
        parse_error,
    )
