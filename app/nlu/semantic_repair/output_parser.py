# app/nlu/semantic_repair/output_parser.py
"""Parse and validate the raw JSON string returned by GPT.

Validation rules (failures produce decision=no_repair with parse_error logged):
  1. Raw string must be valid JSON.
  2. ``requires_handler_validation`` (or ``rhv``) must be exactly ``True``.
  3. If decision includes intent repair: the intent must be in the candidate set.
  4. Slot correction values must appear in the original utterance OR original slots.
     Invalid slot *names* are dropped silently rather than failing the whole response.

Both old schema (repaired_intent, decision="repair") and new compact schema
(intent, rhv, decision="ok|repair|missing_info|...") are accepted.
"""
from __future__ import annotations

import json
from typing import Any

from app.nlu.nlu_result import NLUResult
from app.nlu.semantic_repair.gpt_repair_result import GptRepairItem, GptRepairResult, SlotCorrection

# All recognised decision values (old + new)
_VALID_DECISIONS = frozenset({
    "ok",                      # new: local model correct (alias for no_repair)
    "no_repair",
    "repair",                  # legacy
    "repair_intent",           # legacy
    "repair_slots",            # legacy
    "repair_intent_and_slots", # legacy
    "ask_clarifying_question", # legacy
    "missing_info",            # new: required slot absent from utterance
    "fallback",
})

_VALID_FALLBACK_TYPES = frozenset({
    "off_topic",
    "restaurant_question",
    "user_frustrated",
    "request_human",
    "unclear",
    "unsupported_request",
    "back_to_order",
})

# Known slot names — entries with other names are dropped, not rejected
_VALID_SLOT_NAMES: frozenset[str] = frozenset({
    "ITEM", "MODIFIER", "SIDE", "QUANTITY", "SIZE", "VARIANT", "MENU_ITEM",
})

# Decisions that include an intent repair
_REPAIR_INTENT_DECISIONS = frozenset({
    "repair",
    "repair_intent",
    "repair_intent_and_slots",
})

# Decisions that include slot corrections
_REPAIR_SLOT_DECISIONS = frozenset({
    "repair_slots",
    "repair_intent_and_slots",
})


def parse_output(
    *,
    raw: str,
    candidates: frozenset[str],
    nlu: NLUResult,
    latency_ms: float,
) -> GptRepairResult:
    """Parse ``raw`` and return a validated GptRepairResult.

    Any hard validation failure returns decision=no_repair with parse_error set.
    applied is always False — the caller (shadow mode) never applies the result.
    """
    # 1. JSON parse
    try:
        data: dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return GptRepairResult(
            decision="no_repair",
            parse_error=f"json_decode:{exc}",
            latency_ms=latency_ms,
        )

    if not isinstance(data, dict):
        return GptRepairResult(
            decision="no_repair",
            parse_error="response_not_dict",
            latency_ms=latency_ms,
        )

    # 2. requires_handler_validation (full key) OR rhv (short key) must be True
    rhv = data.get("rhv") or data.get("requires_handler_validation")
    if rhv is not True:
        return GptRepairResult(
            decision="no_repair",
            parse_error="requires_handler_validation_not_true",
            latency_ms=latency_ms,
        )

    raw_decision = data.get("decision", "no_repair")
    decision = raw_decision if raw_decision in _VALID_DECISIONS else "no_repair"

    # Normalise "ok" to "no_repair" (semantically identical)
    if decision == "ok":
        decision = "no_repair"

    # Support both old long keys and new short keys
    repaired_intent: str | None = (
        data.get("intent")           # new short key
        or data.get("selected_intent")  # old key
        or data.get("repaired_intent")  # legacy key
        or None
    )

    # 3. If intent repair: the intent must be in candidates
    if decision in _REPAIR_INTENT_DECISIONS:
        if not repaired_intent or repaired_intent not in candidates:
            return GptRepairResult(
                decision="no_repair",
                parse_error=f"repaired_intent_not_in_candidates:{repaired_intent!r}",
                latency_ms=latency_ms,
            )

    repaired_control_intent: str | None = (
        data.get("control")                  # new short key
        or data.get("selected_control_intent")  # old key
        or data.get("repaired_control_intent")  # legacy key
        or None
    )

    # 4. Slot corrections — invalid slot names are dropped; invalid values reject the entry
    raw_sc = data.get("slots") or data.get("slot_corrections")
    slot_corrections: dict | None = None
    slot_corrections_list: tuple[SlotCorrection, ...] | None = None

    if raw_sc:
        original_text = (getattr(nlu, "normalized_text", "") or "").lower()
        original_slot_values: set[str] = {
            str(sv.value).lower() for sv in (getattr(nlu, "slots", ()) or ())
        }

        if isinstance(raw_sc, dict) and raw_sc:
            # Old dict format {"SLOT_NAME": "value"}
            built_dict: dict[str, Any] = {}
            for slot_name, slot_value in raw_sc.items():
                slot_name_upper = str(slot_name).upper()
                if slot_name_upper not in _VALID_SLOT_NAMES:
                    continue  # drop unknown slot names silently
                sv_str = str(slot_value).lower()
                if sv_str not in original_text and sv_str not in original_slot_values:
                    continue  # drop entries with hallucinated values
                built_dict[slot_name] = slot_value
            if built_dict:
                slot_corrections = built_dict

        elif isinstance(raw_sc, list) and raw_sc:
            # New array format — supports both long keys and short keys:
            #   long: {"slot_name": ..., "new_value": ..., "old_value": ..., "operation": ...}
            #   short: {"n": ..., "v": ..., "op": ...}
            parsed_list: list[SlotCorrection] = []
            built_dict_list: dict[str, Any] = {}
            for item in raw_sc:
                if not isinstance(item, dict):
                    continue
                # Resolve short vs long key
                slot_name = str(item.get("n") or item.get("slot_name") or "")
                new_value = item.get("v") if "v" in item else item.get("new_value")
                old_value = item.get("old_value")
                operation = str(item.get("op") or item.get("operation") or "replace")

                slot_name_upper = slot_name.upper()
                if slot_name_upper not in _VALID_SLOT_NAMES:
                    continue  # drop unknown slot names silently

                if new_value is not None:
                    sv_str = str(new_value).lower()
                    if sv_str not in original_text and sv_str not in original_slot_values:
                        continue  # drop entries with hallucinated values
                    built_dict_list[slot_name] = new_value

                parsed_list.append(SlotCorrection(
                    slot_name=slot_name,
                    old_value=str(old_value) if old_value is not None else None,
                    new_value=str(new_value) if new_value is not None else None,
                    operation=operation,
                ))

            if parsed_list:
                slot_corrections_list = tuple(parsed_list)
                slot_corrections = built_dict_list or None

    # If decision requires slot corrections but all were dropped, downgrade gracefully
    if decision in _REPAIR_SLOT_DECISIONS and not slot_corrections_list:
        if decision == "repair_intent_and_slots":
            # Intent repair can still proceed; drop the slot component
            decision = "repair"
        else:
            # Pure slot repair with no valid corrections → no-op
            decision = "no_repair"

    # Fallback type (only meaningful when decision="fallback")
    raw_fallback_type = str(data.get("fallback_type") or "none").strip()
    if decision == "fallback":
        if raw_fallback_type not in _VALID_FALLBACK_TYPES:
            return GptRepairResult(
                decision="no_repair",
                parse_error=f"fallback_type_invalid:{raw_fallback_type!r}",
                latency_ms=latency_ms,
            )
        fallback_type = raw_fallback_type
    else:
        fallback_type = "none"

    # missing_slots: only parsed for decision="missing_info"
    missing_slots: tuple[str, ...] = ()
    if decision == "missing_info":
        raw_missing = data.get("missing")
        if isinstance(raw_missing, list):
            missing_slots = tuple(
                str(s).upper() for s in raw_missing
                if str(s).upper() in _VALID_SLOT_NAMES
            )
        # Normalise to no_repair if nothing actionable
        decision = "no_repair"

    # items[] — optional multi-item array (parse + log only; never applied to cart)
    items: tuple[GptRepairItem, ...] = _parse_items(data.get("items"))

    # Confidence — support both "conf" (short) and "confidence" (long)
    raw_conf = data.get("conf") if "conf" in data else data.get("confidence")
    confidence: float | None = None
    if isinstance(raw_conf, (int, float)):
        confidence = max(0.0, min(1.0, float(raw_conf)))

    # Reason — support both "why" (short) and "reason" (long)
    reason = data.get("why") or data.get("reason") or None

    return GptRepairResult(
        decision=decision,
        repaired_intent=repaired_intent if decision in _REPAIR_INTENT_DECISIONS else None,
        repaired_control_intent=repaired_control_intent,
        slot_corrections=slot_corrections,
        slot_corrections_list=slot_corrections_list,
        confidence=confidence,
        reason=reason,
        latency_ms=latency_ms,
        timeout=False,
        parse_error=None,
        applied=False,
        fallback_type=fallback_type,
        missing_slots=missing_slots,
        items=items,
    )


# ---------------------------------------------------------------------------
# items[] parser
# ---------------------------------------------------------------------------

# Safe quantity range: 1–99 (guard against hallucinated large numbers)
_ITEM_QUANTITY_MIN: int = 1
_ITEM_QUANTITY_MAX: int = 99


def _parse_items(raw: object) -> tuple[GptRepairItem, ...]:
    """Parse the optional items[] array from a GPT response.

    Rules:
    - Non-list or None input returns empty tuple.
    - Entries where "item" is empty/null are dropped.
    - quantity is clamped to [1, 99]; non-integer defaults to 1.
    - sides/modifiers/missing must be lists of strings; malformed lists are
      treated as empty.
    - Order is preserved.
    - Items are logged only; never applied to cart.
    """
    if not isinstance(raw, list):
        return ()

    parsed: list[GptRepairItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item_name = entry.get("item") or ""
        if not isinstance(item_name, str):
            item_name = str(item_name) if item_name else ""
        item_name = item_name.strip()
        if not item_name:
            continue  # drop entries with empty item name

        # Quantity — clamp to safe range
        raw_qty = entry.get("quantity")
        try:
            qty = int(raw_qty) if raw_qty is not None else 1
        except (TypeError, ValueError):
            qty = 1
        qty = max(_ITEM_QUANTITY_MIN, min(_ITEM_QUANTITY_MAX, qty))

        # Optional string fields
        size = _safe_str_or_none(entry.get("size"))
        variant = _safe_str_or_none(entry.get("variant"))

        # List fields (sides, modifiers, missing)
        sides = _str_list(entry.get("sides"))
        modifiers = _str_list(entry.get("modifiers"))
        missing = _str_list(entry.get("missing"))

        parsed.append(GptRepairItem(
            item=item_name,
            quantity=qty,
            size=size,
            variant=variant,
            sides=sides,
            modifiers=modifiers,
            missing=missing,
        ))

    return tuple(parsed)


def _safe_str_or_none(val: object) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _str_list(val: object) -> tuple[str, ...]:
    if not isinstance(val, list):
        return ()
    return tuple(str(v).strip() for v in val if v is not None and str(v).strip())
