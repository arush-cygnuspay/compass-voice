# app/nlu/semantic_repair/add_item_output_parser.py
"""Parse and sanitise GPT ADD_ITEM extractor output into GptAddItemPlan.

Parsing contract
----------------
* Never throws exceptions to the caller — all errors return a safe no_repair plan.
* Items, sides, and modifiers are validated against utterance_text + choices.
  Hallucinated values (not in utterance and not in choices) are dropped.
* Modifier size/variant is stripped unconditionally (current schema does not
  support modifier variants); a parse note is appended.
* Quantity must be a positive integer; invalid/zero values become None.
* Items are capped to max_items; overflow logged in parse_notes.
* Global slot names are validated against the allowed set.
* This parser does NOT do menu validation — that is a future PR.
"""
from __future__ import annotations

import json
from typing import Any

from app.nlu.semantic_repair.add_item_extractor import (
    ADD_ITEM_NOT_CALLED,
    GptAddItem,
    GptAddItemChild,
    GptAddItemPlan,
)

# Decision values the GPT extractor may return
_VALID_DECISIONS: frozenset[str] = frozenset({
    "ok", "repair", "missing_info", "fallback", "no_repair",
})

# Valid intents the extractor may name
_VALID_INTENTS: frozenset[str] = frozenset({"add_item"})

# Valid operation values for sides/modifiers
_VALID_OPERATIONS: frozenset[str] = frozenset({"add", "remove", "replace"})

# Slot names the extractor is allowed to emit in global_slots
_ALLOWED_GLOBAL_SLOTS: frozenset[str] = frozenset({
    "ITEM", "MODIFIER", "SIDE", "QUANTITY", "SIZE", "VARIANT",
})

# Maximum characters for a size or variant value (sanity clamp)
_MAX_STR_LEN: int = 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_lower(s: Any) -> str:
    return (s or "").lower()


def _is_substring_of(value: str, *, utterance: str, choices: tuple[str, ...]) -> bool:
    """Return True if value appears verbatim in utterance or in any choice."""
    v = value.strip().lower()
    if not v:
        return False
    if v in utterance.lower():
        return True
    return any(v in c.lower() for c in choices)


def _clamp_quantity(q: Any) -> int | None:
    """Return a positive int quantity or None for invalid/zero."""
    try:
        val = int(q)
    except (TypeError, ValueError):
        return None
    return val if val >= 1 else None


def _safe_str(s: Any, max_len: int = _MAX_STR_LEN) -> str | None:
    """Return stripped string or None if empty/not-a-string."""
    if not isinstance(s, str):
        return None
    stripped = s.strip()[:max_len]
    return stripped or None


def _safe_operation(op: Any) -> str:
    """Coerce to a valid operation string, defaulting to 'add'."""
    return op if isinstance(op, str) and op in _VALID_OPERATIONS else "add"


# ---------------------------------------------------------------------------
# Child (side / modifier) parser
# ---------------------------------------------------------------------------


def _parse_child(
    raw: Any,
    *,
    utterance: str,
    choices: tuple[str, ...],
    strip_size_variant: bool = False,
) -> GptAddItemChild | None:
    """Parse one side or modifier dict entry.

    Returns None if the entry should be dropped (e.g. hallucinated name).
    When strip_size_variant=True (modifiers), size and variant are stripped
    unconditionally and a parse note is added.
    """
    if not isinstance(raw, dict):
        return None

    name_raw = raw.get("name") or raw.get("item") or ""
    name = _safe_str(name_raw)
    if not name:
        return None

    # Drop hallucinated names not found in utterance or choices
    if not _is_substring_of(name, utterance=utterance, choices=choices):
        return None

    operation = _safe_operation(raw.get("operation"))
    quantity = _clamp_quantity(raw.get("quantity"))
    notes: list[str] = []

    if strip_size_variant:
        # Modifier size/variant is stripped; schema does not support it yet.
        size: str | None = None
        variant: str | None = None
        if raw.get("size") or raw.get("variant"):
            notes.append("modifier_size_dropped")
    else:
        # Side size/variant: keep only if found in utterance or choices
        size_raw = _safe_str(raw.get("size"))
        if size_raw and _is_substring_of(size_raw, utterance=utterance, choices=choices):
            size = size_raw
        else:
            if size_raw:
                notes.append(f"side_size_dropped:{size_raw}")
            size = None

        variant_raw = _safe_str(raw.get("variant"))
        if variant_raw and _is_substring_of(
            variant_raw, utterance=utterance, choices=choices
        ):
            variant = variant_raw
        else:
            if variant_raw:
                notes.append(f"side_variant_dropped:{variant_raw}")
            variant = None

    # Parse nested modifiers on a child (usually empty list)
    child_mods: list[str] = []
    for m in (raw.get("modifiers") or []):
        if isinstance(m, str) and m.strip():
            child_mods.append(m.strip())

    return GptAddItemChild(
        name=name,
        operation=operation,
        quantity=quantity,
        size=size,
        variant=variant,
        modifiers=tuple(child_mods),
        parse_notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Item parser
# ---------------------------------------------------------------------------


def _parse_item(
    raw: Any,
    *,
    utterance: str,
    choices: tuple[str, ...],
    cart_item_names: tuple[str, ...],
) -> GptAddItem | None:
    """Parse one item dict entry.

    Returns None if the item name is empty or hallucinated.
    """
    if not isinstance(raw, dict):
        return None

    item_name = _safe_str(raw.get("item") or raw.get("name") or "")
    if not item_name:
        return None

    # Item name must appear in utterance OR in the active cart (e.g. the user
    # is modifying a cart item by name).
    item_in_utterance = _is_substring_of(
        item_name, utterance=utterance, choices=choices
    )
    item_in_cart = any(
        item_name.lower() in ci.lower() for ci in cart_item_names
    )
    if not item_in_utterance and not item_in_cart:
        return None

    quantity = _clamp_quantity(raw.get("quantity"))
    notes: list[str] = list(raw.get("parse_notes") or [])

    # Item-level size: validate against utterance + choices
    size_raw = _safe_str(raw.get("size"))
    if size_raw and _is_substring_of(size_raw, utterance=utterance, choices=choices):
        size: str | None = size_raw
    else:
        if size_raw:
            notes.append(f"item_size_dropped:{size_raw}")
        size = None

    # Item-level variant: validate against utterance + choices
    variant_raw = _safe_str(raw.get("variant"))
    if variant_raw and _is_substring_of(
        variant_raw, utterance=utterance, choices=choices
    ):
        variant: str | None = variant_raw
    else:
        if variant_raw:
            notes.append(f"item_variant_dropped:{variant_raw}")
        variant = None

    # Parse sides
    sides: list[GptAddItemChild] = []
    for s in (raw.get("sides") or []):
        child = _parse_child(
            s, utterance=utterance, choices=choices, strip_size_variant=False
        )
        if child is not None:
            sides.append(child)

    # Parse modifiers (strip size/variant unconditionally)
    modifiers: list[GptAddItemChild] = []
    for m in (raw.get("modifiers") or []):
        child = _parse_child(
            m, utterance=utterance, choices=choices, strip_size_variant=True
        )
        if child is not None:
            modifiers.append(child)

    # Missing slots for this item
    missing: list[str] = []
    for ms in (raw.get("missing") or []):
        if isinstance(ms, str) and ms.strip():
            missing.append(ms.strip().upper())

    return GptAddItem(
        item=item_name,
        quantity=quantity,
        size=size,
        variant=variant,
        sides=tuple(sides),
        modifiers=tuple(modifiers),
        missing=tuple(missing),
        parse_notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Global slots parser
# ---------------------------------------------------------------------------


def _parse_global_slots(raw: Any) -> tuple[Any, ...]:
    """Parse global_slots list, keeping only known slot names."""
    if not isinstance(raw, list):
        return ()
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("n") or entry.get("name") or "").upper()
        if name not in _ALLOWED_GLOBAL_SLOTS:
            continue
        out.append(entry)
    return tuple(out)


# ---------------------------------------------------------------------------
# Main parser entry point
# ---------------------------------------------------------------------------


def parse_add_item_output(
    raw: str,
    *,
    utterance_text: str,
    choices: tuple[str, ...] = (),
    cart_item_names: tuple[str, ...] = (),
    latency_ms: float = 0.0,
    max_items: int = 8,
) -> GptAddItemPlan:
    """Parse GPT ADD_ITEM extractor output into a GptAddItemPlan.

    Never raises — all parse errors return a safe no_repair plan.

    Parameters
    ----------
    raw:
        Raw string returned by OpenAI (expected to be JSON).
    utterance_text:
        The normalised customer utterance; used to validate extracted values.
    choices:
        Curated choice labels available in the current FSM state (e.g. side
        names, modifier names).  Values found here are always valid even if
        absent from utterance_text.
    cart_item_names:
        Names of items currently in the cart; used to allow GPT to reference
        existing cart items without the name appearing in utterance_text.
    latency_ms:
        Time from request send to first byte received (ms); populated by caller.
    max_items:
        Maximum number of items[] entries to keep.
    """
    utterance_lower = (utterance_text or "").strip()
    parse_notes: list[str] = []

    # --- JSON decode ---
    try:
        # Strip markdown fences if present
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Remove first and last fence lines
            stripped = "\n".join(
                lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            )
        data: Any = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        return GptAddItemPlan(
            decision="no_repair",
            eligible=True,
            parse_error=f"json_decode:{exc}",
            latency_ms=latency_ms,
            total_ms=latency_ms,
        )

    if not isinstance(data, dict):
        return GptAddItemPlan(
            decision="no_repair",
            eligible=True,
            parse_error="json_not_object",
            latency_ms=latency_ms,
            total_ms=latency_ms,
        )

    # --- requires_handler_validation gate ---
    rhv = data.get("requires_handler_validation") or data.get("rhv")
    if rhv is not True:
        return GptAddItemPlan(
            decision="no_repair",
            eligible=True,
            parse_error="requires_handler_validation",
            latency_ms=latency_ms,
            total_ms=latency_ms,
        )

    # --- Decision ---
    decision_raw = data.get("decision", "no_repair")
    decision = decision_raw if decision_raw in _VALID_DECISIONS else "no_repair"

    # --- Intent ---
    intent_raw = data.get("intent")
    intent: str | None = intent_raw if intent_raw in _VALID_INTENTS else None

    # --- Items ---
    raw_items = data.get("items") or []
    if not isinstance(raw_items, list):
        raw_items = []

    parsed_items: list[GptAddItem] = []
    for entry in raw_items:
        item = _parse_item(
            entry,
            utterance=utterance_lower,
            choices=choices,
            cart_item_names=cart_item_names,
        )
        if item is not None:
            parsed_items.append(item)

    # Cap to max_items
    if len(parsed_items) > max_items:
        overflow = len(parsed_items) - max_items
        parsed_items = parsed_items[:max_items]
        parse_notes.append(f"items_truncated:{overflow}")

    # --- Global slots ---
    global_slots = _parse_global_slots(data.get("global_slots"))

    # --- Missing slots (global level) ---
    missing: list[str] = []
    for ms in (data.get("missing") or []):
        if isinstance(ms, str) and ms.strip():
            missing.append(ms.strip().upper())

    # --- Fallback type ---
    fallback_type = data.get("fallback_type", "none")
    if not isinstance(fallback_type, str):
        fallback_type = "none"

    # --- Confidence ---
    confidence_raw = data.get("confidence")
    try:
        confidence: float | None = float(confidence_raw) if confidence_raw is not None else None
    except (TypeError, ValueError):
        confidence = None

    # --- Reason (capped at 200 chars) ---
    reason_raw = data.get("reason")
    reason: str | None = (
        str(reason_raw)[:200] if reason_raw is not None else None
    )

    return GptAddItemPlan(
        decision=decision,
        intent=intent,
        items=tuple(parsed_items),
        global_slots=global_slots,
        missing=tuple(missing),
        fallback_type=fallback_type,
        confidence=confidence,
        reason=reason,
        latency_ms=latency_ms,
        total_ms=latency_ms,
        eligible=True,
        parse_notes=tuple(parse_notes),
    )
