# app/services/slot_safety_guard.py
"""SlotSafetyGuard — detect slot configurations that are unsafe for direct execution.

When the local NLU produces slots that look broken (multiple ITEM slots, size words
inside an ITEM slot value, "6 piece wings" quantity misparse, etc.) the legacy
parse_multi_item_utterance path must not execute those slots blindly — doing so
creates wrong cart mutations like "Added 6 Tuna Melt".

Public API
----------
slot_pairing_looks_broken(slots, transcript, menu_store, *, local_confidence) -> str | None

Returns a short reason string when the slot configuration looks unsafe, or None
when the slots appear safe for direct execution.

Reason codes
------------
multi_item_slots          — 2+ ITEM slots detected
multi_variant_slots       — 2+ VARIANT/SIZE slots detected
size_word_inside_item     — ITEM slot value contains a size word
merged_item_slot          — ITEM slot appears to contain two menu item names
numeric_piece_variant     — transcript has "N piece" pattern (wings variant)
low_confidence_add_item   — local NLU confidence < LOW_CONFIDENCE_THRESHOLD
long_compound_add_item    — long utterance with compound markers + multi-item signals

Design
------
* Pure function — no I/O, no side effects, never raises.
* Does NOT call GPT or any external service.
* Conservative: returns None (safe) when uncertain.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from app.nlu.nlu_result import SlotValue
    from app.menu.store import MenuStore

# Minimum confidence below which we treat local NLU as unreliable on ADD_ITEM.
LOW_CONFIDENCE_THRESHOLD: float = 0.70

# Size words that should NOT appear inside an ITEM slot value as standalone tokens.
_SIZE_WORDS: frozenset[str] = frozenset({
    "small", "medium", "large", "regular", "xl", "extra large",
    "sm", "md", "lg",
})

# Compound utterance markers.
_COMPOUND_MARKERS: tuple[str, ...] = (" and ", " plus ", " also ", " with ", ", ")

# Pattern for numeric piece variants like "6 piece", "12 pieces", "24 pc", "50 piece".
_NUMERIC_PIECE_RE = re.compile(
    r"\b(\d+)\s*(?:piece|pieces|pc|pcs|oz|ounce|ounces|ct|count)\b",
    re.IGNORECASE,
)

# Minimum tokens for an utterance to be considered "long and compound".
_LONG_COMPOUND_TOKENS = 7


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def slot_pairing_looks_broken(
    slots: "Sequence[SlotValue]",
    transcript: str,
    menu_store: "MenuStore | None" = None,
    *,
    local_confidence: float = 1.0,
) -> "str | None":
    """Return a reason string when the slot configuration looks unsafe.

    Parameters
    ----------
    slots:
        The NLU slot values for the current turn.
    transcript:
        The normalized user utterance.
    menu_store:
        Optional MenuStore for item-name cross-checking (enables merged_item_slot).
    local_confidence:
        Local NLU confidence score (0.0–1.0).  Values below
        LOW_CONFIDENCE_THRESHOLD trigger the low_confidence_add_item reason.

    Returns
    -------
    A reason string when the slot configuration looks broken.
    None when the slots appear safe for direct execution.

    Never raises.
    """
    try:
        return _check(slots, transcript, menu_store, local_confidence)
    except Exception:
        return None


def is_broken(reason: "str | None") -> bool:
    """Convenience predicate — True when reason is non-empty."""
    return bool(reason)


# ---------------------------------------------------------------------------
# Internal checker
# ---------------------------------------------------------------------------


def _check(
    slots: "Sequence[SlotValue]",
    transcript: str,
    menu_store: "MenuStore | None",
    local_confidence: float,
) -> "str | None":
    item_slots = _slots_by_name(slots, {"ITEM", "MENU_ITEM"})
    variant_slots = _slots_by_name(slots, {"VARIANT", "SIZE"})
    text = (transcript or "").strip().lower()

    # ── 1. Multiple ITEM slots ─────────────────────────────────────────────
    if len(item_slots) >= 2:
        return "multi_item_slots"

    # ── 2. Multiple VARIANT/SIZE slots ────────────────────────────────────
    if len(variant_slots) >= 2:
        return "multi_variant_slots"

    # ── 3. ITEM slot value contains a size word ───────────────────────────
    for slot in item_slots:
        slot_value = _slot_str(slot)
        if _item_value_contains_size_word(slot_value):
            return "size_word_inside_item"

    # ── 4. ITEM slot appears to contain two menu items merged ─────────────
    for slot in item_slots:
        slot_value = _slot_str(slot)
        if menu_store is not None and _slot_looks_merged(slot_value, menu_store):
            return "merged_item_slot"
        elif menu_store is None and _heuristic_slot_looks_merged(slot_value):
            return "merged_item_slot"

    # ── 5. Transcript has "N piece" pattern (wings variant) ───────────────
    if _NUMERIC_PIECE_RE.search(text):
        return "numeric_piece_variant"

    # ── 6. Low confidence on ADD_ITEM ─────────────────────────────────────
    if local_confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence_add_item"

    # ── 7. Long compound utterance with multi-item signals ────────────────
    tokens = text.split()
    if len(tokens) >= _LONG_COMPOUND_TOKENS and _is_compound(text) and len(item_slots) >= 1:
        # Only flag when there are also implicit multi-item signals
        if _has_multi_item_signals(text):
            return "long_compound_add_item"

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slots_by_name(
    slots: "Sequence[SlotValue]",
    names: set[str],
) -> "list[SlotValue]":
    """Return slots whose name (uppercased) is in the given set."""
    result = []
    for slot in (slots or []):
        name = str(getattr(slot, "name", "")).upper()
        if name in names:
            value = getattr(slot, "value", None)
            if value is not None and str(value).strip():
                result.append(slot)
    return result


def _slot_str(slot: Any) -> str:
    """Return the slot value as a stripped lowercase string."""
    return str(getattr(slot, "value", "")).strip().lower()


def _item_value_contains_size_word(slot_value: str) -> bool:
    """Return True when a size word appears as a standalone token inside the slot value."""
    tokens = set(slot_value.split())
    return bool(tokens & _SIZE_WORDS)


def _slot_looks_merged(slot_value: str, menu_store: "MenuStore") -> bool:
    """Return True when the slot value appears to span two distinct menu items.

    Uses MenuStore substring matching: if two non-overlapping known item names
    both appear within the slot value, the slot is likely merged.
    """
    if not slot_value or not menu_store:
        return False

    found_items: list[tuple[int, int, str]] = []  # (start, end, name)

    try:
        for item in menu_store.iter_discoverable_items():
            labels: list[str] = [item.normalized_name]
            labels.extend(getattr(item, "normalized_aliases", ()) or ())
            for label in labels:
                if not label or len(label) < 3:
                    continue
                idx = slot_value.find(label)
                if idx >= 0:
                    found_items.append((idx, idx + len(label), label))
    except Exception:
        return False

    if len(found_items) < 2:
        return False

    # Check for non-overlapping pairs
    found_items.sort(key=lambda x: x[0])
    for i in range(len(found_items)):
        for j in range(i + 1, len(found_items)):
            s1, e1, _ = found_items[i]
            s2, e2, _ = found_items[j]
            if e1 <= s2:  # Non-overlapping
                return True

    return False


def _heuristic_slot_looks_merged(slot_value: str) -> bool:
    """Fallback merged-item detection without a menu store.

    Checks whether the slot value:
    - has 5+ tokens (long enough to contain two items)
    - contains a size word in the middle (signals item boundary)
    """
    tokens = slot_value.split()
    if len(tokens) < 5:
        return False
    # Size word appearing after the first token (not at start) → likely boundary
    for i, tok in enumerate(tokens[1:], start=1):
        if tok in _SIZE_WORDS and i < len(tokens) - 1:
            return True
    return False


def _is_compound(text: str) -> bool:
    """Return True when the text contains compound markers."""
    return any(m in text for m in _COMPOUND_MARKERS)


def _has_multi_item_signals(text: str) -> bool:
    """Return True when the text shows strong multi-item signals beyond compound markers."""
    tokens = text.split()
    # Count distinct "article + word" pairs — signals multiple item starts
    article_count = sum(1 for t in tokens if t in {"a", "an", "the"})
    if article_count >= 2:
        return True
    # Count size words — multiple sizes → multiple items
    size_count = sum(1 for t in tokens if t in _SIZE_WORDS)
    if size_count >= 2:
        return True
    return False
