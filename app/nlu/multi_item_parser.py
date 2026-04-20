# app/nlu/multi_item_parser.py
"""
Splits a multi-item utterance into individual item segments.

Examples:
  "a chicken taco with coke and extra cheese and 2 chicken burgers with no onions"
  → [
      ParsedItemSegment(raw="a chicken taco with coke and extra cheese", quantity=1),
      ParsedItemSegment(raw="2 chicken burgers with no onions", quantity=2),
    ]

The parser uses a lightweight heuristic approach:
1. Look for ITEM/MENU_ITEM slots extracted by the NLU slot model
2. Split on "and" boundaries where a new item entity starts
3. Preserve modifier/side associations with the correct item
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text

QUANTITY_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Connectors that MAY separate distinct items
_ITEM_SEPARATORS = re.compile(
    r"\band\b|\bplus\b|\balso\b|\bwith\s+(?:a|an|\d+|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParsedItemSegment:
    """One item segment parsed from a multi-item utterance."""
    raw_text: str                       # original text segment
    item_slot_value: str | None = None  # ITEM slot value if detected
    quantity: int | None = None         # quantity if detected
    slots: tuple[SlotValue, ...] = ()   # all slots belonging to this segment


def _extract_item_slot_values(slots: Sequence[SlotValue]) -> list[SlotValue]:
    """Return all ITEM/MENU_ITEM slot values in slot order."""
    return [
        slot for slot in slots
        if str(slot.name).upper() in {"ITEM", "MENU_ITEM"}
        and isinstance(slot.value, str) and slot.value.strip()
    ]


def _extract_leading_quantity(text: str) -> tuple[int | None, str]:
    """Extract a leading quantity from text, return (quantity, remaining_text)."""
    text = text.strip()
    # Try digit
    m = re.match(r"^(\d+)\s+", text)
    if m:
        return int(m.group(1)), text[m.end():].strip()
    # Try word
    for word, qty in QUANTITY_WORDS.items():
        pattern = rf"^{re.escape(word)}\s+"
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            return qty, text[m.end():].strip()
    return None, text


def parse_multi_item_utterance(
    normalized_text: str,
    slots: Sequence[SlotValue],
) -> list[ParsedItemSegment]:
    """
    Parse a single utterance into multiple item segments.

    Strategy:
    1. Find all ITEM slots with character offsets.
    2. If 0 or 1 ITEM slots → return single segment (no split needed).
    3. If 2+ ITEM slots → split the text at each ITEM slot boundary.

    Each segment inherits the non-ITEM slots that fall within its character range.
    """
    item_slots = _extract_item_slot_values(slots)

    # ── Fast path: 0 or 1 item → no multi-item parsing needed ──
    if len(item_slots) <= 1:
        return []

    # ── Check we have character offsets for splitting ──
    items_with_offsets = [
        s for s in item_slots
        if s.start is not None and s.end is not None
    ]
    if len(items_with_offsets) <= 1:
        # No offset info → fall back to regex-based heuristic splitting
        return _split_by_heuristic(normalized_text, item_slots)

    # ── Split by ITEM slot offsets ──
    items_with_offsets.sort(key=lambda s: s.start)
    segments: list[ParsedItemSegment] = []

    for i, item_slot in enumerate(items_with_offsets):
        # Determine segment boundaries
        # Look backward from the item slot to find where this segment starts
        if i == 0:
            seg_start = 0
        else:
            # Start from just after the previous segment's effective end
            prev_end = items_with_offsets[i - 1].end
            # Find the "and" or separator between prev item and this one
            between = normalized_text[prev_end:item_slot.start]
            sep_match = re.search(r"\band\b|\bplus\b|\balso\b|,", between)
            if sep_match:
                seg_start = prev_end + sep_match.end()
            else:
                seg_start = prev_end

        # Segment ends at the start of the next item's segment, or end of text
        if i < len(items_with_offsets) - 1:
            next_start = items_with_offsets[i + 1].start
            # Look backward to find separator
            between = normalized_text[item_slot.end:next_start]
            sep_match = re.search(r"\band\b|\bplus\b|\balso\b|,", between)
            if sep_match:
                seg_end = item_slot.end + sep_match.start()
            else:
                seg_end = next_start
        else:
            seg_end = len(normalized_text)

        # Look further back for quantity prefix
        prefix_start = _find_quantity_prefix_start(normalized_text, seg_start, item_slot.start)
        if prefix_start < seg_start:
            seg_start = prefix_start

        raw_segment = normalized_text[seg_start:seg_end].strip()
        raw_segment = re.sub(r"^(and|plus|also|,)\s*", "", raw_segment, flags=re.IGNORECASE).strip()

        # Extract quantity from segment
        quantity, _ = _extract_leading_quantity(raw_segment)

        # Collect non-ITEM slots that belong to this segment
        seg_slots = _slots_in_range(slots, seg_start, seg_end, exclude_labels={"ITEM", "MENU_ITEM"})

        # Re-inject the ITEM slot so downstream handlers can use it for slot-based resolution
        item_slot_copy = SlotValue(
            name=item_slot.name,
            value=item_slot.value,
            raw=item_slot.raw,
            start=item_slot.start,
            end=item_slot.end,
            confidence=item_slot.confidence,
        )
        seg_slots_with_item = [item_slot_copy] + seg_slots

        segments.append(ParsedItemSegment(
            raw_text=raw_segment,
            item_slot_value=str(item_slot.value).strip(),
            quantity=quantity,
            slots=tuple(seg_slots_with_item),
        ))

    return segments if len(segments) >= 2 else []


def _find_quantity_prefix_start(text: str, seg_start: int, item_start: int) -> int:
    """Look before seg_start for a quantity that modifies this item."""
    prefix = text[max(0, seg_start - 15):item_start].strip()
    if not prefix:
        return seg_start

    # Check for digit or quantity word right before the item
    m = re.search(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*$", prefix, re.IGNORECASE)
    if m:
        actual_start = max(0, seg_start - 15) + m.start()
        return actual_start
    return seg_start


def _slots_in_range(
    slots: Sequence[SlotValue],
    start: int,
    end: int,
    exclude_labels: set[str] | None = None,
) -> list[SlotValue]:
    """Return slots whose offsets fall within [start, end)."""
    exclude = {label.upper() for label in (exclude_labels or set())}
    result = []
    for slot in slots:
        if str(slot.name).upper() in exclude:
            continue
        if slot.start is not None and slot.end is not None:
            if slot.start >= start and slot.end <= end:
                result.append(slot)
        # Slots without offsets get assigned to the first segment by default
    return result


def _split_by_heuristic(
    normalized_text: str,
    item_slots: list[SlotValue],
) -> list[ParsedItemSegment]:
    """
    Fallback heuristic: split on 'and' boundaries where we can
    match item slot values to sub-segments.
    """
    # Split by 'and' and commas, but only at item boundaries
    parts = re.split(r"\s+and\s+|\s*,\s*", normalized_text)
    if len(parts) < 2:
        return []

    item_values_normalized = {normalize_text(str(s.value)): s for s in item_slots}
    segments: list[ParsedItemSegment] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        matched_item = None
        for item_norm, item_slot in item_values_normalized.items():
            if item_norm in normalize_text(part):
                matched_item = item_slot
                break

        if matched_item is not None:
            quantity, _ = _extract_leading_quantity(part)
            segments.append(ParsedItemSegment(
                raw_text=part,
                item_slot_value=str(matched_item.value).strip(),
                quantity=quantity,
                slots=(),
            ))

    return segments if len(segments) >= 2 else []
