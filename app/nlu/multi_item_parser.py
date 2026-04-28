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

from app.menu.store import MenuStore
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

_ATTACHMENT_PREFIXES = (
    "with ",
    "extra ",
    "more ",
    "double ",
    "no ",
    "without ",
    "light ",
    "less ",
    "on the side ",
)

# Tail-only tokens that indicate the upcoming item is attached to the
# previous one (e.g. "with X", "extra Y", "no Z").
_ATTACHMENT_TAIL_TOKENS_1: frozenset[str] = frozenset(
    {"with", "extra", "more", "double", "no", "without", "light", "less", "hold", "remove"}
)
_ATTACHMENT_TAIL_TOKENS_2: frozenset[tuple[str, str]] = frozenset(
    {("on", "the"), ("hold", "the"), ("remove", "the"), ("on", "side"), ("with", "the")}
)
# Tail-only tokens that signal a *new* item is starting (so the upcoming
# item slot is NOT attached). "and", "plus", "also" alone — or "and a",
# "plus 2", etc.
_RESTART_TAIL_TOKENS_1: frozenset[str] = frozenset({"and", "plus", "also", "then", "or"})
_RESTART_QUANTITY_WORDS: frozenset[str] = frozenset(
    {"a", "an", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
)

_RESTART_WITH_QUANTITY = re.compile(
    r"(?:and|plus|also)(?:\s+also)?\s+(?:a|an|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*$",
    re.IGNORECASE,
)
_BOUNDARY_SEPARATORS = re.compile(r"\b(?:and|plus|also|then)\b|,", re.IGNORECASE)


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


def _slot_key(slot: SlotValue) -> tuple[object, ...]:
    if slot.start is not None and slot.end is not None:
        return (str(slot.name).upper(), slot.start, slot.end)
    return (
        str(slot.name).upper(),
        normalize_text(str(slot.value)),
        slot.start,
        slot.end,
    )


def _is_explicit_item_value(normalized_value: str, menu_store: MenuStore | None) -> bool:
    if menu_store is None or not normalized_value:
        return True

    if menu_store.find_entity(normalized_value, allowed_types={"item"}):
        return True

    if menu_store.find_item_exact(normalized_value) is not None:
        return True

    if menu_store.find_item_ids_by_alias(normalized_value):
        return True

    if menu_store.find_item_ids_by_voice_label(normalized_value):
        return True

    return False


def _collect_candidate_item_slots(
    normalized_text: str,
    slots: Sequence[SlotValue],
    menu_store: MenuStore | None,
) -> list[SlotValue]:
    candidates: list[SlotValue] = []
    seen: set[tuple[object, ...]] = set()

    def add(slot: SlotValue) -> None:
        key = _slot_key(slot)
        if key in seen:
            return
        seen.add(key)
        candidates.append(slot)

    for slot in _extract_item_slot_values(slots):
        normalized_value = normalize_text(str(slot.value))
        if not normalized_value:
            continue
        if not _is_explicit_item_value(normalized_value, menu_store):
            continue
        add(slot)

    if menu_store is None:
        return candidates

    for mention in menu_store.find_discoverable_item_mentions(normalized_text):
        add(
            SlotValue(
                name="ITEM",
                value=mention["item_name"],
                raw=mention["matched_text"],
                start=mention["start"],
                end=mention["end"],
                confidence=1.0,
            )
        )

    return candidates


def _slot_looks_attached(text: str, slot: SlotValue) -> bool:
    """
    Decide whether the upcoming item slot is part of the previous item
    (e.g. "burger WITH cheese") or the start of a new item.

    Implementation note: the historical heuristic was

        " with " in compact   # anywhere in the 60-char lookback

    That over-attaches when a "with" appears far upstream in the same
    utterance (e.g. "taco with coke and jelly a chicken burger" — the
    "with" before "coke" must NOT cause "chicken burger" to be treated
    as attached). The fix is to look only at the immediate tail tokens.
    """
    if slot.start is None:
        return False

    lookback = text[max(0, slot.start - 60):slot.start].lower()
    compact = re.sub(r"\s+", " ", lookback).strip()
    if not compact:
        return False

    tokens = compact.split()
    if not tokens:
        return False

    # Restart patterns explicitly start a new item — never attached.
    #   "...and a chicken burger", "...plus 2 burgers", "...also one ..."
    if (
        len(tokens) >= 2
        and tokens[-2] in _RESTART_TAIL_TOKENS_1
        and (tokens[-1] in _RESTART_QUANTITY_WORDS or tokens[-1].isdigit())
    ):
        return False
    if tokens[-1] in _RESTART_TAIL_TOKENS_1:
        return False

    # Immediate attachment markers: the next slot is part of the previous
    # item.
    if tokens[-1] in _ATTACHMENT_TAIL_TOKENS_1:
        return True
    if (
        len(tokens) >= 2
        and (tokens[-2], tokens[-1]) in _ATTACHMENT_TAIL_TOKENS_2
    ):
        return True
    # "with a", "extra an", "no the", "more a" etc. — an attachment word
    # followed by an article. Still attached.
    if (
        len(tokens) >= 2
        and tokens[-1] in {"a", "an", "the"}
        and tokens[-2] in _ATTACHMENT_TAIL_TOKENS_1
    ):
        return True

    # Bare "a"/"an" without a preceding restart or attachment connector
    # still signals a new item ("...jelly a chicken burger") — treat as
    # not attached.
    return False


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


def _has_explicit_split_boundary(
    text: str,
    previous_slot: SlotValue,
    next_slot: SlotValue,
) -> bool:
    if previous_slot.end is None or next_slot.start is None:
        return False

    between = text[previous_slot.end:next_slot.start]
    if not between:
        return False

    compact = re.sub(r"\s+", " ", between).strip()
    if not compact:
        return False

    if _BOUNDARY_SEPARATORS.search(compact):
        return True

    quantity, remainder = _extract_leading_quantity(compact)
    return quantity is not None and not remainder


def parse_multi_item_utterance(
    normalized_text: str,
    slots: Sequence[SlotValue],
    menu_store: MenuStore | None = None,
) -> list[ParsedItemSegment]:
    """
    Parse a single utterance into multiple item segments.

    Strategy:
    1. Find all ITEM slots with character offsets.
    2. If 0 or 1 ITEM slots → return single segment (no split needed).
    3. If 2+ ITEM slots → split the text at each ITEM slot boundary.

    Each segment inherits the non-ITEM slots that fall within its character range.
    """
    item_slots = _collect_candidate_item_slots(normalized_text, slots, menu_store)
    split_item_slots = [
        slot for slot in item_slots
        if not _slot_looks_attached(normalized_text, slot)
    ]

    # ── Fast path: 0 or 1 item → no multi-item parsing needed ──
    if len(split_item_slots) <= 1:
        return []

    # ── Check we have character offsets for splitting ──
    items_with_offsets = [
        s for s in split_item_slots
        if s.start is not None and s.end is not None
    ]
    if len(items_with_offsets) <= 1:
        # No offset info → fall back to regex-based heuristic splitting
        return _split_by_heuristic(normalized_text, split_item_slots)

    # ── Split by ITEM slot offsets ──
    items_with_offsets.sort(key=lambda s: s.start)
    anchored_items: list[SlotValue] = []
    for slot in items_with_offsets:
        if not anchored_items:
            anchored_items.append(slot)
            continue
        if _has_explicit_split_boundary(normalized_text, anchored_items[-1], slot):
            anchored_items.append(slot)

    if len(anchored_items) <= 1:
        return []

    segments: list[ParsedItemSegment] = []
    boundary_slot_keys = {_slot_key(slot) for slot in anchored_items}

    for i, item_slot in enumerate(anchored_items):
        # Determine segment boundaries.
        #
        # Strategy (in priority order):
        #   1) Honour the quantity prefix immediately before THIS item slot
        #      ("a", "an", "2", "two", ...). The segment starts AT that
        #      quantity word, regardless of any "and" further back. This
        #      keeps modifier lists like "...and jelly" attached to the
        #      previous item even when the next item is preceded by a
        #      bare "a" instead of "and a".
        #   2) Otherwise fall back to the LAST connector ("and", ",", ...)
        #      between the previous item end and this item start.
        if i == 0:
            seg_start = 0
        else:
            prev_end = anchored_items[i - 1].end
            qty_prefix_start = _quantity_prefix_start(normalized_text, item_slot.start)
            if qty_prefix_start is not None and qty_prefix_start >= prev_end:
                seg_start = qty_prefix_start
            else:
                between = normalized_text[prev_end:item_slot.start]
                sep_matches = list(re.finditer(r"\band\b|\bplus\b|\balso\b|,", between))
                if sep_matches:
                    seg_start = prev_end + sep_matches[-1].end()
                else:
                    seg_start = prev_end

        # Segment ends at the start of the next item's segment.
        if i < len(anchored_items) - 1:
            next_slot = anchored_items[i + 1]
            next_start = next_slot.start
            qty_prefix_start = _quantity_prefix_start(normalized_text, next_start)
            if qty_prefix_start is not None and qty_prefix_start > item_slot.end:
                # Stop just before the next item's quantity prefix so that
                # any trailing modifiers ("...and jelly") stay with this
                # segment.
                seg_end = qty_prefix_start
            else:
                between = normalized_text[item_slot.end:next_start]
                sep_matches = list(re.finditer(r"\band\b|\bplus\b|\balso\b|,", between))
                if sep_matches:
                    seg_end = item_slot.end + sep_matches[-1].start()
                else:
                    seg_end = next_start
        else:
            seg_end = len(normalized_text)

        raw_segment = normalized_text[seg_start:seg_end].strip()
        # Strip leading and trailing connectors. Loop the trailing strip so
        # compound connectors like "and also" are removed in one pass.
        raw_segment = re.sub(r"^(and|plus|also|then|,)\s*", "", raw_segment, flags=re.IGNORECASE).strip()
        prev_segment: str | None = None
        while raw_segment != prev_segment:
            prev_segment = raw_segment
            raw_segment = re.sub(
                r"(?:\s+(?:and|plus|also|then)|,)\s*$",
                "",
                raw_segment,
                flags=re.IGNORECASE,
            ).strip()

        # Extract quantity from segment
        quantity, _ = _extract_leading_quantity(raw_segment)

        # Collect all in-range slots except the boundary item anchors for split segments.
        seg_slots = _slots_in_range(
            slots,
            seg_start,
            seg_end,
            exclude_slot_keys=boundary_slot_keys,
            include_offsetless=i == 0,
        )

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


_QTY_PREFIX_TAIL_RE = re.compile(
    r"(?:^|\s)((?:a|an|the|\d+|one|two|three|four|five|six|seven|eight|nine|ten))\s+$",
    re.IGNORECASE,
)


def _quantity_prefix_start(text: str, item_start: int) -> int | None:
    """
    Return the absolute index of the quantity-prefix word ("a", "an",
    "the", a digit, or one of the spelled-out numbers 1–10) that appears
    immediately before `item_start`, separated only by whitespace.

    Returns None when no such prefix exists. The caller uses this to
    snap segment boundaries to the start of the quantity rather than to
    the last "and"/comma — important for utterances like
    "...jelly a chicken burger..." where there is no connector at all.
    """
    if item_start <= 0:
        return None

    window_start = max(0, item_start - 15)
    backwindow = text[window_start:item_start]
    m = _QTY_PREFIX_TAIL_RE.search(backwindow)
    if not m:
        return None
    return window_start + m.start(1)


def _slots_in_range(
    slots: Sequence[SlotValue],
    start: int,
    end: int,
    exclude_slot_keys: set[tuple[object, ...]] | None = None,
    include_offsetless: bool = False,
) -> list[SlotValue]:
    """Return slots whose offsets fall within [start, end)."""
    excluded = exclude_slot_keys or set()
    result = []
    for slot in slots:
        if _slot_key(slot) in excluded:
            continue
        if slot.start is not None and slot.end is not None:
            if slot.start >= start and slot.end <= end:
                result.append(slot)
        elif include_offsetless:
            result.append(slot)
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
        if any(part.lower().startswith(prefix) for prefix in _ATTACHMENT_PREFIXES):
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
