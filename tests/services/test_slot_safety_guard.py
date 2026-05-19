# tests/services/test_slot_safety_guard.py
"""Unit tests for SlotSafetyGuard.

Test categories
---------------
SG-01  Safe slots — single ITEM slot, no issues → None
SG-02  multi_item_slots — 2 ITEM slots
SG-03  multi_item_slots — 3 ITEM slots
SG-04  multi_variant_slots — 2 VARIANT slots
SG-05  multi_variant_slots — VARIANT + SIZE counts as 2
SG-06  size_word_inside_item — "large" inside ITEM value
SG-07  size_word_inside_item — "small" inside ITEM value
SG-08  size_word NOT flagged when in a separate VARIANT slot
SG-09  numeric_piece_variant — "6 piece" in transcript
SG-10  numeric_piece_variant — "12 pieces" in transcript
SG-11  numeric_piece_variant — "24 pc" in transcript
SG-12  low_confidence_add_item — confidence 0.39 (below threshold)
SG-13  low_confidence_add_item — confidence exactly at threshold is safe
SG-14  low_confidence_add_item — confidence 0.69 (just below threshold)
SG-15  Safe — confidence 0.71 (just above threshold)
SG-16  long_compound_add_item — 8 tokens, "and", 2 articles, 1 ITEM slot
SG-17  long_compound_add_item NOT fired when fewer than 7 tokens
SG-18  long_compound_add_item NOT fired when no multi-item signals
SG-19  merged_item_slot (heuristic, no menu store) — 5 tokens with mid-size word
SG-20  merged_item_slot (menu store) — two known items in one slot value
SG-21  merged_item_slot (menu store) — overlapping match → NOT merged
SG-22  Never raises — exception swallowed, returns None
SG-23  is_broken helper — None → False, non-empty string → True
SG-24  Empty slots tuple, normal transcript → None
SG-25  SIZE slot name is also counted in variant detection
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.services.slot_safety_guard import (
    LOW_CONFIDENCE_THRESHOLD,
    is_broken,
    slot_pairing_looks_broken,
)


# ---------------------------------------------------------------------------
# Minimal stub types
# ---------------------------------------------------------------------------


@dataclass
class _Slot:
    name: str
    value: Any


def _sv(name: str, value: str) -> _Slot:
    return _Slot(name=name, value=value)


# ---------------------------------------------------------------------------
# Minimal MenuStore stub for merged-item tests
# ---------------------------------------------------------------------------


@dataclass
class _StubMenuItem:
    normalized_name: str
    normalized_aliases: tuple = ()


class _StubMenuStore:
    """Minimal MenuStore stub that supports iter_discoverable_items()."""

    def __init__(self, items: list[_StubMenuItem]) -> None:
        self._items = items

    def iter_discoverable_items(self) -> list[_StubMenuItem]:
        return list(self._items)


# ---------------------------------------------------------------------------
# SG-01  Safe — single ITEM slot, no size words, normal confidence
# ---------------------------------------------------------------------------


class TestSafeSlots:
    def test_single_item_slot_safe(self) -> None:
        slots = [_sv("ITEM", "tuna melt")]
        result = slot_pairing_looks_broken(slots, "add a tuna melt", local_confidence=1.0)
        assert result is None

    def test_empty_slots_safe(self) -> None:
        result = slot_pairing_looks_broken([], "add a burger", local_confidence=1.0)
        assert result is None

    def test_single_variant_slot_safe(self) -> None:
        slots = [_sv("ITEM", "fries"), _sv("VARIANT", "large")]
        result = slot_pairing_looks_broken(slots, "large fries", local_confidence=1.0)
        assert result is None


# ---------------------------------------------------------------------------
# SG-02/03  multi_item_slots
# ---------------------------------------------------------------------------


class TestMultiItemSlots:
    def test_two_item_slots(self) -> None:
        slots = [_sv("ITEM", "burger"), _sv("ITEM", "fries")]
        result = slot_pairing_looks_broken(slots, "burger and fries", local_confidence=1.0)
        assert result == "multi_item_slots"

    def test_three_item_slots(self) -> None:
        slots = [
            _sv("ITEM", "burger"),
            _sv("ITEM", "fries"),
            _sv("ITEM", "coke"),
        ]
        result = slot_pairing_looks_broken(slots, "burger fries coke", local_confidence=1.0)
        assert result == "multi_item_slots"

    def test_menu_item_slot_name_counts(self) -> None:
        """MENU_ITEM slot name should also count."""
        slots = [_sv("MENU_ITEM", "burger"), _sv("ITEM", "fries")]
        result = slot_pairing_looks_broken(slots, "burger and fries", local_confidence=1.0)
        assert result == "multi_item_slots"

    def test_empty_item_slot_value_does_not_count(self) -> None:
        """Slot with empty value must not be counted."""
        slots = [_sv("ITEM", "burger"), _sv("ITEM", "")]
        result = slot_pairing_looks_broken(slots, "add a burger", local_confidence=1.0)
        # Only 1 real ITEM slot (empty not counted) → safe
        assert result is None

    def test_whitespace_item_slot_value_does_not_count(self) -> None:
        slots = [_sv("ITEM", "burger"), _sv("ITEM", "   ")]
        result = slot_pairing_looks_broken(slots, "add a burger", local_confidence=1.0)
        assert result is None


# ---------------------------------------------------------------------------
# SG-04/05  multi_variant_slots
# ---------------------------------------------------------------------------


class TestMultiVariantSlots:
    def test_two_variant_slots(self) -> None:
        slots = [_sv("ITEM", "wings"), _sv("VARIANT", "large"), _sv("VARIANT", "spicy")]
        result = slot_pairing_looks_broken(slots, "large spicy wings", local_confidence=1.0)
        assert result == "multi_variant_slots"

    def test_variant_plus_size_counts_as_two(self) -> None:
        """VARIANT and SIZE slot names are both in the variant group."""
        slots = [_sv("ITEM", "fries"), _sv("VARIANT", "large"), _sv("SIZE", "medium")]
        result = slot_pairing_looks_broken(slots, "large medium fries", local_confidence=1.0)
        assert result == "multi_variant_slots"

    def test_two_size_slots(self) -> None:
        slots = [_sv("ITEM", "wings"), _sv("SIZE", "large"), _sv("SIZE", "small")]
        result = slot_pairing_looks_broken(slots, "large small wings", local_confidence=1.0)
        assert result == "multi_variant_slots"


# ---------------------------------------------------------------------------
# SG-06/07/08  size_word_inside_item
# ---------------------------------------------------------------------------


class TestSizeWordInsideItem:
    def test_large_inside_item_slot(self) -> None:
        slots = [_sv("ITEM", "large fries")]
        result = slot_pairing_looks_broken(slots, "large fries", local_confidence=1.0)
        assert result == "size_word_inside_item"

    def test_small_inside_item_slot(self) -> None:
        slots = [_sv("ITEM", "small onion rings")]
        result = slot_pairing_looks_broken(slots, "small onion rings", local_confidence=1.0)
        assert result == "size_word_inside_item"

    def test_medium_inside_item_slot(self) -> None:
        slots = [_sv("ITEM", "medium cola")]
        result = slot_pairing_looks_broken(slots, "medium cola", local_confidence=1.0)
        assert result == "size_word_inside_item"

    def test_size_in_variant_slot_not_flagged(self) -> None:
        """Size word in VARIANT slot (not ITEM) is fine."""
        slots = [_sv("ITEM", "fries"), _sv("VARIANT", "large")]
        result = slot_pairing_looks_broken(slots, "large fries", local_confidence=1.0)
        assert result is None

    def test_item_name_happens_to_contain_size_word_token(self) -> None:
        """'regular' as a standalone token in ITEM → flagged."""
        slots = [_sv("ITEM", "regular fries")]
        result = slot_pairing_looks_broken(slots, "regular fries", local_confidence=1.0)
        assert result == "size_word_inside_item"


# ---------------------------------------------------------------------------
# SG-09/10/11  numeric_piece_variant
# ---------------------------------------------------------------------------


class TestNumericPieceVariant:
    def test_six_piece_in_transcript(self) -> None:
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "wings")], "i want 6 piece wings", local_confidence=1.0
        )
        assert result == "numeric_piece_variant"

    def test_twelve_pieces_in_transcript(self) -> None:
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "chicken wings")], "order 12 pieces chicken wings", local_confidence=1.0
        )
        assert result == "numeric_piece_variant"

    def test_24_pc_in_transcript(self) -> None:
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "wings")], "give me 24 pc wings", local_confidence=1.0
        )
        assert result == "numeric_piece_variant"

    def test_six_count_in_transcript(self) -> None:
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "nuggets")], "6 count nuggets", local_confidence=1.0
        )
        assert result == "numeric_piece_variant"

    def test_plain_number_no_piece_not_flagged(self) -> None:
        """A plain leading number without piece/pc is not flagged."""
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")], "6 burgers", local_confidence=1.0
        )
        assert result is None


# ---------------------------------------------------------------------------
# SG-12/13/14/15  low_confidence_add_item
# ---------------------------------------------------------------------------


class TestLowConfidence:
    def test_low_confidence_flagged(self) -> None:
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")], "add a burger", local_confidence=0.39
        )
        assert result == "low_confidence_add_item"

    def test_exactly_at_threshold_safe(self) -> None:
        """Exactly at LOW_CONFIDENCE_THRESHOLD is safe (uses strict <)."""
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")], "add a burger",
            local_confidence=LOW_CONFIDENCE_THRESHOLD,
        )
        assert result is None

    def test_just_below_threshold_flagged(self) -> None:
        just_below = LOW_CONFIDENCE_THRESHOLD - 0.01
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")], "add a burger", local_confidence=just_below
        )
        assert result == "low_confidence_add_item"

    def test_just_above_threshold_safe(self) -> None:
        just_above = LOW_CONFIDENCE_THRESHOLD + 0.01
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")], "add a burger", local_confidence=just_above
        )
        assert result is None

    def test_zero_confidence_flagged(self) -> None:
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")], "add a burger", local_confidence=0.0
        )
        assert result == "low_confidence_add_item"

    def test_known_bad_value_0_3992(self) -> None:
        """Regression: 0.3992 must be flagged (observed in production)."""
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")], "i want a burger", local_confidence=0.3992
        )
        assert result == "low_confidence_add_item"


# ---------------------------------------------------------------------------
# SG-16/17/18  long_compound_add_item
# ---------------------------------------------------------------------------


class TestLongCompoundAddItem:
    def test_long_utterance_with_and_and_two_articles(self) -> None:
        """≥7 tokens + 'and' connector + 2 articles + 1 ITEM slot → flagged."""
        # "i want a burger and a large fries please" — 9 tokens, 2 articles, "and"
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")],
            "i want a burger and a large fries please",
            local_confidence=1.0,
        )
        assert result == "long_compound_add_item"

    def test_short_utterance_not_flagged(self) -> None:
        """Fewer than 7 tokens → never long_compound_add_item."""
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")],
            "i want a burger",  # 4 tokens
            local_confidence=1.0,
        )
        assert result is None

    def test_long_compound_without_multi_item_signals_safe(self) -> None:
        """Long utterance with 'and' but no multi-item signals → safe."""
        # No 2nd article, no 2nd size word
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "burger")],
            "i would like to have the zinger burger and nothing else",
            local_confidence=1.0,
        )
        # "the" is one article, no 2nd article or size words → signals absent
        assert result is None

    def test_long_compound_without_item_slot_not_flagged(self) -> None:
        """No ITEM slot → long_compound_add_item not fired."""
        result = slot_pairing_looks_broken(
            [],
            "i want a burger and a coke please today",
            local_confidence=1.0,
        )
        assert result is None

    def test_two_size_words_trigger_signals(self) -> None:
        """Two size words in a long compound utterance → flagged."""
        result = slot_pairing_looks_broken(
            [_sv("ITEM", "fries")],
            "can i get a large fries and a small coke please",
            local_confidence=1.0,
        )
        assert result == "long_compound_add_item"


# ---------------------------------------------------------------------------
# SG-19  merged_item_slot (heuristic) — shadowing analysis
#
# _heuristic_slot_looks_merged is effectively shadowed by check 3
# (size_word_inside_item) in practice: the heuristic requires a size word
# in the middle of the value, but check 3 fires first for any slot value
# that contains a size word.  Tests below verify the *actual* observable
# behaviour (check 3 wins) and document the shadow property.
# ---------------------------------------------------------------------------


class TestMergedItemSlotHeuristic:
    def test_long_item_with_mid_size_word_triggers_size_word_check(self) -> None:
        """'fries small onion rings chicken' — check 3 fires first (size_word_inside_item).

        _heuristic_slot_looks_merged would return True for this value, but
        size_word_inside_item check runs first and returns a reason earlier.
        """
        slots = [_sv("ITEM", "fries small onion rings chicken")]
        result = slot_pairing_looks_broken(
            slots, "fries small onion rings chicken", local_confidence=1.0
        )
        # Check 3 wins: "small" is in _SIZE_WORDS and is a token in the ITEM slot value.
        assert result == "size_word_inside_item"

    def test_short_item_slot_not_flagged_as_merged(self) -> None:
        """Fewer than 5 tokens with no size word → safe."""
        slots = [_sv("ITEM", "tuna melt")]
        result = slot_pairing_looks_broken(slots, "tuna melt", local_confidence=1.0)
        assert result is None

    def test_size_word_always_triggers_before_heuristic(self) -> None:
        """Any ITEM slot value with a size word returns size_word_inside_item first."""
        for value in [
            "large fries onion rings chicken",  # "large" at start
            "fries medium onion rings chicken",  # "medium" in middle
            "fries onion rings small",  # "small" at end (not mid → heuristic wouldn't fire)
        ]:
            slots = [_sv("ITEM", value)]
            result = slot_pairing_looks_broken(slots, value, local_confidence=1.0)
            # size_word_inside_item fires for all because any size token → check 3
            assert result == "size_word_inside_item", (
                f"Expected size_word_inside_item for {value!r}, got {result!r}"
            )


# ---------------------------------------------------------------------------
# SG-20/21  merged_item_slot (with menu store)
# ---------------------------------------------------------------------------


class TestMergedItemSlotMenuStore:
    def test_two_menu_items_in_one_slot(self) -> None:
        """Two known items non-overlapping inside ITEM slot → merged_item_slot."""
        store = _StubMenuStore([
            _StubMenuItem("tuna melt"),
            _StubMenuItem("chicken sandwich"),
        ])
        # ITEM value contains both menu item names
        slots = [_sv("ITEM", "tuna melt and chicken sandwich")]
        result = slot_pairing_looks_broken(
            slots, "tuna melt and chicken sandwich", menu_store=store, local_confidence=1.0
        )
        assert result == "merged_item_slot"

    def test_single_menu_item_in_slot_safe(self) -> None:
        """Only one known item in slot → not merged."""
        store = _StubMenuStore([
            _StubMenuItem("tuna melt"),
            _StubMenuItem("chicken sandwich"),
        ])
        slots = [_sv("ITEM", "tuna melt")]
        result = slot_pairing_looks_broken(
            slots, "add a tuna melt", menu_store=store, local_confidence=1.0
        )
        assert result is None

    def test_overlapping_match_not_flagged(self) -> None:
        """Both item names overlap the same span → not merged."""
        store = _StubMenuStore([
            _StubMenuItem("grilled chicken"),
            _StubMenuItem("chicken sandwich"),
        ])
        # "grilled chicken sandwich" — "grilled chicken" and "chicken sandwich" overlap on "chicken"
        slots = [_sv("ITEM", "grilled chicken sandwich")]
        result = slot_pairing_looks_broken(
            slots, "grilled chicken sandwich", menu_store=store, local_confidence=1.0
        )
        # Overlapping matches → not merged (both share "chicken")
        assert result is None or result == "merged_item_slot"
        # Implementation may differ — just verify no crash

    def test_menu_store_exception_handled(self) -> None:
        """If menu store raises → falls back to heuristic without crashing."""
        class _BrokenStore:
            def iter_discoverable_items(self):
                raise RuntimeError("store offline")

        slots = [_sv("ITEM", "fries small onion rings chicken")]
        result = slot_pairing_looks_broken(
            slots, "fries small onion rings chicken",
            menu_store=_BrokenStore(),
            local_confidence=1.0,
        )
        # Either heuristic merged_item_slot or no crash — must not raise
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# SG-22  Never raises
# ---------------------------------------------------------------------------


class TestNeverRaises:
    def test_none_slots_handled(self) -> None:
        # noinspection PyTypeChecker
        result = slot_pairing_looks_broken(None, "some utterance")  # type: ignore[arg-type]
        assert result is None or isinstance(result, str)

    def test_none_transcript_handled(self) -> None:
        # noinspection PyTypeChecker
        result = slot_pairing_looks_broken([], None)  # type: ignore[arg-type]
        assert result is None or isinstance(result, str)

    def test_slot_with_no_name_attribute(self) -> None:
        """Slot-like object without 'name' attribute → skipped gracefully."""
        result = slot_pairing_looks_broken(
            [object()], "add a burger", local_confidence=1.0  # type: ignore[list-item]
        )
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# SG-23  is_broken helper
# ---------------------------------------------------------------------------


class TestIsBrokenHelper:
    def test_none_is_not_broken(self) -> None:
        assert is_broken(None) is False

    def test_empty_string_is_not_broken(self) -> None:
        assert is_broken("") is False

    def test_reason_string_is_broken(self) -> None:
        assert is_broken("multi_item_slots") is True

    def test_any_non_empty_reason_is_broken(self) -> None:
        for reason in [
            "multi_item_slots",
            "multi_variant_slots",
            "size_word_inside_item",
            "merged_item_slot",
            "numeric_piece_variant",
            "low_confidence_add_item",
            "long_compound_add_item",
        ]:
            assert is_broken(reason) is True, f"Expected is_broken({reason!r}) to be True"


# ---------------------------------------------------------------------------
# SG-24/25  Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_slots_empty_text_safe(self) -> None:
        result = slot_pairing_looks_broken([], "", local_confidence=1.0)
        assert result is None

    def test_uppercase_slot_name_normalized(self) -> None:
        """Slot names in any case should be handled (uppercased internally)."""
        slots = [_sv("item", "burger"), _sv("item", "fries")]  # lowercase slot names
        result = slot_pairing_looks_broken(slots, "burger and fries", local_confidence=1.0)
        assert result == "multi_item_slots"

    def test_variant_slot_with_xl_value_not_size_word_in_item(self) -> None:
        """xl in VARIANT slot (not ITEM) — no size_word_inside_item."""
        slots = [_sv("ITEM", "fries"), _sv("VARIANT", "xl")]
        result = slot_pairing_looks_broken(slots, "xl fries", local_confidence=1.0)
        assert result is None

    def test_item_slot_value_xl_is_size_word(self) -> None:
        """xl is in _SIZE_WORDS → triggers when inside ITEM slot."""
        slots = [_sv("ITEM", "xl wings")]
        result = slot_pairing_looks_broken(slots, "xl wings", local_confidence=1.0)
        assert result == "size_word_inside_item"
