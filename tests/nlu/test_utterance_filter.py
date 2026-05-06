# tests/nlu/test_utterance_filter.py
"""Unit tests for FillerFilter.

All tests are deterministic and require no fixtures, DB, or NLU model.
"""
from __future__ import annotations

import pytest

from app.nlu.utterance_filter import (
    DEFAULT_FILTER,
    FillerFilter,
    _FILLER_STOP_TOKENS,
    _FILLER_PHRASES,
    _NON_ECHABLE_CONTROL_PHRASES,
)

FILTER = DEFAULT_FILTER


# ===========================================================================
# A. strip_ordering_filler
# ===========================================================================


class TestStripOrderingFiller:
    @pytest.mark.parametrize("raw,expected", [
        ("i would like to order a chicken burger", "chicken burger"),
        ("i would like a chicken burger", "chicken burger"),
        ("i want a dragon burger", "dragon burger"),
        ("give me a large fries", "large fries"),
        ("let me get a bacon cheeseburger", "bacon cheeseburger"),
        ("can i get a mango shake", "mango shake"),
        ("okay then give me a chicken burger", "chicken burger"),
        ("just a coke", "coke"),
        ("please give me fries", "fries"),
        ("with a side of coleslaw", "side of coleslaw"),
        ("to order a burger", "burger"),
        ("ok then", ""),
        ("okay", ""),
        ("and a", ""),
        ("chicken burger", "chicken burger"),  # no filler → unchanged
        ("", ""),
    ])
    def test_strip_filler(self, raw, expected):
        result = FILTER.strip_ordering_filler(raw)
        assert result == expected, f"strip({raw!r}) → {result!r}, expected {expected!r}"

    def test_does_not_strip_burger_noun(self):
        """'burger' must never be stripped."""
        result = FILTER.strip_ordering_filler("i want a chicken burger")
        assert "burger" in result

    def test_does_not_strip_cheese_noun(self):
        result = FILTER.strip_ordering_filler("i would like extra cheese")
        assert "cheese" in result

    def test_does_not_strip_bun_noun(self):
        result = FILTER.strip_ordering_filler("i want a sesame bun")
        assert "bun" in result

    def test_does_not_strip_onions_noun(self):
        result = FILTER.strip_ordering_filler("give me no onions")
        assert "onions" in result

    def test_does_not_strip_fries_noun(self):
        result = FILTER.strip_ordering_filler("i would like large fries")
        assert "fries" in result


# ===========================================================================
# B. is_filler_only
# ===========================================================================


class TestIsFillerOnly:
    @pytest.mark.parametrize("text", [
        "to order a",
        "okay then",
        "okay then give me a",
        "okay so",
        "i would like to order",
        "i want",
        "give me",
        "",
        "just",
        "and the",
        "please",
        "with a",
    ])
    def test_structural_filler_is_filler_only(self, text):
        assert FILTER.is_filler_only(text), f"{text!r} should be filler-only"

    @pytest.mark.parametrize("text", [
        # Control phrases that must not be echoed
        "no skip that",
        "can you repeat",
        "add done",
        "repeat",
        "repeat that",
        "skip that",
        "no skip",
        "done",
        "no done",
    ])
    def test_control_phrases_are_filler_only(self, text):
        assert FILTER.is_filler_only(text), (
            f"{text!r} should be filler-only (control phrase)"
        )

    @pytest.mark.parametrize("text", [
        "dragon burger",
        "bacon burger",
        "mango shake",
        "large fries",
        "chicken burger",
        "extra cheese",
        "sesame bun",
    ])
    def test_real_items_are_not_filler_only(self, text):
        assert not FILTER.is_filler_only(text), (
            f"{text!r} must NOT be filler-only — it is a real menu candidate"
        )

    def test_none_is_filler_only(self):
        assert FILTER.is_filler_only("")

    def test_single_article_is_filler_only(self):
        assert FILTER.is_filler_only("a")
        assert FILTER.is_filler_only("the")


# ===========================================================================
# C. strip_unmatched
# ===========================================================================


class TestStripUnmatched:
    def test_removes_filler_only_values(self):
        raw = ["to order a", "okay then", "dragon burger", "i want"]
        result = FILTER.strip_unmatched(raw)
        assert result == ["dragon burger"]

    def test_removes_control_phrases(self):
        raw = ["no skip that", "can you repeat", "add done"]
        result = FILTER.strip_unmatched(raw)
        assert result == []

    def test_preserves_real_unmatched_items(self):
        raw = ["dragon burger", "bacon cheeseburger", "mango smoothie"]
        result = FILTER.strip_unmatched(raw)
        assert result == raw

    def test_empty_list_returns_empty(self):
        assert FILTER.strip_unmatched([]) == []

    def test_generator_input_accepted(self):
        result = FILTER.strip_unmatched(v for v in ["dragon burger", "okay then"])
        assert result == ["dragon burger"]

    def test_none_strings_filtered(self):
        result = FILTER.strip_unmatched(["", "dragon burger", ""])
        assert result == ["dragon burger"]

    def test_mixed_bag(self):
        raw = [
            "no skip that",       # control phrase → remove
            "chicken burger",     # real item → keep
            "to order a",         # structural filler → remove
            "can you repeat",     # control phrase → remove
            "add done",           # control phrase → remove
            "large fries",        # real item → keep
        ]
        result = FILTER.strip_unmatched(raw)
        assert result == ["chicken burger", "large fries"]


# ===========================================================================
# D. Defensive: filler phrase list must not contain menu nouns
# ===========================================================================

_MENU_NOUNS: frozenset[str] = frozenset({
    "burger", "cheese", "bun", "onions", "onion", "chicken", "fries",
    "coke", "cola", "drink", "shake", "coffee", "sandwich", "pizza",
    "beef", "pork", "fish", "lettuce", "tomato", "bacon", "sauce",
    "mayo", "ketchup", "mustard", "pickle", "pickles", "mushroom",
    "pepperoni", "sausage", "ham", "turkey", "shrimp", "mango",
    "extra", "large", "medium", "small", "regular",
})


class TestFillerListSafety:
    def test_filler_phrases_contain_no_menu_nouns(self):
        """Filler phrases must only contain structural words, not menu nouns."""
        violations: list[tuple[str, str]] = []
        for phrase in _FILLER_PHRASES:
            for noun in _MENU_NOUNS:
                if noun in phrase.lower().split():
                    violations.append((phrase, noun))

        assert not violations, (
            "Filler phrases contain menu nouns — remove them:\n"
            + "\n".join(f"  {phrase!r} contains {noun!r}" for phrase, noun in violations)
        )

    def test_stop_tokens_contain_no_menu_nouns(self):
        """Stop-word token set must not contain menu nouns."""
        overlap = _FILLER_STOP_TOKENS & _MENU_NOUNS
        assert not overlap, (
            f"Filler stop tokens contain menu nouns: {overlap}"
        )

    def test_non_echable_control_phrases_contain_no_bare_menu_nouns(self):
        """Non-echable phrase set should not accidentally block real menu items."""
        _single_word_items = {n for n in _MENU_NOUNS if " " not in n}
        # Multi-word phrases are fine; only flag bare single-token entries
        bare_violations = _NON_ECHABLE_CONTROL_PHRASES & _single_word_items
        assert not bare_violations, (
            f"Non-echable set contains bare menu nouns: {bare_violations}"
        )


# ===========================================================================
# E. Singleton
# ===========================================================================


class TestSingleton:
    def test_default_filter_is_filler_filter_instance(self):
        assert isinstance(DEFAULT_FILTER, FillerFilter)

    def test_repeated_calls_are_idempotent(self):
        r1 = FILTER.is_filler_only("dragon burger")
        r2 = FILTER.is_filler_only("dragon burger")
        assert r1 == r2 == False
