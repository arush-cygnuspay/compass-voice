# tests/nlu/test_quantity_parser_side_phrases.py
"""Tests for side-selection quantity phrase parsing.

Covers the multiplier phrases and x-notation added to normalize_quantity:
  twice/thrice/double/triple → integer
  x2 / 2x / ×2 variants → integer
  decimal inputs → None (rejected, not rounded)
  existing phrases still work (regression guard)
"""
import pytest

from app.nlu.matching.quantity_parser import normalize_quantity, SPECIAL_QUANTITIES


# ---------------------------------------------------------------------------
# New multiplier words
# ---------------------------------------------------------------------------

class TestMultiplierWords:
    def test_twice_returns_2(self):
        assert normalize_quantity("twice") == 2

    def test_thrice_returns_3(self):
        assert normalize_quantity("thrice") == 3

    def test_double_returns_2(self):
        assert normalize_quantity("double") == 2

    def test_triple_returns_3(self):
        assert normalize_quantity("triple") == 3

    def test_TWICE_uppercase_returns_2(self):
        # normalize_quantity lowercases before lookup
        assert normalize_quantity("TWICE") == 2

    def test_Double_mixed_case_returns_2(self):
        assert normalize_quantity("Double") == 2

    def test_multiplier_words_in_special_quantities_dict(self):
        assert SPECIAL_QUANTITIES["twice"] == 2
        assert SPECIAL_QUANTITIES["thrice"] == 3
        assert SPECIAL_QUANTITIES["double"] == 2
        assert SPECIAL_QUANTITIES["triple"] == 3


# ---------------------------------------------------------------------------
# x-notation
# ---------------------------------------------------------------------------

class TestXNotation:
    def test_x2_prefix(self):
        assert normalize_quantity("x2") == 2

    def test_x3_prefix(self):
        assert normalize_quantity("x3") == 3

    def test_2x_suffix(self):
        assert normalize_quantity("2x") == 2

    def test_3x_suffix(self):
        assert normalize_quantity("3x") == 3

    def test_unicode_times_prefix(self):
        assert normalize_quantity("×2") == 2

    def test_unicode_times_suffix(self):
        assert normalize_quantity("2×") == 2

    def test_x_with_space_prefix(self):
        assert normalize_quantity("x 2") == 2

    def test_x_with_space_suffix(self):
        assert normalize_quantity("2 x") == 2

    def test_X_uppercase_prefix(self):
        assert normalize_quantity("X2") == 2

    def test_x0_returns_none(self):
        # Zero quantity makes no sense; should be rejected
        assert normalize_quantity("x0") is None

    def test_x1_returns_1(self):
        assert normalize_quantity("x1") == 1


# ---------------------------------------------------------------------------
# Decimal / float inputs — must be rejected (return None), not rounded
# ---------------------------------------------------------------------------

class TestDecimalRejection:
    def test_zero_point_one_string_returns_none(self):
        # "0.1" as ASR text should not produce a quantity
        assert normalize_quantity("0.1") is None

    def test_one_point_five_string_returns_none(self):
        assert normalize_quantity("1.5") is None

    def test_two_point_zero_string_returns_none(self):
        # "2.0" could come from ASR rendering a float; must not silently return 2
        assert normalize_quantity("2.0") is None

    def test_point_five_string_returns_none(self):
        assert normalize_quantity(".5") is None


# ---------------------------------------------------------------------------
# Existing phrases — regression guard
# ---------------------------------------------------------------------------

class TestExistingPhrasesUnchanged:
    @pytest.mark.parametrize("phrase,expected", [
        ("2", 2),
        ("3", 3),
        ("10", 10),
        ("two", 2),
        ("three", 3),
        ("twelve", 12),
        ("a", 1),
        ("an", 1),
        ("single", 1),
        ("couple", 2),
        ("half dozen", 6),
        ("a dozen", 12),
        ("two dozen", 24),
        ("2 pcs", 2),
        ("3 pieces", 3),
        ("two times", 2),       # _first_number_word catches "two"
        ("three times", 3),
        ("2 times", 2),
        ("all three", 3),       # _first_number_word catches "three"
        ("two of them", 2),
    ])
    def test_phrase_maps_to_expected(self, phrase, expected):
        assert normalize_quantity(phrase) == expected

    @pytest.mark.parametrize("phrase", [
        "no",
        "none",
        "skip",
        "",
        "   ",
    ])
    def test_non_quantity_phrases_return_none(self, phrase):
        assert normalize_quantity(phrase) is None
