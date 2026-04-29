# tests/nlu/test_matching_package.py
"""Smoke tests for the app.nlu.matching package.

Validates:
- normalize_text contract
- tokenize contract
- matcher predicates
- scorer
- quantity_parser
- index (slot helpers)
- backward-compat wrappers in app.utils.* and app.menu.slot_helpers
"""
from __future__ import annotations

import pytest


# ── normalization ─────────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lowercases(self) -> None:
        from app.nlu.matching.normalization import normalize_text
        assert normalize_text("HELLO") == "hello"

    def test_strips_punctuation(self) -> None:
        from app.nlu.matching.normalization import normalize_text
        assert normalize_text("hello, world!") == "hello world"

    def test_normalizes_whitespace(self) -> None:
        from app.nlu.matching.normalization import normalize_text
        assert normalize_text("  a  b  ") == "a b"

    def test_empty_returns_empty(self) -> None:
        from app.nlu.matching.normalization import normalize_text
        assert normalize_text("") == ""

    def test_none_like_falsy(self) -> None:
        from app.nlu.matching.normalization import normalize_text
        assert normalize_text("") == ""


class TestTokenize:
    def test_basic_split(self) -> None:
        from app.nlu.matching.normalization import tokenize
        assert "burger" in tokenize("a burger please")

    def test_stop_words_removed(self) -> None:
        from app.nlu.matching.normalization import tokenize
        tokens = tokenize("a burger with sauce")
        assert "a" not in tokens
        assert "with" not in tokens
        assert "burger" in tokens

    def test_singularization(self) -> None:
        from app.nlu.matching.normalization import tokenize
        # "tomatoes" → "tomato" via oes rule
        assert "tomato" in tokenize("tomatoes")

    def test_empty_returns_empty_list(self) -> None:
        from app.nlu.matching.normalization import tokenize
        assert tokenize("") == []

    def test_digits_removed(self) -> None:
        from app.nlu.matching.normalization import tokenize
        tokens = tokenize("2 burgers")
        assert "2" not in tokens
        assert "burger" in tokens

    def test_single_char_removed(self) -> None:
        from app.nlu.matching.normalization import tokenize
        tokens = tokenize("i want a burger")
        assert "i" not in tokens


# ── matcher ───────────────────────────────────────────────────────────────────

class TestExactMatch:
    def test_hit(self) -> None:
        from app.nlu.matching.matcher import exact_match
        assert exact_match("burger", {"burger": ["Burger"]}) == ["Burger"]

    def test_miss_returns_none(self) -> None:
        from app.nlu.matching.matcher import exact_match
        assert exact_match("pizza", {"burger": ["Burger"]}) is None


class TestTokenMatch:
    def test_single_token_hit(self) -> None:
        from app.nlu.matching.matcher import token_match
        assert token_match("big burger", {"burger": ["Burger"]}) == ["Burger"]

    def test_miss_returns_none(self) -> None:
        from app.nlu.matching.matcher import token_match
        assert token_match("pizza", {"burger": ["Burger"]}) is None


class TestResolveChoice:
    def test_exact_takes_priority(self) -> None:
        from app.nlu.matching.matcher import resolve_choice
        lookup = {"big burger": ["Big Burger"], "burger": ["Burger"]}
        assert resolve_choice("big burger", lookup) == ["Big Burger"]

    def test_token_fallback(self) -> None:
        from app.nlu.matching.matcher import resolve_choice
        assert resolve_choice("i want a burger", {"burger": ["Burger"]}) == ["Burger"]

    def test_no_match_returns_none(self) -> None:
        from app.nlu.matching.matcher import resolve_choice
        assert resolve_choice("pizza", {"burger": ["Burger"]}) is None


class TestSkipChoiceAnswer:
    @pytest.mark.parametrize("text", ["no", "none", "skip", "skip it", "nothing", "without it"])
    def test_skip_phrases(self, text: str) -> None:
        from app.nlu.matching.matcher import _looks_like_skip_choice_answer
        assert _looks_like_skip_choice_answer(text)

    def test_non_skip(self) -> None:
        from app.nlu.matching.matcher import _looks_like_skip_choice_answer
        assert not _looks_like_skip_choice_answer("burger please")


class TestIsStrongTokenMatch:
    def test_exact_match(self) -> None:
        from app.nlu.matching.matcher import is_strong_token_match
        assert is_strong_token_match("burger", "burger")

    def test_subset_match(self) -> None:
        from app.nlu.matching.matcher import is_strong_token_match
        # multi-token user tokens must be subset of choice tokens
        assert is_strong_token_match("grilled chicken", "grilled chicken sandwich")

    def test_single_token_must_be_single_choice(self) -> None:
        from app.nlu.matching.matcher import is_strong_token_match
        # "chicken" (1 token) should NOT match "grilled chicken" (2 tokens)
        assert not is_strong_token_match("chicken", "grilled chicken")

    def test_empty_returns_false(self) -> None:
        from app.nlu.matching.matcher import is_strong_token_match
        assert not is_strong_token_match("", "burger")


class TestIsControlledPartialMatch:
    def test_multi_token_contained(self) -> None:
        from app.nlu.matching.matcher import is_controlled_partial_match
        # "grilled chicken" tokens in "grilled chicken sandwich" tokens
        assert is_controlled_partial_match("grilled chicken", "grilled chicken sandwich")

    def test_short_text_rejected(self) -> None:
        from app.nlu.matching.matcher import is_controlled_partial_match
        assert not is_controlled_partial_match("ok", "okay burger")

    def test_single_token_rejected(self) -> None:
        from app.nlu.matching.matcher import is_controlled_partial_match
        assert not is_controlled_partial_match("burger", "double burger")


# ── scorer ────────────────────────────────────────────────────────────────────

class TestScoreItemNormalized:
    def test_exact_match_returns_max(self) -> None:
        from app.nlu.matching.scorer import score_item_normalized
        assert score_item_normalized("burger", "burger") == 10.0

    def test_empty_returns_zero(self) -> None:
        from app.nlu.matching.scorer import score_item_normalized
        assert score_item_normalized("", "burger") == 0.0
        assert score_item_normalized("burger", "") == 0.0

    def test_partial_overlap_positive(self) -> None:
        from app.nlu.matching.scorer import score_item_normalized
        assert score_item_normalized("chicken burger", "double chicken burger") > 0.0

    def test_no_overlap_returns_zero(self) -> None:
        from app.nlu.matching.scorer import score_item_normalized
        assert score_item_normalized("pizza", "burger") == 0.0


class TestScoreItem:
    def test_normalizes_case(self) -> None:
        from app.nlu.matching.scorer import score_item
        assert score_item("BURGER", "burger") == 10.0

    def test_normalizes_whitespace(self) -> None:
        from app.nlu.matching.scorer import score_item
        assert score_item("  burger  ", "burger") == 10.0


# ── quantity_parser ───────────────────────────────────────────────────────────

class TestNormalizeQuantity:
    @pytest.mark.parametrize("text,expected", [
        ("2", 2),
        ("two", 2),
        ("2 pcs", 2),
        ("half dozen", 6),
        ("one dozen", 12),
        ("forty nine", 49),
        ("a", 1),
        ("couple", 2),
    ])
    def test_known_inputs(self, text: str, expected: int) -> None:
        from app.nlu.matching.quantity_parser import normalize_quantity
        assert normalize_quantity(text) == expected

    def test_empty_returns_none(self) -> None:
        from app.nlu.matching.quantity_parser import normalize_quantity
        assert normalize_quantity("") is None

    def test_unrecognized_returns_none(self) -> None:
        from app.nlu.matching.quantity_parser import normalize_quantity
        assert normalize_quantity("some burgers") is None


class TestDetectQuantity:
    def test_exact(self) -> None:
        from app.nlu.matching.quantity_parser import detect_quantity
        result = detect_quantity("3")
        assert result == {"type": "exact", "value": 3}

    def test_incremental(self) -> None:
        from app.nlu.matching.quantity_parser import detect_quantity
        result = detect_quantity("one more")
        assert result is not None
        assert result["type"] == "incremental"

    def test_vague(self) -> None:
        from app.nlu.matching.quantity_parser import detect_quantity
        result = detect_quantity("a few")
        assert result == {"type": "vague", "value": None}

    def test_empty_returns_none(self) -> None:
        from app.nlu.matching.quantity_parser import detect_quantity
        assert detect_quantity("") is None


class TestExtractLeadingQuantityPhrase:
    def test_digit_prefix(self) -> None:
        from app.nlu.matching.quantity_parser import extract_leading_quantity_phrase
        result = extract_leading_quantity_phrase("2 burgers please")
        assert result is not None
        qty, rest, token = result
        assert qty == 2
        assert "burger" in rest

    def test_word_prefix(self) -> None:
        from app.nlu.matching.quantity_parser import extract_leading_quantity_phrase
        result = extract_leading_quantity_phrase("three fries")
        assert result is not None
        assert result[0] == 3

    def test_no_prefix_returns_none(self) -> None:
        from app.nlu.matching.quantity_parser import extract_leading_quantity_phrase
        assert extract_leading_quantity_phrase("add a burger") is None


class TestExtractWeightQuantity:
    def test_half_pound(self) -> None:
        from app.nlu.matching.quantity_parser import extract_weight_quantity
        result = extract_weight_quantity("half pound")
        assert result == {"value": 0.5, "unit": "lb", "ounces": 8.0}

    def test_digit_oz(self) -> None:
        from app.nlu.matching.quantity_parser import extract_weight_quantity
        result = extract_weight_quantity("6 oz")
        assert result is not None
        assert result["ounces"] == 6.0

    def test_empty_returns_none(self) -> None:
        from app.nlu.matching.quantity_parser import extract_weight_quantity
        assert extract_weight_quantity("") is None


# ── index (slot helpers) ──────────────────────────────────────────────────────

class FakeSlot:
    def __init__(self, name: str, value) -> None:
        self.name = name
        self.value = value


class TestSlotValues:
    def test_returns_matching_values(self) -> None:
        from app.nlu.matching.index import slot_values
        slots = [FakeSlot("ITEM", "burger"), FakeSlot("SIZE", "large")]
        assert slot_values(slots, "item") == ["burger"]

    def test_case_insensitive_label(self) -> None:
        from app.nlu.matching.index import slot_values
        slots = [FakeSlot("item", "burger")]
        assert slot_values(slots, "ITEM") == ["burger"]

    def test_deduplicates(self) -> None:
        from app.nlu.matching.index import slot_values
        slots = [FakeSlot("ITEM", "burger"), FakeSlot("ITEM", "burger")]
        assert slot_values(slots, "item") == ["burger"]

    def test_skips_non_string_values(self) -> None:
        from app.nlu.matching.index import slot_values
        slots = [FakeSlot("ITEM", 42)]
        assert slot_values(slots, "item") == []


class TestFirstSlotValue:
    def test_returns_first(self) -> None:
        from app.nlu.matching.index import first_slot_value
        slots = [FakeSlot("ITEM", "burger"), FakeSlot("ITEM", "fries")]
        assert first_slot_value(slots, "item") == "burger"

    def test_returns_none_when_missing(self) -> None:
        from app.nlu.matching.index import first_slot_value
        slots = [FakeSlot("SIZE", "large")]
        assert first_slot_value(slots, "item") is None


# ── backward-compat wrappers ──────────────────────────────────────────────────

class TestBackwardCompatWrappers:
    def test_token_matcher_tokenize(self) -> None:
        from app.utils.token_matcher import tokenize
        assert isinstance(tokenize("burger"), list)

    def test_token_matcher_is_strong_token_match(self) -> None:
        from app.utils.token_matcher import is_strong_token_match
        assert is_strong_token_match("burger", "burger")

    def test_token_matcher_is_controlled_partial_match(self) -> None:
        from app.utils.token_matcher import is_controlled_partial_match
        assert callable(is_controlled_partial_match)

    def test_token_matcher_token_overlap_score(self) -> None:
        from app.utils.token_matcher import token_overlap_score
        assert token_overlap_score("burger", "burger") >= 1

    def test_item_matching_score_item(self) -> None:
        from app.utils.item_matching import score_item
        assert score_item("burger", "burger") == 10.0

    def test_item_matching_score_item_normalized(self) -> None:
        from app.utils.item_matching import score_item_normalized
        assert score_item_normalized("burger", "burger") == 10.0

    def test_quantity_detection_normalize_quantity(self) -> None:
        from app.utils.quantity_detection import normalize_quantity
        assert normalize_quantity("two") == 2

    def test_quantity_detection_detect_quantity(self) -> None:
        from app.utils.quantity_detection import detect_quantity
        assert detect_quantity("3") == {"type": "exact", "value": 3}

    def test_quantity_detection_unit_pattern(self) -> None:
        from app.utils.quantity_detection import UNIT_PATTERN
        assert isinstance(UNIT_PATTERN, str)
        assert "dozen" in UNIT_PATTERN

    def test_deterministic_matcher_resolve_choice(self) -> None:
        from app.utils.deterministic_matcher import resolve_choice
        assert resolve_choice("burger", {"burger": ["Burger"]}) == ["Burger"]

    def test_slot_helpers_slot_values(self) -> None:
        from app.menu.slot_helpers import slot_values
        slots = [FakeSlot("ITEM", "burger")]
        assert slot_values(slots, "item") == ["burger"]

    def test_slot_helpers_first_slot_value(self) -> None:
        from app.menu.slot_helpers import first_slot_value
        slots = [FakeSlot("ITEM", "burger")]
        assert first_slot_value(slots, "item") == "burger"
