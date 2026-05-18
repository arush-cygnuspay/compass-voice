# tests/nlu/semantic_repair/test_output_parser_items.py
"""Tests for items[] parsing in output_parser (Part 6)."""
from __future__ import annotations

import json

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.nlu.semantic_repair.gpt_repair_result import GptRepairItem, GptRepairResult
from app.nlu.semantic_repair.output_parser import _parse_items, parse_output
from app.nlu.semantic_repair.prompt_builder import get_candidates
from app.state_machine.models.conversation_state import ConversationState


def _nlu(text: str = "give me a burger") -> NLUResult:
    return NLUResult(
        effective_intent=Intent.UNKNOWN,
        intent_confidence=0.1,
        raw_text=text,
        normalized_text=text,
    )


def _candidates() -> frozenset[str]:
    return get_candidates(ConversationState.IDLE.value)


def _full_response(items: list) -> str:
    return json.dumps({
        "rhv": True,
        "decision": "no_repair",
        "items": items,
    })


# ---------------------------------------------------------------------------
# _parse_items unit tests
# ---------------------------------------------------------------------------


class TestParseItemsUnit:
    def test_none_returns_empty(self):
        assert _parse_items(None) == ()

    def test_non_list_returns_empty(self):
        assert _parse_items("burger") == ()
        assert _parse_items(42) == ()
        assert _parse_items({}) == ()

    def test_empty_list_returns_empty(self):
        assert _parse_items([]) == ()

    def test_single_item_parsed(self):
        raw = [{"item": "burger", "quantity": 2}]
        result = _parse_items(raw)
        assert len(result) == 1
        assert result[0].item == "burger"
        assert result[0].quantity == 2

    def test_quantity_clamped_to_max(self):
        raw = [{"item": "burger", "quantity": 500}]
        result = _parse_items(raw)
        assert result[0].quantity == 99

    def test_quantity_clamped_to_min(self):
        raw = [{"item": "burger", "quantity": -10}]
        result = _parse_items(raw)
        assert result[0].quantity == 1

    def test_quantity_zero_clamped_to_one(self):
        raw = [{"item": "fries", "quantity": 0}]
        result = _parse_items(raw)
        assert result[0].quantity == 1

    def test_quantity_default_one(self):
        raw = [{"item": "coke"}]
        result = _parse_items(raw)
        assert result[0].quantity == 1

    def test_non_integer_quantity_defaults_to_one(self):
        raw = [{"item": "coffee", "quantity": "two"}]
        result = _parse_items(raw)
        assert result[0].quantity == 1

    def test_entry_with_empty_item_dropped(self):
        raw = [{"item": "", "quantity": 1}, {"item": "fries"}]
        result = _parse_items(raw)
        assert len(result) == 1
        assert result[0].item == "fries"

    def test_entry_with_null_item_dropped(self):
        raw = [{"item": None}, {"item": "salad"}]
        result = _parse_items(raw)
        assert len(result) == 1
        assert result[0].item == "salad"

    def test_non_dict_entry_dropped(self):
        raw = ["not a dict", {"item": "wrap"}]
        result = _parse_items(raw)
        assert len(result) == 1
        assert result[0].item == "wrap"

    def test_sides_parsed(self):
        raw = [{"item": "combo", "sides": ["fries", "coleslaw"]}]
        result = _parse_items(raw)
        assert result[0].sides == ("fries", "coleslaw")

    def test_modifiers_parsed(self):
        raw = [{"item": "burger", "modifiers": ["no onions", "extra cheese"]}]
        result = _parse_items(raw)
        assert result[0].modifiers == ("no onions", "extra cheese")

    def test_missing_parsed(self):
        raw = [{"item": "pizza", "missing": ["SIZE"]}]
        result = _parse_items(raw)
        assert result[0].missing == ("SIZE",)

    def test_size_and_variant_parsed(self):
        raw = [{"item": "drink", "size": "large", "variant": "diet"}]
        result = _parse_items(raw)
        assert result[0].size == "large"
        assert result[0].variant == "diet"

    def test_null_size_and_variant_become_none(self):
        raw = [{"item": "water", "size": None, "variant": None}]
        result = _parse_items(raw)
        assert result[0].size is None
        assert result[0].variant is None

    def test_order_preserved(self):
        raw = [{"item": "a"}, {"item": "b"}, {"item": "c"}]
        result = _parse_items(raw)
        assert [r.item for r in result] == ["a", "b", "c"]

    def test_malformed_sides_list_gives_empty(self):
        raw = [{"item": "burger", "sides": "not-a-list"}]
        result = _parse_items(raw)
        assert result[0].sides == ()

    def test_non_string_sides_values_coerced(self):
        raw = [{"item": "meal", "sides": [1, 2, 3]}]
        result = _parse_items(raw)
        # Values are stringified
        assert result[0].sides == ("1", "2", "3")


# ---------------------------------------------------------------------------
# parse_output integration: items in result
# ---------------------------------------------------------------------------


class TestParseOutputWithItems:
    def test_items_parsed_in_full_response(self):
        nlu = _nlu()
        raw = _full_response([
            {"item": "burger", "quantity": 1, "modifiers": ["no pickles"]},
        ])
        result = parse_output(raw=raw, candidates=_candidates(), nlu=nlu, latency_ms=50.0)
        assert len(result.items) == 1
        assert result.items[0].item == "burger"
        assert result.items[0].modifiers == ("no pickles",)

    def test_items_not_applied_to_cart(self):
        """Items are parsed and stored in result, but applied=False always."""
        nlu = _nlu()
        raw = _full_response([{"item": "burger", "quantity": 2}])
        result = parse_output(raw=raw, candidates=_candidates(), nlu=nlu, latency_ms=50.0)
        assert result.applied is False

    def test_empty_items_returns_empty_tuple(self):
        nlu = _nlu()
        raw = _full_response([])
        result = parse_output(raw=raw, candidates=_candidates(), nlu=nlu, latency_ms=50.0)
        assert result.items == ()

    def test_no_items_key_returns_empty_tuple(self):
        nlu = _nlu()
        raw = json.dumps({"rhv": True, "decision": "no_repair"})
        result = parse_output(raw=raw, candidates=_candidates(), nlu=nlu, latency_ms=50.0)
        assert result.items == ()

    def test_malformed_item_entries_dropped(self):
        nlu = _nlu()
        raw = _full_response([
            "not-a-dict",
            {"item": ""},
            {"item": "fries"},
        ])
        result = parse_output(raw=raw, candidates=_candidates(), nlu=nlu, latency_ms=50.0)
        assert len(result.items) == 1
        assert result.items[0].item == "fries"

    def test_items_are_tuple_of_gptrepairitem(self):
        nlu = _nlu()
        raw = _full_response([{"item": "salad"}])
        result = parse_output(raw=raw, candidates=_candidates(), nlu=nlu, latency_ms=50.0)
        assert isinstance(result.items, tuple)
        assert all(isinstance(it, GptRepairItem) for it in result.items)
