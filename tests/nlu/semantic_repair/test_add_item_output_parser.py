# tests/nlu/semantic_repair/test_add_item_output_parser.py
"""Tests for the ADD_ITEM output parser."""
from __future__ import annotations

import json

import pytest

from app.nlu.semantic_repair.add_item_output_parser import parse_add_item_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_json(**kwargs) -> str:
    """Build a minimal valid JSON response the parser accepts."""
    base = {
        "requires_handler_validation": True,
        "decision": "ok",
        "intent": "add_item",
        "items": [],
        "global_slots": [],
        "missing": [],
        "fallback_type": "none",
        "confidence": 0.9,
        "reason": "test",
    }
    base.update(kwargs)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# JSON decode / structure errors
# ---------------------------------------------------------------------------

class TestJsonDecodeErrors:
    def test_empty_string_returns_no_repair(self):
        plan = parse_add_item_output("", utterance_text="burger")
        assert plan.decision == "no_repair"
        assert plan.parse_error is not None
        assert "json_decode" in plan.parse_error

    def test_malformed_json_returns_no_repair(self):
        plan = parse_add_item_output("{not valid json", utterance_text="burger")
        assert plan.decision == "no_repair"

    def test_json_array_not_object_returns_no_repair(self):
        plan = parse_add_item_output("[]", utterance_text="burger")
        assert plan.parse_error == "json_not_object"

    def test_markdown_fence_stripped(self):
        raw = "```json\n" + _valid_json(items=[{"item": "burger", "requires_handler_validation": True}]) + "\n```"
        # Parser should strip fences and succeed
        plan = parse_add_item_output(raw, utterance_text="burger")
        # No parse error expected (rhv check passes)
        assert plan.decision in ("ok", "no_repair")

    def test_requires_handler_validation_missing_returns_no_repair(self):
        data = {"decision": "ok", "intent": "add_item", "items": []}
        plan = parse_add_item_output(json.dumps(data), utterance_text="burger")
        assert plan.parse_error == "requires_handler_validation"

    def test_requires_handler_validation_false_returns_no_repair(self):
        data = {"requires_handler_validation": False, "decision": "ok", "items": []}
        plan = parse_add_item_output(json.dumps(data), utterance_text="burger")
        assert plan.parse_error == "requires_handler_validation"


# ---------------------------------------------------------------------------
# Decision coercion
# ---------------------------------------------------------------------------

class TestDecisionCoercion:
    def test_valid_decision_ok(self):
        plan = parse_add_item_output(_valid_json(decision="ok"), utterance_text="burger")
        assert plan.decision == "ok"

    def test_valid_decision_repair(self):
        plan = parse_add_item_output(_valid_json(decision="repair"), utterance_text="burger")
        assert plan.decision == "repair"

    def test_invalid_decision_coerced_to_no_repair(self):
        plan = parse_add_item_output(_valid_json(decision="invalid_value"), utterance_text="burger")
        assert plan.decision == "no_repair"

    def test_intent_validated(self):
        plan = parse_add_item_output(_valid_json(intent="add_item"), utterance_text="burger")
        assert plan.intent == "add_item"

    def test_invalid_intent_dropped(self):
        plan = parse_add_item_output(_valid_json(intent="checkout"), utterance_text="burger")
        assert plan.intent is None


# ---------------------------------------------------------------------------
# Items parsing
# ---------------------------------------------------------------------------

class TestItemsParsing:
    def test_parses_simple_item(self):
        raw = _valid_json(items=[{"item": "burger", "quantity": 2}])
        plan = parse_add_item_output(raw, utterance_text="two burgers")
        assert len(plan.items) == 1
        assert plan.items[0].item == "burger"
        assert plan.items[0].quantity == 2

    def test_item_not_in_utterance_dropped(self):
        raw = _valid_json(items=[{"item": "pizza"}])
        plan = parse_add_item_output(raw, utterance_text="burger")
        assert len(plan.items) == 0

    def test_item_empty_name_dropped(self):
        raw = _valid_json(items=[{"item": ""}, {"item": "  "}])
        plan = parse_add_item_output(raw, utterance_text="burger")
        assert len(plan.items) == 0

    def test_item_in_cart_allowed_even_if_not_in_utterance(self):
        raw = _valid_json(items=[{"item": "burger"}])
        plan = parse_add_item_output(
            raw,
            utterance_text="extra cheese",
            cart_item_names=("burger",),
        )
        assert len(plan.items) == 1

    def test_non_dict_item_dropped(self):
        raw = _valid_json(items=["burger", None, 42])
        plan = parse_add_item_output(raw, utterance_text="burger")
        assert len(plan.items) == 0

    def test_items_capped_to_max_items(self):
        items = [{"item": f"item{i}"} for i in range(5)]
        utterance = " ".join(f"item{i}" for i in range(5))
        raw = _valid_json(items=items)
        plan = parse_add_item_output(raw, utterance_text=utterance, max_items=3)
        assert len(plan.items) == 3
        assert any("items_truncated:2" in n for n in plan.parse_notes)

    def test_size_validated_against_utterance(self):
        raw = _valid_json(items=[{"item": "pizza", "size": "large"}])
        plan = parse_add_item_output(raw, utterance_text="large pizza")
        assert plan.items[0].size == "large"

    def test_size_hallucinated_dropped(self):
        raw = _valid_json(items=[{"item": "pizza", "size": "extra-large"}])
        plan = parse_add_item_output(raw, utterance_text="pizza")
        assert plan.items[0].size is None
        assert any("item_size_dropped" in n for n in plan.items[0].parse_notes)

    def test_size_valid_via_choices(self):
        raw = _valid_json(items=[{"item": "pizza", "size": "medium"}])
        plan = parse_add_item_output(
            raw,
            utterance_text="pizza",
            choices=("small", "medium", "large"),
        )
        assert plan.items[0].size == "medium"

    def test_invalid_quantity_becomes_none(self):
        raw = _valid_json(items=[{"item": "burger", "quantity": 0}])
        plan = parse_add_item_output(raw, utterance_text="burger")
        assert plan.items[0].quantity is None

    def test_negative_quantity_becomes_none(self):
        raw = _valid_json(items=[{"item": "burger", "quantity": -1}])
        plan = parse_add_item_output(raw, utterance_text="burger")
        assert plan.items[0].quantity is None


# ---------------------------------------------------------------------------
# Sides and modifiers
# ---------------------------------------------------------------------------

class TestSidesAndModifiers:
    def test_side_in_utterance_kept(self):
        raw = _valid_json(items=[{
            "item": "burger",
            "sides": [{"name": "fries", "operation": "add"}],
        }])
        plan = parse_add_item_output(raw, utterance_text="burger with fries")
        assert len(plan.items[0].sides) == 1
        assert plan.items[0].sides[0].name == "fries"

    def test_side_hallucinated_dropped(self):
        raw = _valid_json(items=[{
            "item": "burger",
            "sides": [{"name": "onion rings"}],
        }])
        plan = parse_add_item_output(raw, utterance_text="burger")
        assert len(plan.items[0].sides) == 0

    def test_modifier_in_utterance_kept(self):
        raw = _valid_json(items=[{
            "item": "burger",
            "modifiers": [{"name": "cheese", "operation": "add"}],
        }])
        plan = parse_add_item_output(raw, utterance_text="burger with cheese")
        assert len(plan.items[0].modifiers) == 1

    def test_modifier_size_stripped_with_note(self):
        raw = _valid_json(items=[{
            "item": "burger",
            "modifiers": [{"name": "cheese", "size": "large"}],
        }])
        plan = parse_add_item_output(raw, utterance_text="burger with cheese")
        assert plan.items[0].modifiers[0].size is None
        assert "modifier_size_dropped" in plan.items[0].modifiers[0].parse_notes

    def test_modifier_variant_stripped_with_note(self):
        raw = _valid_json(items=[{
            "item": "burger",
            "modifiers": [{"name": "cheese", "variant": "american"}],
        }])
        plan = parse_add_item_output(raw, utterance_text="burger with cheese")
        assert plan.items[0].modifiers[0].variant is None
        assert "modifier_size_dropped" in plan.items[0].modifiers[0].parse_notes


# ---------------------------------------------------------------------------
# Global slots and missing
# ---------------------------------------------------------------------------

class TestGlobalSlots:
    def test_valid_global_slot_kept(self):
        raw = _valid_json(global_slots=[{"n": "SIZE", "v": "large"}])
        plan = parse_add_item_output(raw, utterance_text="large pizza")
        assert len(plan.global_slots) == 1

    def test_invalid_global_slot_name_dropped(self):
        raw = _valid_json(global_slots=[{"n": "PAYMENT_TYPE", "v": "cash"}])
        plan = parse_add_item_output(raw, utterance_text="large pizza")
        assert len(plan.global_slots) == 0

    def test_missing_slots_captured(self):
        raw = _valid_json(missing=["SIZE", "QUANTITY"])
        plan = parse_add_item_output(raw, utterance_text="burger")
        assert "SIZE" in plan.missing
        assert "QUANTITY" in plan.missing

    def test_missing_slots_uppercased(self):
        raw = _valid_json(missing=["size", "modifier"])
        plan = parse_add_item_output(raw, utterance_text="burger")
        assert "SIZE" in plan.missing


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------

class TestMetadataFields:
    def test_confidence_parsed(self):
        plan = parse_add_item_output(_valid_json(confidence=0.87), utterance_text="burger")
        assert plan.confidence == pytest.approx(0.87)

    def test_invalid_confidence_becomes_none(self):
        plan = parse_add_item_output(_valid_json(confidence="high"), utterance_text="burger")
        assert plan.confidence is None

    def test_reason_capped_at_200_chars(self):
        long_reason = "x" * 300
        plan = parse_add_item_output(_valid_json(reason=long_reason), utterance_text="burger")
        assert plan.reason is not None
        assert len(plan.reason) <= 200

    def test_latency_ms_propagated(self):
        plan = parse_add_item_output(_valid_json(), utterance_text="burger", latency_ms=123.4)
        assert plan.latency_ms == pytest.approx(123.4)
        assert plan.total_ms == pytest.approx(123.4)

    def test_eligible_always_true_when_parsed(self):
        plan = parse_add_item_output(_valid_json(), utterance_text="burger")
        assert plan.eligible is True

    def test_never_raises_on_garbage_input(self):
        # Should return a safe plan, not raise
        for bad in [None, "", "garbage", "[]", "{}", '{"rhv":false}']:
            try:
                plan = parse_add_item_output(bad or "", utterance_text="test")
                assert plan.decision == "no_repair"
            except Exception as exc:
                pytest.fail(f"Parser raised on input {bad!r}: {exc}")
