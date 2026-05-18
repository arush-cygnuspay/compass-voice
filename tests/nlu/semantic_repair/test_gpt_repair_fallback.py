# tests/nlu/semantic_repair/test_gpt_repair_fallback.py
"""Tests for GPT fallback classification: output parser, logging, and templates."""
from __future__ import annotations

import json

import pytest

from app.logging.gpt_repair_csv_logger import HEADERS
from app.nlu.nlu_result import NLUResult
from app.nlu.semantic_repair.gpt_log_record_builder import (
    build_gpt_repair_csv_row,
    build_gpt_repair_log_record,
)
from app.nlu.semantic_repair.gpt_repair_result import GptRepairResult
from app.nlu.semantic_repair.output_parser import parse_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FORBIDDEN_INTERNAL_WORDS = {
    "intent", "slot", "gpt", "model", "fallback",
    "state", "handler", "modifier",
}


def _nlu(text: str = "hello") -> NLUResult:
    return NLUResult(
        effective_intent=None,
        intent_confidence=0.0,
        raw_text=text,
        normalized_text=text,
    )


def _parse(data: dict, *, text: str = "hello") -> GptRepairResult:
    raw = json.dumps(data)
    return parse_output(
        raw=raw,
        candidates=frozenset({"add_item", "checkout", "unknown"}),
        nlu=_nlu(text),
        latency_ms=10.0,
    )


def _make_event(**overrides):
    from app.diagnostics.turn_event import TurnEvent
    defaults = dict(
        session_id="s1",
        turn_index=0,
        state_before="idle",
        state_after="idle",
        next_state="idle",
        pending_action="",
        current_prompt_field="",
        current_item_id="",
        current_item_name="",
        raw_user_text="",
        user_text="what are your hours",
        normalized_text="what are your hours",
        pred_main_intent="unknown",
        pred_sub_intent="",
        pred_intent="unknown",
        pred_intent_confidence=0.4,
        slot_model_ran=False,
        slots=(),
        response_key="intent_not_allowed",
        response_text="",
        command=None,
        normalized_values={},
        missing_required_fields=(),
        reprompt_field="",
        reprompt_count=0,
        reprompt_escalated=False,
        reprompt_escalation_count=0,
        fallback_triggered=False,
        fallback_reason="",
        fallback_count=0,
        slot_extraction_failed=False,
        slot_extraction_failure_count=0,
        invalid_modifier=False,
        invalid_modifier_count=0,
        user_repeated=False,
        repeated_user_turn_count=0,
    )
    defaults.update(overrides)
    return TurnEvent(**defaults)


# ---------------------------------------------------------------------------
# output_parser: fallback decision parsing
# ---------------------------------------------------------------------------


class TestOutputParserFallback:
    def test_off_topic_utterance_produces_fallback(self):
        result = _parse({
            "decision": "fallback",
            "fallback_type": "off_topic",
            "confidence": 0.85,
            "reason": "completely unrelated to food ordering",
            "requires_handler_validation": True,
        })
        assert result.decision == "fallback"
        assert result.fallback_type == "off_topic"

    def test_user_frustrated_produces_fallback(self):
        result = _parse({
            "decision": "fallback",
            "fallback_type": "user_frustrated",
            "confidence": 0.9,
            "reason": "user expressed frustration",
            "requires_handler_validation": True,
        })
        assert result.decision == "fallback"
        assert result.fallback_type == "user_frustrated"

    def test_request_human_produces_fallback(self):
        result = _parse({
            "decision": "fallback",
            "fallback_type": "request_human",
            "confidence": 0.95,
            "reason": "user wants to speak to a person",
            "requires_handler_validation": True,
        })
        assert result.decision == "fallback"
        assert result.fallback_type == "request_human"

    def test_unsupported_request_produces_fallback(self):
        result = _parse({
            "decision": "fallback",
            "fallback_type": "unsupported_request",
            "confidence": 0.8,
            "reason": "unsupported request type",
            "requires_handler_validation": True,
        })
        assert result.decision == "fallback"
        assert result.fallback_type == "unsupported_request"

    def test_all_valid_fallback_types_accepted(self):
        valid_types = [
            "off_topic", "restaurant_question", "user_frustrated",
            "request_human", "unclear", "unsupported_request", "back_to_order",
        ]
        for ft in valid_types:
            result = _parse({
                "decision": "fallback",
                "fallback_type": ft,
                "confidence": 0.8,
                "reason": "test",
                "requires_handler_validation": True,
            })
            assert result.decision == "fallback", f"Expected fallback for type={ft}"
            assert result.fallback_type == ft

    def test_fallback_decision_with_invalid_type_returns_no_repair(self):
        result = _parse({
            "decision": "fallback",
            "fallback_type": "bad_type",
            "requires_handler_validation": True,
        })
        assert result.decision == "no_repair"
        assert result.parse_error is not None
        assert "fallback_type_invalid" in result.parse_error

    def test_fallback_decision_with_none_type_returns_no_repair(self):
        result = _parse({
            "decision": "fallback",
            "fallback_type": "none",
            "requires_handler_validation": True,
        })
        assert result.decision == "no_repair"
        assert "fallback_type_invalid" in (result.parse_error or "")

    def test_non_fallback_decision_has_fallback_type_none(self):
        result = _parse({
            "decision": "no_repair",
            "fallback_type": "off_topic",
            "requires_handler_validation": True,
        })
        assert result.decision == "no_repair"
        assert result.fallback_type == "none"

    def test_fallback_result_has_no_repaired_intent(self):
        result = _parse({
            "decision": "fallback",
            "fallback_type": "unclear",
            "selected_intent": "add_item",
            "requires_handler_validation": True,
        })
        assert result.decision == "fallback"
        assert result.repaired_intent is None

    def test_fallback_never_includes_customer_facing_text(self):
        import dataclasses
        result = _parse({
            "decision": "fallback",
            "fallback_type": "off_topic",
            "customer_response": "I can help you order food.",
            "requires_handler_validation": True,
        })
        # GPT output parser must not copy any customer_response field into the result
        for f in dataclasses.fields(result):
            val = getattr(result, f.name)
            if isinstance(val, str):
                assert "I can help you order food" not in val


# ---------------------------------------------------------------------------
# CSV row builder: fallback fields present
# ---------------------------------------------------------------------------


class TestCsvRowFallbackFields:
    def test_gpt_fallback_type_in_csv_row(self):
        event = _make_event(gpt_fallback_type="off_topic", fallback_response_key="fallback_off_topic")
        row = build_gpt_repair_csv_row(event)
        assert "gpt_fallback_type" in row
        assert row["gpt_fallback_type"] == "off_topic"

    def test_fallback_response_key_in_csv_row(self):
        event = _make_event(gpt_fallback_type="user_frustrated", fallback_response_key="fallback_user_frustrated")
        row = build_gpt_repair_csv_row(event)
        assert "fallback_response_key" in row
        assert row["fallback_response_key"] == "fallback_user_frustrated"

    def test_default_event_has_none_fallback_type(self):
        event = _make_event()
        row = build_gpt_repair_csv_row(event)
        assert row["gpt_fallback_type"] == "none"
        assert row["fallback_response_key"] is None

    def test_csv_row_keys_match_headers(self):
        event = _make_event(gpt_fallback_type="off_topic", fallback_response_key="fallback_off_topic")
        row = build_gpt_repair_csv_row(event)
        assert set(row.keys()) == set(HEADERS)

    def test_user_text_in_csv_row(self):
        event = _make_event(user_text="what are your hours", normalized_text="what are your hours")
        row = build_gpt_repair_csv_row(event)
        assert row["user_text"] == "what are your hours"
        assert row["normalized_text"] == "what are your hours"


# ---------------------------------------------------------------------------
# JSONL record builder: fallback fields present
# ---------------------------------------------------------------------------


class TestJsonlRecordFallbackFields:
    def test_fallback_type_in_gpt_block(self):
        event = _make_event(gpt_fallback_type="restaurant_question")
        record = build_gpt_repair_log_record(event)
        assert record["gpt"]["fallback_type"] == "restaurant_question"

    def test_fallback_response_key_in_final_block(self):
        event = _make_event(fallback_response_key="fallback_restaurant_question")
        record = build_gpt_repair_log_record(event)
        assert record["final"]["fallback_response_key"] == "fallback_restaurant_question"


# ---------------------------------------------------------------------------
# Phase 2 invariant: gpt_applied always False
# ---------------------------------------------------------------------------


class TestFallbackApplication:
    def test_fallback_event_gpt_applied_true_when_applied(self):
        # When fallback is actually applied, gpt_applied should be True
        event = _make_event(
            gpt_fallback_type="off_topic",
            fallback_response_key="fallback_off_topic",
            gpt_called=True,
            gpt_decision="fallback",
            gpt_applied=True,
        )
        assert event.gpt_applied is True

    def test_fallback_event_gpt_applied_false_when_not_applied(self):
        # Shadow-mode (GPT called but result not applied) keeps gpt_applied=False
        event = _make_event(
            gpt_called=True,
            gpt_decision="no_repair",
            gpt_applied=False,
        )
        assert event.gpt_applied is False

    def test_fallback_response_key_matches_fallback_type(self):
        event = _make_event(
            gpt_fallback_type="unclear",
            fallback_response_key="fallback_unclear",
        )
        assert event.fallback_response_key == f"fallback_{event.gpt_fallback_type}"


# ---------------------------------------------------------------------------
# Response templates: voice-friendly, no forbidden words, under 110 chars
# ---------------------------------------------------------------------------


class TestFallbackResponseTemplates:
    @pytest.fixture
    def registry(self):
        from unittest.mock import MagicMock
        from app.core.response_builder import ResponseBuilder
        mock_repo = MagicMock()
        return ResponseBuilder(mock_repo)._registry

    @pytest.mark.parametrize("key", [
        "fallback_off_topic",
        "fallback_restaurant_question",
        "fallback_user_frustrated",
        "fallback_request_human",
        "fallback_unclear",
        "fallback_unsupported_request",
        "fallback_back_to_order",
    ])
    def test_template_exists_in_registry(self, registry, key):
        assert key in registry, f"Missing key: {key}"

    @pytest.mark.parametrize("key", [
        "fallback_off_topic",
        "fallback_restaurant_question",
        "fallback_user_frustrated",
        "fallback_request_human",
        "fallback_unclear",
        "fallback_unsupported_request",
        "fallback_back_to_order",
    ])
    def test_template_under_110_chars(self, registry, key):
        fn = registry[key]
        text = fn(None, None, {})
        assert len(text) <= 110, f"{key!r} is {len(text)} chars: {text!r}"

    @pytest.mark.parametrize("key", [
        "fallback_off_topic",
        "fallback_restaurant_question",
        "fallback_user_frustrated",
        "fallback_request_human",
        "fallback_unclear",
        "fallback_unsupported_request",
        "fallback_back_to_order",
    ])
    def test_template_has_no_forbidden_internal_words(self, registry, key):
        fn = registry[key]
        text = fn(None, None, {}).lower()
        for word in _FORBIDDEN_INTERNAL_WORDS:
            assert word not in text, f"{key!r} contains forbidden word {word!r}: {text!r}"
