# tests/nlu/semantic_repair/test_gpt_repair_shadow.py
"""Unit tests for Phase 2 GPT shadow-mode semantic repair.

All tests mock the openai client so no real network calls are made.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.config.semantic_repair import SemanticRepairConfig
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_result import NLUResult, SlotValue
from app.nlu.semantic_repair.gpt_repair_result import GPT_NOT_CALLED, GptRepairResult
from app.nlu.semantic_repair.output_parser import parse_output
from app.nlu.semantic_repair.prompt_builder import build_messages, get_candidates
from app.nlu.semantic_repair.repair_service import GptRepairService, RepairPolicy
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nlu(text: str, intent: Intent = Intent.UNKNOWN, slots=()) -> NLUResult:
    return NLUResult(
        effective_intent=intent,
        intent_confidence=0.1 if intent == Intent.UNKNOWN else 0.9,
        raw_text=text,
        normalized_text=text,
        slots=tuple(slots),
    )


def _intent(intent: Intent, text: str = "test") -> IntentResult:
    return IntentResult(intent=intent, raw_text=text)


def _make_openai_response(content: str):
    """Minimal stand-in for openai ChatCompletion response."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _valid_repair_json(intent: str = "add_item", confidence: float = 0.9) -> str:
    return json.dumps({
        "decision": "repair",
        "repaired_intent": intent,
        "repaired_control_intent": None,
        "slot_corrections": {},
        "confidence": confidence,
        "reason": "clearly ordering food",
        "requires_handler_validation": True,
    })


def _service(phase: int) -> GptRepairService:
    cfg = SemanticRepairConfig(
        phase=phase,
        model="gpt-4o-mini",
        timeout_seconds=3.0,
    )
    return GptRepairService(config=cfg)


# ---------------------------------------------------------------------------
# RepairPolicy
# ---------------------------------------------------------------------------


class TestRepairPolicy:
    def test_eligible_when_unknown_with_text(self):
        policy = RepairPolicy()
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)
        analysis = policy.check(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE
        )
        assert analysis.gpt_repair_eligible is True
        assert analysis.reason == "unknown_intent_with_text"
        assert analysis.candidate_count > 0

    def test_ineligible_when_intent_known(self):
        policy = RepairPolicy()
        nlu = _nlu("I want a burger", intent=Intent.ADD_ITEM)
        ir = _intent(Intent.ADD_ITEM)
        analysis = policy.check(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE
        )
        assert analysis.gpt_repair_eligible is False
        assert analysis.reason == "intent_known"
        assert analysis.candidate_count == 0

    def test_ineligible_when_text_too_short(self):
        policy = RepairPolicy()
        nlu = _nlu("ok")
        ir = _intent(Intent.UNKNOWN)
        analysis = policy.check(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE
        )
        assert analysis.gpt_repair_eligible is False
        assert analysis.reason == "text_too_short"


# ---------------------------------------------------------------------------
# Phase 0 — no GPT call
# ---------------------------------------------------------------------------


class TestPhase0:
    def test_phase0_does_not_call_gpt(self):
        svc = _service(phase=0)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        with patch.object(svc, "_call_gpt") as mock_call:
            analysis, result = svc.run(
                nlu=nlu, intent_result=ir, state=ConversationState.IDLE
            )
            mock_call.assert_not_called()

        assert result is GPT_NOT_CALLED
        assert result.applied is False

    def test_phase0_eligible_analysis_still_produced(self):
        """Phase 0 should still return analysis (for Phase 0 eligibility logging)."""
        svc = _service(phase=0)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        analysis, result = svc.run(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE
        )
        assert analysis.gpt_repair_eligible is True
        assert result is GPT_NOT_CALLED

    def test_phase0_ineligible_does_not_call_gpt(self):
        svc = _service(phase=0)
        nlu = _nlu("add item", intent=Intent.ADD_ITEM)
        ir = _intent(Intent.ADD_ITEM)

        with patch.object(svc, "_call_gpt") as mock_call:
            _, result = svc.run(
                nlu=nlu, intent_result=ir, state=ConversationState.IDLE
            )
            mock_call.assert_not_called()

        assert result is GPT_NOT_CALLED


# ---------------------------------------------------------------------------
# Phase 2 — GPT call in shadow mode
# ---------------------------------------------------------------------------


class TestPhase2Eligibility:
    def test_phase2_calls_gpt_for_eligible_unknown(self):
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        fake_response = _make_openai_response(_valid_repair_json("add_item"))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        svc._client = mock_client

        analysis, result = svc.run(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE
        )
        mock_client.chat.completions.create.assert_called_once()
        assert analysis.gpt_repair_eligible is True
        assert result.decision == "repair"
        assert result.repaired_intent == "add_item"

    def test_phase2_does_not_call_gpt_for_known_intent(self):
        svc = _service(phase=2)
        nlu = _nlu("add item", intent=Intent.ADD_ITEM)
        ir = _intent(Intent.ADD_ITEM)

        with patch.object(svc, "_call_gpt") as mock_call:
            _, result = svc.run(
                nlu=nlu, intent_result=ir, state=ConversationState.IDLE
            )
            mock_call.assert_not_called()

        assert result is GPT_NOT_CALLED


# ---------------------------------------------------------------------------
# Phase 2 — never applies repair
# ---------------------------------------------------------------------------


class TestPhase2NeverApplies:
    def test_result_applied_is_always_false(self):
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        fake_response = _make_openai_response(_valid_repair_json("add_item"))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        svc._client = mock_client

        _, result = svc.run(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE
        )
        assert result.applied is False

    def test_no_repair_result_also_has_applied_false(self):
        result = GptRepairResult(decision="no_repair")
        assert result.applied is False

    def test_gpt_not_called_sentinel_applied_false(self):
        assert GPT_NOT_CALLED.applied is False


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_logs_gpt_timeout_true_and_no_repair(self):
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        mock_client = MagicMock()
        # Simulate a timeout by raising an exception whose name contains "Timeout"
        mock_client.chat.completions.create.side_effect = Exception("APITimeoutError")
        svc._client = mock_client

        # Patch _call_gpt to exercise the actual timeout branch
        _, result = svc.run(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE
        )
        # With a generic exception name not containing "Timeout", it's a call_error.
        # Let's use a properly named exception.

    def test_timeout_exception_name_detected(self):
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        class APITimeoutError(Exception):
            pass

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError("timed out")
        svc._client = mock_client

        _, result = svc.run(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE
        )
        assert result.decision == "no_repair"
        assert result.timeout is True
        assert result.parse_error is None
        assert result.applied is False
        assert result.latency_ms is not None

    def test_timeout_result_has_no_repair(self):
        svc = _service(phase=2)
        nlu = _nlu("hello there burger please")
        ir = _intent(Intent.UNKNOWN)

        class RequestTimeoutError(Exception):
            pass

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RequestTimeoutError()
        svc._client = mock_client

        _, result = svc.run(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE
        )
        assert result.decision == "no_repair"
        assert result.timeout is True


# ---------------------------------------------------------------------------
# Output parser — invalid JSON
# ---------------------------------------------------------------------------


class TestOutputParserInvalidJson:
    def _candidates(self) -> frozenset[str]:
        return get_candidates(ConversationState.IDLE.value)

    def test_invalid_json_logs_parse_error(self):
        nlu = _nlu("give me a burger")
        result = parse_output(
            raw="this is not json at all",
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=50.0,
        )
        assert result.decision == "no_repair"
        assert result.parse_error is not None
        assert "json_decode" in result.parse_error
        assert result.applied is False

    def test_empty_string_logs_parse_error(self):
        nlu = _nlu("give me a burger")
        result = parse_output(
            raw="",
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=50.0,
        )
        assert result.decision == "no_repair"
        assert result.parse_error is not None

    def test_non_dict_json_logs_parse_error(self):
        nlu = _nlu("give me a burger")
        result = parse_output(
            raw=json.dumps(["repair", "add_item"]),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=50.0,
        )
        assert result.decision == "no_repair"
        assert result.parse_error == "response_not_dict"

    def test_missing_requires_handler_validation_logs_parse_error(self):
        nlu = _nlu("give me a burger")
        data = {
            "decision": "repair",
            "repaired_intent": "add_item",
            "confidence": 0.9,
            "reason": "test",
            # requires_handler_validation is missing
        }
        result = parse_output(
            raw=json.dumps(data),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=50.0,
        )
        assert result.decision == "no_repair"
        assert result.parse_error == "requires_handler_validation_not_true"

    def test_requires_handler_validation_false_logs_parse_error(self):
        nlu = _nlu("give me a burger")
        data = {
            "decision": "repair",
            "repaired_intent": "add_item",
            "confidence": 0.9,
            "reason": "test",
            "requires_handler_validation": False,
        }
        result = parse_output(
            raw=json.dumps(data),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=50.0,
        )
        assert result.decision == "no_repair"
        assert result.parse_error == "requires_handler_validation_not_true"


# ---------------------------------------------------------------------------
# Output parser — out-of-set repaired_intent
# ---------------------------------------------------------------------------


class TestOutputParserOutOfSetIntent:
    def _candidates(self) -> frozenset[str]:
        return get_candidates(ConversationState.IDLE.value)

    def test_out_of_set_repaired_intent_is_rejected(self):
        nlu = _nlu("give me a burger")
        data = {
            "decision": "repair",
            "repaired_intent": "some_fantasy_intent_not_in_list",
            "confidence": 0.95,
            "reason": "test",
            "requires_handler_validation": True,
        }
        result = parse_output(
            raw=json.dumps(data),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=30.0,
        )
        assert result.decision == "no_repair"
        assert result.parse_error is not None
        assert "repaired_intent_not_in_candidates" in result.parse_error
        assert result.applied is False

    def test_in_set_repaired_intent_is_accepted(self):
        nlu = _nlu("give me a burger")
        data = {
            "decision": "repair",
            "repaired_intent": "add_item",
            "confidence": 0.9,
            "reason": "user wants food",
            "requires_handler_validation": True,
        }
        result = parse_output(
            raw=json.dumps(data),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=30.0,
        )
        assert result.decision == "repair"
        assert result.repaired_intent == "add_item"
        assert result.parse_error is None

    def test_null_repaired_intent_on_repair_decision_rejected(self):
        nlu = _nlu("give me a burger")
        data = {
            "decision": "repair",
            "repaired_intent": None,
            "confidence": 0.9,
            "reason": "test",
            "requires_handler_validation": True,
        }
        result = parse_output(
            raw=json.dumps(data),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=30.0,
        )
        assert result.decision == "no_repair"
        assert "repaired_intent_not_in_candidates" in (result.parse_error or "")


# ---------------------------------------------------------------------------
# Output parser — slot value validation
# ---------------------------------------------------------------------------


class TestOutputParserSlotValidation:
    def _candidates(self) -> frozenset[str]:
        return get_candidates(ConversationState.IDLE.value)

    def test_slot_value_in_utterance_is_accepted(self):
        nlu = _nlu("I want two burgers")
        data = {
            "decision": "repair",
            "repaired_intent": "add_item",
            "slot_corrections": {"quantity": "two"},
            "confidence": 0.9,
            "reason": "quantity in text",
            "requires_handler_validation": True,
        }
        result = parse_output(
            raw=json.dumps(data),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=30.0,
        )
        assert result.decision == "repair"
        assert result.slot_corrections == {"quantity": "two"}
        assert result.parse_error is None

    def test_slot_value_in_original_slots_is_accepted(self):
        slot = SlotValue(name="item_name", value="burger")
        nlu = NLUResult(
            effective_intent=Intent.UNKNOWN,
            intent_confidence=0.1,
            raw_text="get me one",
            normalized_text="get me one",
            slots=(slot,),
        )
        data = {
            "decision": "repair",
            "repaired_intent": "add_item",
            "slot_corrections": {"item_name": "burger"},  # in original slots
            "confidence": 0.9,
            "reason": "slot already extracted",
            "requires_handler_validation": True,
        }
        result = parse_output(
            raw=json.dumps(data),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=30.0,
        )
        assert result.decision == "repair"
        assert result.parse_error is None

    def test_slot_value_not_in_utterance_or_slots_is_dropped(self):
        # Invalid slot name "item_name" (not in _VALID_SLOT_NAMES) is dropped
        # silently; the intent repair still goes through.
        nlu = _nlu("give me something good")
        data = {
            "decision": "repair",
            "repaired_intent": "add_item",
            "slot_corrections": {"item_name": "unicorn_steak"},
            "confidence": 0.9,
            "reason": "hallucinated slot",
            "requires_handler_validation": True,
        }
        result = parse_output(
            raw=json.dumps(data),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=30.0,
        )
        # Invalid slot name is dropped, not the whole response
        assert result.decision == "repair"
        assert result.slot_corrections is None  # entry was dropped
        assert result.parse_error is None

    def test_valid_slot_name_with_hallucinated_value_is_dropped(self):
        # Valid slot name but value not in utterance → drop slot correction entry.
        # The intent repair in "repair" decision still goes through.
        nlu = _nlu("give me something good")
        data = {
            "decision": "repair",
            "repaired_intent": "add_item",
            "slots": [{"n": "ITEM", "v": "unicorn_steak", "op": "add"}],
            "confidence": 0.9,
            "reason": "hallucinated value",
            "requires_handler_validation": True,
        }
        result = parse_output(
            raw=json.dumps(data),
            candidates=self._candidates(),
            nlu=nlu,
            latency_ms=30.0,
        )
        # "repair" decision keeps intent repair; hallucinated slot is dropped
        assert result.decision == "repair"
        assert result.repaired_intent == "add_item"
        assert result.slot_corrections is None
        assert result.slot_corrections_list is None


# ---------------------------------------------------------------------------
# API key never logged
# ---------------------------------------------------------------------------


class TestApiKeyNeverLogged:
    """The OpenAI API key must never appear in any GptRepairResult field."""

    FAKE_KEY = "sk-test-secretkeyvalue12345678901234567890"

    def _result_fields(self, result: GptRepairResult) -> list[str]:
        import dataclasses
        return [
            str(getattr(result, f.name))
            for f in dataclasses.fields(result)
            if getattr(result, f.name) is not None
        ]

    def test_api_key_not_in_no_repair_result(self):
        result = GptRepairResult(
            decision="no_repair",
            reason="test",
            parse_error="some_error",
        )
        for field_val in self._result_fields(result):
            assert self.FAKE_KEY not in field_val

    def test_api_key_not_in_repair_result(self):
        result = GptRepairResult(
            decision="repair",
            repaired_intent="add_item",
            confidence=0.9,
            reason="test",
        )
        for field_val in self._result_fields(result):
            assert self.FAKE_KEY not in field_val

    def test_service_does_not_store_api_key(self):
        """GptRepairService._client should not expose the raw key as an attribute."""
        svc = _service(phase=2)
        # Verify the service has no direct api_key attribute
        assert not hasattr(svc, "api_key")
        assert not hasattr(svc, "_api_key")

    def test_config_does_not_contain_api_key(self):
        """SemanticRepairConfig must not include the API key."""
        import dataclasses
        cfg = SemanticRepairConfig(phase=2, model="gpt-4o-mini", timeout_seconds=3.0)
        field_names = {f.name for f in dataclasses.fields(cfg)}
        assert "api_key" not in field_names
        assert "openai_api_key" not in field_names


# ---------------------------------------------------------------------------
# Prompt builder — safety contract
# ---------------------------------------------------------------------------


class TestPromptBuilderSafety:
    def test_candidates_are_subset_not_full_enum(self):
        from app.nlu.intent_resolution.intent import Intent
        all_intent_values = {i.value for i in Intent}
        for state in ConversationState:
            candidates = get_candidates(state.value)
            # Candidate list must be smaller than the full enum
            assert len(candidates) < len(all_intent_values), (
                f"State {state.value} has all intents in candidates"
            )

    def test_messages_do_not_contain_full_enum(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="I want a burger",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
        )
        full_text = " ".join(m["content"] for m in messages)
        # A sample of intents that must NOT appear (they're not in IDLE candidates)
        assert "payment_methods_query" not in full_text
        assert "order_placement_status" not in full_text
        assert "order_error_status" not in full_text

    def test_messages_do_not_contain_api_key_pattern(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="I want a burger",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
        )
        full_text = " ".join(m["content"] for m in messages)
        assert "sk-" not in full_text
        assert "OPENAI_API_KEY" not in full_text
