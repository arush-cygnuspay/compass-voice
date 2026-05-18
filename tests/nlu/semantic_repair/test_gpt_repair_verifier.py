# tests/nlu/semantic_repair/test_gpt_repair_verifier.py
"""Tests for Phase 2 GPT intent/slot verifier — extended spec.

Covers:
  - Top-K intent candidates sorted by confidence
  - Extended LocalTurnAnalysis fields
  - Compact JSON payload exclusions
  - New decision values and slot_corrections_list
  - Richer eligibility rules (confidence gap, waiting state exit, fallback count)
  - High-confidence skip
  - Timing fields logged
  - API key never in any logged field
  - New SemanticRepairConfig fields
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.config.semantic_repair import SemanticRepairConfig
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_result import IntentCandidate, NLUResult, SlotValue
from app.nlu.semantic_repair.gpt_repair_result import GPT_NOT_CALLED, GptRepairResult, SlotCorrection
from app.nlu.semantic_repair.output_parser import parse_output
from app.nlu.semantic_repair.prompt_builder import build_messages, get_candidates
from app.nlu.semantic_repair.repair_service import GptRepairService, LocalTurnAnalysis, RepairPolicy
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nlu(
    text: str,
    intent: Intent = Intent.UNKNOWN,
    slots: tuple = (),
    confidence: float | None = None,
    candidates: tuple[IntentCandidate, ...] = (),
) -> NLUResult:
    conf = confidence if confidence is not None else (0.1 if intent == Intent.UNKNOWN else 0.9)
    return NLUResult(
        effective_intent=intent,
        intent_confidence=conf,
        raw_text=text,
        normalized_text=text,
        slots=tuple(slots),
        intent_candidates=candidates,
    )


def _intent(intent: Intent, text: str = "test") -> IntentResult:
    return IntentResult(intent=intent, raw_text=text)


def _candidates(top: str = "add_item", second: str = "checkout", gap: float = 0.10) -> tuple[IntentCandidate, ...]:
    """Two top-K candidates with a configurable confidence gap."""
    top_conf = 0.80
    second_conf = top_conf - gap
    return (
        IntentCandidate(intent_main="ordering", intent_sub_intent=top, canonical_intent=top, confidence=top_conf),
        IntentCandidate(intent_main="ordering", intent_sub_intent=second, canonical_intent=second, confidence=second_conf),
    )


def _make_response(payload: dict) -> SimpleNamespace:
    msg = SimpleNamespace(content=json.dumps(payload))
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _service(phase: int = 2) -> GptRepairService:
    cfg = SemanticRepairConfig(phase=phase, model="gpt-4o-mini", timeout_seconds=1.0)
    return GptRepairService(config=cfg)


# ---------------------------------------------------------------------------
# SemanticRepairConfig — new fields
# ---------------------------------------------------------------------------


class TestSemanticRepairConfigNewFields:
    def test_default_timeout_ms(self):
        cfg = SemanticRepairConfig(phase=0, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.timeout_ms == 350  # default

    def test_default_top_k_intents(self):
        cfg = SemanticRepairConfig(phase=0, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.top_k_intents == 4

    def test_default_slo_budget_ms(self):
        cfg = SemanticRepairConfig(phase=0, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.slo_budget_ms == 1800

    def test_default_daily_budget(self):
        cfg = SemanticRepairConfig(phase=0, model="gpt-4o-mini", timeout_seconds=0.35)
        assert cfg.daily_budget == 10000


# ---------------------------------------------------------------------------
# IntentCandidate
# ---------------------------------------------------------------------------


class TestIntentCandidate:
    def test_fields_present(self):
        c = IntentCandidate(
            intent_main="ordering",
            intent_sub_intent="add_item",
            canonical_intent="add_item",
            confidence=0.85,
            source="model_sub",
        )
        assert c.intent_main == "ordering"
        assert c.intent_sub_intent == "add_item"
        assert c.canonical_intent == "add_item"
        assert c.confidence == 0.85
        assert c.source == "model_sub"

    def test_default_source(self):
        c = IntentCandidate(
            intent_main="ordering",
            intent_sub_intent="add_item",
            canonical_intent="add_item",
            confidence=0.9,
        )
        assert c.source == "model_sub"

    def test_nlu_result_carries_candidates(self):
        cands = _candidates()
        nlu = _nlu("I want a burger", candidates=cands)
        assert len(nlu.intent_candidates) == 2
        assert nlu.intent_candidates[0].canonical_intent == "add_item"
        assert nlu.intent_candidates[1].confidence < nlu.intent_candidates[0].confidence

    def test_empty_candidates_by_default(self):
        nlu = _nlu("hello")
        assert nlu.intent_candidates == ()


# ---------------------------------------------------------------------------
# LocalTurnAnalysis — extended fields
# ---------------------------------------------------------------------------


class TestLocalTurnAnalysisExtended:
    def test_analysis_carries_intent_candidates(self):
        policy = RepairPolicy()
        cands = _candidates()
        nlu = _nlu("I want a burger", candidates=cands)
        ir = _intent(Intent.UNKNOWN)

        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert len(analysis.intent_candidates) == 2
        assert analysis.intent_candidates[0].canonical_intent == "add_item"

    def test_analysis_carries_normalized_text(self):
        policy = RepairPolicy()
        nlu = _nlu("give me a burger")
        ir = _intent(Intent.UNKNOWN)

        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert analysis.normalized_text == "give me a burger"

    def test_analysis_carries_state_before(self):
        policy = RepairPolicy()
        nlu = _nlu("give me a burger")
        ir = _intent(Intent.UNKNOWN)

        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert analysis.state_before == ConversationState.IDLE.value

    def test_analysis_carries_intent_effective(self):
        policy = RepairPolicy()
        nlu = _nlu("give me a burger", intent=Intent.ADD_ITEM, confidence=0.4)
        ir = _intent(Intent.ADD_ITEM)

        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert analysis.intent_effective == Intent.ADD_ITEM.value

    def test_analysis_carries_slots(self):
        policy = RepairPolicy()
        slot = SlotValue(name="ITEM", value="Burger")
        nlu = _nlu("I want a burger", slots=(slot,))
        ir = _intent(Intent.UNKNOWN)

        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert len(analysis.slots) == 1
        assert analysis.slots[0].name == "ITEM"


# ---------------------------------------------------------------------------
# Richer eligibility rules
# ---------------------------------------------------------------------------


class TestRicherEligibility:
    def test_low_confidence_gap_triggers_eligibility(self):
        """Top1-top2 gap < 0.20 makes a KNOWN intent eligible for GPT check."""
        policy = RepairPolicy()
        # gap = 0.10, below threshold 0.20
        cands = _candidates("add_item", "checkout", gap=0.10)
        nlu = _nlu("I want to pay", intent=Intent.ADD_ITEM, confidence=0.80, candidates=cands)
        ir = _intent(Intent.ADD_ITEM)

        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert analysis.gpt_repair_eligible is True
        assert analysis.reason == "low_confidence_gap"

    def test_large_confidence_gap_does_not_trigger(self):
        """Top1-top2 gap >= 0.20 with known intent → ineligible."""
        policy = RepairPolicy()
        # gap = 0.40, above threshold
        cands = _candidates("add_item", "checkout", gap=0.40)
        nlu = _nlu("add a burger", intent=Intent.ADD_ITEM, confidence=0.90, candidates=cands)
        ir = _intent(Intent.ADD_ITEM)

        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert analysis.gpt_repair_eligible is False

    def test_high_confidence_known_intent_skipped(self):
        """confidence >= 0.85 with known intent → skip even with unknown gap."""
        policy = RepairPolicy()
        nlu = _nlu("add a burger", intent=Intent.ADD_ITEM, confidence=0.95)
        ir = _intent(Intent.ADD_ITEM)

        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert analysis.gpt_repair_eligible is False
        assert analysis.skipped_reason == "high_confidence"

    def test_cancel_phrase_in_waiting_state_eligible(self):
        """User says 'cancel' while in WAITING_FOR_MODIFIER → eligible."""
        policy = RepairPolicy()
        nlu = _nlu("cancel that", intent=Intent.ADD_ITEM, confidence=0.60)
        ir = _intent(Intent.ADD_ITEM)

        analysis = policy.check(
            nlu=nlu, intent_result=ir, state=ConversationState.WAITING_FOR_MODIFIER
        )
        assert analysis.gpt_repair_eligible is True
        assert analysis.reason == "exit_phrase_in_waiting_state"

    def test_done_phrase_in_waiting_state_eligible(self):
        policy = RepairPolicy()
        nlu = _nlu("done with that", intent=Intent.ADD_ITEM, confidence=0.60)
        ir = _intent(Intent.ADD_ITEM)

        analysis = policy.check(
            nlu=nlu, intent_result=ir, state=ConversationState.WAITING_FOR_SIDE
        )
        assert analysis.gpt_repair_eligible is True

    def test_terminal_state_ineligible(self):
        """COMPLETED state → always ineligible."""
        policy = RepairPolicy()
        nlu = _nlu("anything")
        ir = _intent(Intent.UNKNOWN)

        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.COMPLETED)
        assert analysis.gpt_repair_eligible is False
        assert analysis.reason == "terminal_state"

    def test_high_fallback_count_eligible(self):
        """session.fallback_count >= 2 → eligible."""
        policy = RepairPolicy()
        nlu = _nlu("something something", intent=Intent.ADD_ITEM, confidence=0.60)
        ir = _intent(Intent.ADD_ITEM)

        session = SimpleNamespace(
            fallback_count=3,
            last_response_key="",
            cart=SimpleNamespace(is_empty=lambda: True, get_items=lambda: []),
        )

        analysis = policy.check(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE, session=session
        )
        assert analysis.gpt_repair_eligible is True
        assert analysis.reason == "high_fallback_count"

    def test_previous_intent_not_allowed_eligible(self):
        """last_response_key == "intent_not_allowed" → eligible."""
        policy = RepairPolicy()
        nlu = _nlu("show me the thing", intent=Intent.ADD_ITEM, confidence=0.60)
        ir = _intent(Intent.ADD_ITEM)

        session = SimpleNamespace(
            fallback_count=0,
            last_response_key="intent_not_allowed",
            cart=SimpleNamespace(is_empty=lambda: True, get_items=lambda: []),
        )

        analysis = policy.check(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE, session=session
        )
        assert analysis.gpt_repair_eligible is True
        assert analysis.reason == "previous_intent_not_allowed"


# ---------------------------------------------------------------------------
# Compact payload structure
# ---------------------------------------------------------------------------


class TestCompactPayload:
    def _parse_payload(self, messages: list[dict]) -> dict:
        user_content = messages[1]["content"]
        # Extract the JSON part — everything before the output schema block
        json_part = user_content.split("\n\nOutput schema:")[0]
        return json.loads(json_part)

    def test_payload_has_task_field(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="I want a burger",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
        )
        payload = self._parse_payload(messages)
        assert payload["t"] == "verify_extract"

    def test_payload_has_allowed_block(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="I want a burger",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
        )
        payload = self._parse_payload(messages)
        assert "allowed" in payload
        # New compact keys: intents + control (not repair_intents + control_intents)
        assert "intents" in payload["allowed"]
        assert "control" in payload["allowed"]

    def test_payload_local_block_with_candidates(self):
        cands = (
            IntentCandidate(intent_main="ordering", intent_sub_intent="add_item", canonical_intent="add_item", confidence=0.85),
            IntentCandidate(intent_main="ordering", intent_sub_intent="checkout", canonical_intent="checkout", confidence=0.70),
        )
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="I want something",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
            intent_candidates=cands,
            selected_intent="add_item",
            selected_confidence=0.85,
        )
        payload = self._parse_payload(messages)
        local = payload["local"]
        # New compact keys: intent, conf, top_k
        assert local["intent"] == "add_item"
        assert local["conf"] == pytest.approx(0.85, abs=0.001)
        assert len(local["top_k"]) == 2
        assert local["top_k"][0]["intent"] == "add_item"

    def test_payload_does_not_contain_full_menu(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="burger",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
        )
        full_text = " ".join(m["content"] for m in messages)
        assert "menu.json" not in full_text
        assert "Bourbon Chicken" not in full_text

    def test_payload_does_not_contain_api_key_pattern(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="burger",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
        )
        full_text = " ".join(m["content"] for m in messages)
        assert "sk-" not in full_text
        assert "OPENAI_API_KEY" not in full_text

    def test_cart_summary_truncated_to_10_items(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        big_cart = {"count": 15, "items": [f"Item{i}" for i in range(15)]}
        messages = build_messages(
            utterance="add something",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
            cart_summary=big_cart,
        )
        payload = self._parse_payload(messages)
        # New compact key: "cart" (not "cart_summary"), items list capped at 10
        assert len(payload["cart"]["items"]) <= 10

    def test_candidates_are_subset_not_full_enum(self):
        all_intent_values = {i.value for i in Intent}
        for state in ConversationState:
            cands = get_candidates(state.value)
            assert len(cands) < len(all_intent_values)


# ---------------------------------------------------------------------------
# New output schema — decision values and slot_corrections_list
# ---------------------------------------------------------------------------


class TestNewOutputSchema:
    def _candidates(self) -> frozenset[str]:
        return get_candidates(ConversationState.IDLE.value)

    def _nlu(self, text: str = "I want a burger") -> NLUResult:
        return NLUResult(
            effective_intent=Intent.UNKNOWN,
            intent_confidence=0.1,
            raw_text=text,
            normalized_text=text,
        )

    def test_repair_intent_decision_accepted(self):
        nlu = self._nlu("I want a burger")
        data = {
            "decision": "repair_intent",
            "selected_intent": "add_item",
            "slot_corrections": [],
            "confidence": 0.88,
            "reason": "user ordering",
            "requires_handler_validation": True,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=50.0)
        assert result.decision == "repair_intent"
        assert result.repaired_intent == "add_item"
        assert result.parse_error is None

    def test_repair_slots_decision_accepted(self):
        nlu = self._nlu("I want two burgers")
        data = {
            "decision": "repair_slots",
            "slot_corrections": [
                {"slot_name": "QUANTITY", "old_value": None, "new_value": "two", "operation": "add"}
            ],
            "confidence": 0.82,
            "reason": "quantity in text",
            "requires_handler_validation": True,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=50.0)
        assert result.decision == "repair_slots"
        assert result.slot_corrections_list is not None
        assert len(result.slot_corrections_list) == 1
        assert result.slot_corrections_list[0].slot_name == "QUANTITY"
        assert result.slot_corrections_list[0].operation == "add"

    def test_repair_intent_and_slots_decision(self):
        nlu = self._nlu("I want two burgers")
        data = {
            "decision": "repair_intent_and_slots",
            "selected_intent": "add_item",
            "slot_corrections": [
                {"slot_name": "QUANTITY", "old_value": None, "new_value": "two", "operation": "add"}
            ],
            "confidence": 0.90,
            "reason": "combined repair",
            "requires_handler_validation": True,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=50.0)
        assert result.decision == "repair_intent_and_slots"
        assert result.repaired_intent == "add_item"
        assert result.slot_corrections_list is not None

    def test_slot_corrections_list_operation_values(self):
        nlu = self._nlu("give me the burger not fries")
        data = {
            "decision": "repair_slots",
            "slot_corrections": [
                {"slot_name": "ITEM", "old_value": "fries", "new_value": "burger", "operation": "replace"}
            ],
            "confidence": 0.85,
            "reason": "replace item",
            "requires_handler_validation": True,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=50.0)
        assert result.slot_corrections_list is not None
        assert result.slot_corrections_list[0].operation == "replace"
        assert result.slot_corrections_list[0].old_value == "fries"
        assert result.slot_corrections_list[0].new_value == "burger"

    def test_hallucinated_slot_value_is_dropped(self):
        # Valid slot name but value not in utterance → slot entry is dropped silently.
        # repair_slots with no remaining corrections → downgraded to no_repair.
        nlu = self._nlu("I want a burger")
        data = {
            "decision": "repair_slots",
            "slot_corrections": [
                {"slot_name": "ITEM", "old_value": None, "new_value": "unicorn_steak", "operation": "add"}
            ],
            "confidence": 0.90,
            "reason": "test",
            "requires_handler_validation": True,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=30.0)
        assert result.decision == "no_repair"
        assert result.slot_corrections is None
        assert result.slot_corrections_list is None

    def test_selected_intent_key_accepted(self):
        """New 'selected_intent' key is equivalent to old 'repaired_intent'."""
        nlu = self._nlu("I want a burger")
        data = {
            "decision": "repair_intent",
            "selected_intent": "add_item",
            "requires_handler_validation": True,
            "confidence": 0.90,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=30.0)
        assert result.repaired_intent == "add_item"
        assert result.decision == "repair_intent"

    def test_out_of_candidates_selected_intent_rejected(self):
        nlu = self._nlu("I want a burger")
        data = {
            "decision": "repair_intent",
            "selected_intent": "fantasy_intent_xyz",
            "requires_handler_validation": True,
            "confidence": 0.95,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=30.0)
        assert result.decision == "no_repair"
        assert "repaired_intent_not_in_candidates" in (result.parse_error or "")


# ---------------------------------------------------------------------------
# Timing fields logged by GptRepairService
# ---------------------------------------------------------------------------


class TestTimingFieldsLogged:
    def test_timing_fields_set_on_successful_call(self):
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        payload = {
            "decision": "repair_intent",
            "selected_intent": "add_item",
            "confidence": 0.9,
            "reason": "test",
            "requires_handler_validation": True,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(payload)
        svc._client = mock_client

        _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert result.payload_build_ms is not None
        assert result.request_ms is not None
        assert result.parse_ms is not None
        assert result.total_ms is not None
        assert result.model == "gpt-4o-mini"

    def test_prompt_chars_logged(self):
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        payload = {
            "decision": "no_repair",
            "confidence": 0.3,
            "reason": "ambiguous",
            "requires_handler_validation": True,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(payload)
        svc._client = mock_client

        _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert result.prompt_chars is not None
        assert result.prompt_chars > 0

    def test_completion_chars_logged(self):
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        payload = {
            "decision": "no_repair",
            "confidence": 0.5,
            "reason": "ok",
            "requires_handler_validation": True,
        }
        response_json = json.dumps(payload)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(payload)
        svc._client = mock_client

        _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert result.completion_chars is not None
        assert result.completion_chars == len(response_json)

    def test_timing_fields_set_on_timeout(self):
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        class APITimeoutError(Exception):
            pass

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError("timed out")
        svc._client = mock_client

        _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert result.timeout is True
        assert result.total_ms is not None
        assert result.payload_build_ms is not None

    def test_api_key_not_in_timing_fields(self):
        svc = _service(phase=2)
        FAKE_KEY = "sk-test-secretkeyvalue12345678901234567890"
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            f"Auth failed: {FAKE_KEY}"
        )
        svc._client = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": FAKE_KEY}):
            _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)

        import dataclasses
        for f in dataclasses.fields(result):
            val = getattr(result, f.name)
            if val is not None:
                assert FAKE_KEY not in str(val), f"API key in field {f.name}"


# ---------------------------------------------------------------------------
# Phase 2 — confidence gap trigger causes GPT call
# ---------------------------------------------------------------------------


class TestConfidenceGapPhase2:
    def test_phase2_calls_gpt_when_confidence_gap_small(self):
        """Phase 2 calls GPT even for KNOWN intent when confidence gap < 0.20."""
        svc = _service(phase=2)
        # Known intent but top1-top2 gap is small
        cands = _candidates("add_item", "checkout", gap=0.10)
        nlu = _nlu("I want to pay", intent=Intent.ADD_ITEM, confidence=0.80, candidates=cands)
        ir = _intent(Intent.ADD_ITEM)

        payload = {
            "decision": "no_repair",
            "confidence": 0.6,
            "reason": "local model correct",
            "requires_handler_validation": True,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(payload)
        svc._client = mock_client

        analysis, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        mock_client.chat.completions.create.assert_called_once()
        assert analysis.gpt_repair_eligible is True
        assert analysis.reason == "low_confidence_gap"

    def test_phase2_skips_gpt_when_confidence_gap_large(self):
        """Phase 2 skips GPT when confidence gap >= 0.20 for known intent."""
        svc = _service(phase=2)
        cands = _candidates("add_item", "checkout", gap=0.40)
        nlu = _nlu("add a burger", intent=Intent.ADD_ITEM, confidence=0.90, candidates=cands)
        ir = _intent(Intent.ADD_ITEM)

        with patch.object(svc, "_call_gpt") as mock_call:
            _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
            mock_call.assert_not_called()

        assert result is GPT_NOT_CALLED

    def test_phase0_no_call_even_with_small_gap(self):
        """Phase 0 never calls GPT regardless of gap."""
        svc = _service(phase=0)
        cands = _candidates("add_item", "checkout", gap=0.05)
        nlu = _nlu("I want to pay", intent=Intent.ADD_ITEM, confidence=0.80, candidates=cands)
        ir = _intent(Intent.ADD_ITEM)

        with patch.object(svc, "_call_gpt") as mock_call:
            analysis, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
            mock_call.assert_not_called()

        assert result is GPT_NOT_CALLED
        assert analysis.gpt_repair_eligible is True  # eligible, but phase 0 skips call


# ---------------------------------------------------------------------------
# GptRepairResult — applied always False
# ---------------------------------------------------------------------------


class TestAppliedAlwaysFalse:
    def test_gpt_not_called_applied_false(self):
        assert GPT_NOT_CALLED.applied is False

    def test_repair_result_applied_false(self):
        r = GptRepairResult(decision="repair_intent", repaired_intent="add_item", confidence=0.9)
        assert r.applied is False

    def test_slot_correction_has_operation_field(self):
        sc = SlotCorrection(slot_name="ITEM", old_value=None, new_value="burger", operation="add")
        assert sc.operation == "add"
        assert sc.slot_name == "ITEM"


# ---------------------------------------------------------------------------
# Missing API key — safe skip (no exception, no_repair with skipped_reason)
# ---------------------------------------------------------------------------


class TestMissingApiKey:
    def test_missing_api_key_returns_no_repair(self):
        """Absent OPENAI_API_KEY must return no_repair with skipped_reason, not raise."""
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        import unittest.mock as _mock
        with _mock.patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            analysis, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)

        assert result.decision == "no_repair"
        assert result.skipped_reason == "missing_api_key"
        assert result.applied is False

    def test_missing_api_key_does_not_raise(self):
        """No exception may propagate when the API key is absent."""
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        import unittest.mock as _mock
        with _mock.patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            # If this raises, the test fails.
            analysis, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)

        assert result is not None

    def test_missing_api_key_skipped_reason_not_in_api_key_field(self):
        """The skipped_reason string must not contain the actual key value."""
        FAKE_KEY = "sk-test-missingkeytest12345"
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        import unittest.mock as _mock
        with _mock.patch.dict("os.environ", {"OPENAI_API_KEY": FAKE_KEY}):
            # Force client to None so key lookup re-runs inside _get_client
            svc._client = None
            # Patch openai.OpenAI to raise ImportError so _get_client falls back
            # Instead, just clear the key by overriding to empty
            with _mock.patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
                _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)

        assert FAKE_KEY not in (result.skipped_reason or "")


# ---------------------------------------------------------------------------
# SLO budget enforcement
# ---------------------------------------------------------------------------


class TestSloBudget:
    def test_slo_budget_exceeded_skips_gpt(self):
        """engine_elapsed_ms > slo_budget_ms → GPT_NOT_CALLED with skipped_reason."""
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(
            phase=2, model="gpt-4o-mini", timeout_seconds=1.0, slo_budget_ms=500
        )
        svc = GptRepairService(config=cfg)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        with patch.object(svc, "_call_gpt") as mock_call:
            analysis, result = svc.run(
                nlu=nlu, intent_result=ir, state=ConversationState.IDLE,
                engine_elapsed_ms=1000.0,  # > 500ms
            )
            mock_call.assert_not_called()

        assert result is GPT_NOT_CALLED
        assert analysis.skipped_reason == "slo_budget_exceeded"

    def test_slo_budget_not_exceeded_calls_gpt(self):
        """engine_elapsed_ms <= slo_budget_ms → GPT call proceeds."""
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(
            phase=2, model="gpt-4o-mini", timeout_seconds=1.0, slo_budget_ms=1800
        )
        svc = GptRepairService(config=cfg)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        payload = {"decision": "no_repair", "requires_handler_validation": True}
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(payload)
        svc._client = mock_client

        _, result = svc.run(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE,
            engine_elapsed_ms=400.0,  # < 1800ms
        )
        mock_client.chat.completions.create.assert_called_once()

    def test_slo_budget_zero_means_unlimited(self):
        """slo_budget_ms=0 disables budget enforcement."""
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(
            phase=2, model="gpt-4o-mini", timeout_seconds=1.0, slo_budget_ms=0
        )
        svc = GptRepairService(config=cfg)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        payload = {"decision": "no_repair", "requires_handler_validation": True}
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(payload)
        svc._client = mock_client

        _, result = svc.run(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE,
            engine_elapsed_ms=99999.0,  # Very high, but budget disabled
        )
        mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# Retry logic — one retry for transient 429/5xx errors
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_one_retry_on_rate_limit_error(self):
        """RateLimitError on first attempt triggers one retry; second attempt succeeds."""
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        class RateLimitError(Exception):
            pass

        call_count = 0
        successful_payload = {"decision": "no_repair", "requires_handler_validation": True}

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("rate limited")
            return _make_response(successful_payload)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = mock_create
        svc._client = mock_client

        _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert mock_client.chat.completions.create.call_count == 2
        assert result.decision == "no_repair"
        assert result.parse_error is None

    def test_no_retry_on_non_transient_error(self):
        """Non-retryable errors are attempted exactly once."""
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ValueError("bad request")
        svc._client = mock_client

        _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert mock_client.chat.completions.create.call_count == 1
        assert result.decision == "no_repair"
        assert result.parse_error is not None

    def test_two_consecutive_transient_errors_returns_no_repair(self):
        """Two consecutive transient errors exhaust retries → no_repair."""
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        class InternalServerError(Exception):
            pass

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = InternalServerError("500 error")
        svc._client = mock_client

        _, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert mock_client.chat.completions.create.call_count == 2  # tried + retried
        assert result.decision == "no_repair"


# ---------------------------------------------------------------------------
# No cart mutation, no customer-facing GPT text
# ---------------------------------------------------------------------------


class TestNoSideEffects:
    def test_cart_not_mutated_by_gpt_result(self):
        """Even when GPT suggests repair_intent, the cart is never touched."""
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        cart_items: list = []
        cart = SimpleNamespace(
            is_empty=lambda: True,
            get_items=lambda: cart_items,
        )
        session = SimpleNamespace(
            fallback_count=0,
            last_response_key="",
            cart=cart,
        )

        payload = {
            "decision": "repair_intent",
            "selected_intent": "add_item",
            "confidence": 0.9,
            "reason": "user ordering",
            "requires_handler_validation": True,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(payload)
        svc._client = mock_client

        analysis, result = svc.run(
            nlu=nlu, intent_result=ir, state=ConversationState.IDLE, session=session
        )

        assert result.applied is False
        assert cart_items == []  # Cart unchanged
        assert result.decision == "repair_intent"  # Suggestion logged, not applied

    def test_gpt_text_never_returned_to_caller(self):
        """GptRepairService.run() never returns a spoken-response string."""
        svc = _service(phase=2)
        nlu = _nlu("I want a burger")
        ir = _intent(Intent.UNKNOWN)

        payload = {
            "decision": "ask_clarifying_question",
            "reason": "Did you want to order or checkout?",
            "requires_handler_validation": True,
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(payload)
        svc._client = mock_client

        analysis, result = svc.run(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)

        # result is a GptRepairResult — no field called "response" or "spoken_text"
        assert not hasattr(result, "response")
        assert not hasattr(result, "spoken_text")
        assert not hasattr(result, "customer_response")
        assert result.applied is False


# ---------------------------------------------------------------------------
# New compact payload fields: choices, required, history
# ---------------------------------------------------------------------------


class TestCompactPayloadNewFields:
    def _parse_payload(self, messages: list[dict]) -> dict:
        user_content = messages[1]["content"]
        return json.loads(user_content.split("\n\nOutput schema:")[0])

    def test_choices_included_when_provided(self):
        candidates = get_candidates(ConversationState.WAITING_FOR_MODIFIER.value)
        messages = build_messages(
            utterance="mushroom",
            state_name=ConversationState.WAITING_FOR_MODIFIER.value,
            candidates=candidates,
            choices=("Mushroom", "Peppers", "Onions"),
        )
        payload = self._parse_payload(messages)
        assert "choices" in payload
        assert "Mushroom" in payload["choices"]

    def test_required_included_when_provided(self):
        candidates = get_candidates(ConversationState.WAITING_FOR_MODIFIER.value)
        messages = build_messages(
            utterance="yes",
            state_name=ConversationState.WAITING_FOR_MODIFIER.value,
            candidates=candidates,
            required_missing=("MODIFIER",),
        )
        payload = self._parse_payload(messages)
        assert "required" in payload
        assert "MODIFIER" in payload["required"]

    def test_history_included_when_provided(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="actually never mind",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
            previous_turns=(("user", "I want a burger"), ("bot", "Which size?")),
        )
        payload = self._parse_payload(messages)
        assert "history" in payload
        assert len(payload["history"]) == 2
        assert payload["history"][0][0] == "user"

    def test_choices_omitted_when_empty(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="burger",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
        )
        payload = self._parse_payload(messages)
        assert "choices" not in payload
        assert "required" not in payload
        assert "history" not in payload

    def test_local_block_uses_short_keys(self):
        candidates = get_candidates(ConversationState.IDLE.value)
        messages = build_messages(
            utterance="burger",
            state_name=ConversationState.IDLE.value,
            candidates=candidates,
            selected_intent="add_item",
            selected_confidence=0.75,
        )
        payload = self._parse_payload(messages)
        local = payload["local"]
        assert "intent" in local
        assert "conf" in local
        assert "selected_intent" not in local
        assert "selected_confidence" not in local


# ---------------------------------------------------------------------------
# New output parser decisions: ok, missing_info, short keys
# ---------------------------------------------------------------------------


class TestNewOutputParserDecisions:
    def _candidates(self) -> frozenset[str]:
        return get_candidates(ConversationState.IDLE.value)

    def _nlu(self, text: str = "I want a burger") -> NLUResult:
        return NLUResult(
            effective_intent=Intent.UNKNOWN,
            intent_confidence=0.1,
            raw_text=text,
            normalized_text=text,
        )

    def test_ok_decision_normalised_to_no_repair(self):
        nlu = self._nlu()
        data = {"decision": "ok", "rhv": True, "conf": 0.95, "why": "local model correct"}
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=10.0)
        assert result.decision == "no_repair"
        assert result.parse_error is None

    def test_rhv_short_key_accepted(self):
        nlu = self._nlu()
        data = {
            "decision": "repair",
            "intent": "add_item",
            "rhv": True,
            "conf": 0.9,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=10.0)
        assert result.decision == "repair"
        assert result.repaired_intent == "add_item"

    def test_intent_short_key_accepted(self):
        nlu = self._nlu()
        data = {
            "decision": "repair",
            "intent": "checkout",
            "rhv": True,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=10.0)
        assert result.repaired_intent == "checkout"

    def test_conf_short_key_accepted(self):
        nlu = self._nlu()
        data = {
            "decision": "no_repair",
            "rhv": True,
            "conf": 0.88,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=10.0)
        assert result.confidence == pytest.approx(0.88, abs=0.01)

    def test_why_short_key_accepted(self):
        nlu = self._nlu()
        data = {
            "decision": "no_repair",
            "rhv": True,
            "why": "local model correct",
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=10.0)
        assert result.reason == "local model correct"

    def test_missing_info_decision_normalised_with_missing_slots(self):
        nlu = self._nlu("yes")
        data = {
            "decision": "missing_info",
            "missing": ["MODIFIER"],
            "rhv": True,
            "conf": 0.7,
            "why": "modifier not in utterance",
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=10.0)
        # missing_info is normalised to no_repair; missing_slots extracted
        assert result.decision == "no_repair"
        assert "MODIFIER" in result.missing_slots

    def test_missing_info_drops_invalid_slot_names(self):
        nlu = self._nlu("yes")
        data = {
            "decision": "missing_info",
            "missing": ["MODIFIER", "SAUCE", "UNKNOWN_SLOT"],
            "rhv": True,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=10.0)
        assert "MODIFIER" in result.missing_slots
        assert "SAUCE" not in result.missing_slots

    def test_short_slot_key_format_parsed(self):
        nlu = self._nlu("I want two burgers")
        data = {
            "decision": "repair",
            "intent": "add_item",
            "slots": [{"n": "QUANTITY", "v": "two", "op": "add"}],
            "rhv": True,
            "conf": 0.9,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=10.0)
        assert result.decision == "repair"
        assert result.slot_corrections_list is not None
        assert result.slot_corrections_list[0].slot_name == "QUANTITY"
        assert result.slot_corrections_list[0].new_value == "two"

    def test_unknown_slot_name_dropped_silently(self):
        nlu = self._nlu("I want extra sauce")
        data = {
            "decision": "no_repair",
            "slots": [{"n": "SAUCE_LEVEL", "v": "extra sauce", "op": "add"}],
            "rhv": True,
        }
        result = parse_output(raw=json.dumps(data), candidates=self._candidates(), nlu=nlu, latency_ms=10.0)
        # Unknown slot name dropped; response still valid
        assert result.decision == "no_repair"
        assert result.slot_corrections is None


# ---------------------------------------------------------------------------
# RepairPolicy: required_missing and choices/previous_turns extraction
# ---------------------------------------------------------------------------


class TestRepairPolicyNewTriggers:
    def _policy(self) -> RepairPolicy:
        return RepairPolicy()

    def _nlu(self, text: str, intent: Intent = Intent.UNKNOWN, slots: tuple = ()) -> NLUResult:
        return NLUResult(
            effective_intent=intent,
            intent_confidence=0.1 if intent == Intent.UNKNOWN else 0.9,
            raw_text=text,
            normalized_text=text,
            slots=slots,
        )

    def _ir(self, intent: Intent) -> IntentResult:
        return IntentResult(intent=intent, raw_text="")

    def test_required_slot_missing_triggers_eligibility(self):
        # WAITING_FOR_MODIFIER + no MODIFIER slot in utterance → eligible
        policy = self._policy()
        nlu = self._nlu("yes please", intent=Intent.UNKNOWN, slots=())
        ir = self._ir(Intent.UNKNOWN)
        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.WAITING_FOR_MODIFIER)
        assert analysis.gpt_repair_eligible is True
        assert analysis.reason == "unknown_intent_with_text"  # fires before required_slot_missing

    def test_required_slot_missing_populated_in_analysis(self):
        policy = self._policy()
        nlu = self._nlu("something good", intent=Intent.UNKNOWN, slots=())
        ir = self._ir(Intent.UNKNOWN)
        # Override high-confidence skip so we reach required_missing check
        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.WAITING_FOR_MODIFIER)
        assert "MODIFIER" in analysis.required_missing

    def test_required_slot_satisfied_not_in_required_missing(self):
        policy = self._policy()
        nlu = self._nlu(
            "mushroom",
            intent=Intent.UNKNOWN,
            slots=(SlotValue(name="MODIFIER", value="mushroom"),),
        )
        ir = self._ir(Intent.UNKNOWN)
        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.WAITING_FOR_MODIFIER)
        assert "MODIFIER" not in analysis.required_missing

    def test_choices_extracted_from_session_context(self):
        policy = self._policy()
        nlu = self._nlu("which one")
        ir = self._ir(Intent.UNKNOWN)
        ctx = SimpleNamespace(
            available_choices_values=("Mushroom", "Peppers", "Onions"),
            get_turn_memory=lambda limit=3: (),
        )
        session = SimpleNamespace(
            fallback_count=0,
            last_response_key="",
            cart=SimpleNamespace(is_empty=lambda: True, get_items=lambda: []),
            conversation_context=ctx,
        )
        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.WAITING_FOR_MODIFIER, session=session)
        assert "Mushroom" in analysis.choices
        assert "Peppers" in analysis.choices

    def test_previous_turns_extracted_from_session_context(self):
        policy = self._policy()
        nlu = self._nlu("never mind")
        ir = self._ir(Intent.UNKNOWN)
        ctx = SimpleNamespace(
            available_choices_values=(),
            get_turn_memory=lambda limit=3: (("user", "I want a burger"), ("bot", "Which size?")),
        )
        session = SimpleNamespace(
            fallback_count=0,
            last_response_key="",
            cart=SimpleNamespace(is_empty=lambda: True, get_items=lambda: []),
            conversation_context=ctx,
        )
        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE, session=session)
        assert len(analysis.previous_turns) == 2
        assert analysis.previous_turns[0] == ("user", "I want a burger")

    def test_required_slot_missing_for_side(self):
        policy = self._policy()
        nlu = self._nlu("yes", intent=Intent.UNKNOWN, slots=())
        ir = self._ir(Intent.UNKNOWN)
        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.WAITING_FOR_SIDE)
        assert "SIDE" in analysis.required_missing

    def test_required_slot_missing_for_size(self):
        policy = self._policy()
        nlu = self._nlu("medium", intent=Intent.UNKNOWN, slots=())
        ir = self._ir(Intent.UNKNOWN)
        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.WAITING_FOR_SIZE)
        assert "SIZE" in analysis.required_missing

    def test_idle_state_has_no_required_missing(self):
        policy = self._policy()
        nlu = self._nlu("I want a burger", intent=Intent.UNKNOWN, slots=())
        ir = self._ir(Intent.UNKNOWN)
        analysis = policy.check(nlu=nlu, intent_result=ir, state=ConversationState.IDLE)
        assert analysis.required_missing == ()

    def test_build_messages_includes_choices_and_required(self):
        candidates = get_candidates(ConversationState.WAITING_FOR_MODIFIER.value)
        messages = build_messages(
            utterance="yes",
            state_name=ConversationState.WAITING_FOR_MODIFIER.value,
            candidates=candidates,
            choices=("Mushroom", "Peppers"),
            required_missing=("MODIFIER",),
            previous_turns=(("user", "what are the options"), ("bot", "Mushroom or Peppers")),
        )
        user_content = messages[1]["content"]
        payload = json.loads(user_content.split("\n\nOutput schema:")[0])
        assert payload["choices"] == ["Mushroom", "Peppers"]
        assert payload["required"] == ["MODIFIER"]
        assert payload["history"][0] == ["user", "what are the options"]
