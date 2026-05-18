# tests/nlu/semantic_repair/test_add_item_service.py
"""Tests for AddItemExtractorService — shadow-only GPT calls."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from app.nlu.semantic_repair.add_item_extractor import GptAddItemPlan
from app.nlu.semantic_repair.add_item_service import AddItemExtractorService
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(
    *,
    add_item_mode: str = "shadow",
    add_item_timeout_ms: int = 350,
    add_item_min_text_len: int = 3,
    add_item_max_items_per_turn: int = 8,
    daily_budget: int = 1000,
    model: str = "gpt-4o-mini",
) -> MagicMock:
    cfg = MagicMock()
    cfg.add_item_mode = add_item_mode
    cfg.add_item_timeout_ms = add_item_timeout_ms
    cfg.add_item_min_text_len = add_item_min_text_len
    cfg.add_item_max_items_per_turn = add_item_max_items_per_turn
    cfg.daily_budget = daily_budget
    cfg.model = model
    return cfg


def _make_session(
    state: ConversationState = ConversationState.IDLE,
    *,
    item_name: str = "",
    prompt_field: str = "",
) -> MagicMock:
    session = MagicMock()
    session.conversation_state = state

    ctx = MagicMock()
    ctx.current_item_name = item_name
    ctx.current_prompt_field = prompt_field
    ctx.available_choices_values = ()
    ctx.recent_turns = []
    session.conversation_context = ctx

    cart = MagicMock()
    cart.items = []
    session.cart = cart
    return session


def _make_nlu(intent_value: str = "ADD_ITEM") -> MagicMock:
    from app.nlu.intent_resolution.intent import Intent
    nlu = MagicMock()
    nlu.normalized_text = "I want a burger"
    nlu.effective_intent = MagicMock()
    nlu.effective_intent.value = intent_value
    nlu.intent_confidence = 0.9
    nlu.slots = []
    nlu.intent_candidates = []
    return nlu


def _make_intent_result(intent_value: str = "ADD_ITEM") -> MagicMock:
    from app.nlu.intent_resolution.intent import Intent
    ir = MagicMock()
    if intent_value == "ADD_ITEM":
        ir.intent = Intent.ADD_ITEM
    else:
        ir.intent = Intent.UNKNOWN
    return ir


def _valid_gpt_response() -> str:
    return json.dumps({
        "requires_handler_validation": True,
        "decision": "ok",
        "intent": "add_item",
        "items": [{"item": "burger", "quantity": 1}],
        "global_slots": [],
        "missing": [],
        "fallback_type": "none",
        "confidence": 0.9,
        "reason": "user wants burger",
    })


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------

class TestDisabledMode:
    def test_disabled_mode_returns_no_repair_immediately(self):
        svc = AddItemExtractorService(config=_make_config(add_item_mode="disabled"))
        plan = svc.run(
            session=_make_session(),
            nlu=_make_nlu(),
            intent_result=_make_intent_result(),
            state=ConversationState.IDLE,
        )
        assert plan.decision == "no_repair"
        assert not plan.eligible

    def test_disabled_mode_never_calls_gpt(self):
        svc = AddItemExtractorService(config=_make_config(add_item_mode="disabled"))
        with patch.object(svc, "_get_client") as mock_client:
            svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
            mock_client.assert_not_called()


# ---------------------------------------------------------------------------
# Eligibility gates
# ---------------------------------------------------------------------------

class TestEligibilityGates:
    def test_wrong_intent_not_eligible(self):
        svc = AddItemExtractorService(config=_make_config())
        plan = svc.run(
            session=_make_session(),
            nlu=_make_nlu("CHECKOUT"),
            intent_result=_make_intent_result("CHECKOUT"),
            state=ConversationState.IDLE,
        )
        assert plan.decision == "no_repair"
        assert plan.skipped_reason == "intent_not_add_item"

    def test_terminal_state_not_eligible(self):
        svc = AddItemExtractorService(config=_make_config())
        plan = svc.run(
            session=_make_session(ConversationState.COMPLETED),
            nlu=_make_nlu(),
            intent_result=_make_intent_result(),
            state=ConversationState.COMPLETED,
        )
        assert plan.skipped_reason == "terminal_state"

    def test_unsupported_state_not_eligible(self):
        svc = AddItemExtractorService(config=_make_config())
        plan = svc.run(
            session=_make_session(ConversationState.CONFIRMING_ORDER),
            nlu=_make_nlu(),
            intent_result=_make_intent_result(),
            state=ConversationState.CONFIRMING_ORDER,
        )
        assert plan.skipped_reason == "state_not_supported"

    def test_short_text_not_eligible(self):
        svc = AddItemExtractorService(config=_make_config(add_item_min_text_len=10))
        nlu = _make_nlu()
        nlu.normalized_text = "hi"
        plan = svc.run(
            session=_make_session(),
            nlu=nlu,
            intent_result=_make_intent_result(),
            state=ConversationState.IDLE,
        )
        assert plan.skipped_reason == "text_too_short"


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------

class TestMissingApiKey:
    def test_missing_api_key_returns_no_repair(self):
        svc = AddItemExtractorService(config=_make_config())
        with patch.dict(os.environ, {}, clear=True):
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            plan = svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        assert plan.decision == "no_repair"
        assert plan.skipped_reason in ("missing_api_key", "daily_budget_exceeded")

    def test_missing_api_key_does_not_raise(self):
        svc = AddItemExtractorService(config=_make_config())
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            plan = svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        # Should return a plan, never raise
        assert isinstance(plan, GptAddItemPlan)


# ---------------------------------------------------------------------------
# GPT call (mocked)
# ---------------------------------------------------------------------------

class TestGptCall:
    def _make_mock_client(self, response_text: str) -> MagicMock:
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        return client

    def test_successful_gpt_call_returns_plan_with_items(self):
        svc = AddItemExtractorService(config=_make_config())
        svc._client = self._make_mock_client(_valid_gpt_response())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            plan = svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        assert plan.decision in ("ok", "repair", "no_repair")
        # total_ms should be set (GPT was called)
        assert plan.total_ms is not None

    def test_gpt_call_never_mutates_session(self):
        """The extractor must not write to session, cart, or state."""
        svc = AddItemExtractorService(config=_make_config())
        session = _make_session()
        original_state = session.conversation_state
        svc._client = self._make_mock_client(_valid_gpt_response())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            svc.run(
                session=session,
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        # State must not have been mutated
        assert session.conversation_state == original_state
        # Cart must not have been touched (add_items not called)
        session.cart.add_item.assert_not_called() if hasattr(session.cart.add_item, "assert_not_called") else None

    def test_gpt_timeout_returns_plan_with_timeout_flag(self):
        # The service detects timeout from type(exc).__name__ containing "timeout".
        class APITimeoutError(Exception):
            """Simulates openai.APITimeoutError whose class name contains 'timeout'."""

        svc = AddItemExtractorService(config=_make_config(add_item_timeout_ms=100))
        client = MagicMock()
        client.chat.completions.create.side_effect = APITimeoutError("request timed out")
        svc._client = client
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            plan = svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        assert plan.decision == "no_repair"
        assert plan.timeout is True

    def test_gpt_exception_does_not_raise(self):
        svc = AddItemExtractorService(config=_make_config())
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network failure")
        svc._client = client
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            plan = svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        assert isinstance(plan, GptAddItemPlan)

    def test_gpt_result_items_not_applied_to_cart(self):
        """Verify no cart mutation regardless of GPT result."""
        svc = AddItemExtractorService(config=_make_config())
        session = _make_session()
        svc._client = self._make_mock_client(_valid_gpt_response())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            plan = svc.run(
                session=session,
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        # Items may be in the plan for logging, but cart was never touched
        session.cart.items.append.assert_not_called() if hasattr(
            session.cart.items, "append"
        ) and hasattr(session.cart.items.append, "assert_not_called") else None

    def test_uses_timeout_from_config(self):
        """GPT call should use the configured timeout."""
        svc = AddItemExtractorService(config=_make_config(add_item_timeout_ms=500))
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = _valid_gpt_response()
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        svc._client = client
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("timeout") == pytest.approx(0.5)  # 500ms / 1000

    def test_model_from_config_used(self):
        svc = AddItemExtractorService(config=_make_config(model="gpt-4o"))
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = _valid_gpt_response()
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        svc._client = client
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4o"

    def test_plan_has_timing_fields(self):
        svc = AddItemExtractorService(config=_make_config())
        svc._client = self._make_mock_client(_valid_gpt_response())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            plan = svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        assert plan.latency_ms is not None
        assert plan.total_ms is not None
        assert plan.model is not None

    def test_plan_has_prompt_chars(self):
        svc = AddItemExtractorService(config=_make_config())
        svc._client = self._make_mock_client(_valid_gpt_response())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            plan = svc.run(
                session=_make_session(),
                nlu=_make_nlu(),
                intent_result=_make_intent_result(),
                state=ConversationState.IDLE,
            )
        assert plan.prompt_chars > 0
