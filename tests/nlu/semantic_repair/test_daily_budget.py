# tests/nlu/semantic_repair/test_daily_budget.py
"""Tests for GptDailyBudget in-memory daily call counter."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.nlu.semantic_repair.daily_budget import GptDailyBudget, _utc_date_str


class TestGptDailyBudget:
    def test_consume_under_budget_returns_true(self):
        budget = GptDailyBudget(limit=10)
        assert budget.try_consume() is True

    def test_consume_increments_count(self):
        budget = GptDailyBudget(limit=10)
        budget.try_consume()
        budget.try_consume()
        assert budget.count == 2

    def test_consume_at_limit_returns_false(self):
        budget = GptDailyBudget(limit=3)
        budget.try_consume()
        budget.try_consume()
        budget.try_consume()
        # At limit: next call should be blocked
        assert budget.try_consume() is False

    def test_count_not_incremented_when_blocked(self):
        budget = GptDailyBudget(limit=2)
        budget.try_consume()
        budget.try_consume()
        budget.try_consume()  # blocked
        assert budget.count == 2

    def test_unlimited_budget_always_allows(self):
        budget = GptDailyBudget(limit=0)
        for _ in range(1000):
            assert budget.try_consume() is True

    def test_is_exceeded_false_when_under(self):
        budget = GptDailyBudget(limit=5)
        assert budget.is_exceeded() is False

    def test_is_exceeded_true_when_at_limit(self):
        budget = GptDailyBudget(limit=2)
        budget.try_consume()
        budget.try_consume()
        assert budget.is_exceeded() is True

    def test_budget_resets_on_new_utc_date(self):
        budget = GptDailyBudget(limit=5)
        # Consume all
        for _ in range(5):
            budget.try_consume()
        assert budget.try_consume() is False  # blocked

        # Simulate day rollover by patching _utc_date_str inside budget
        tomorrow = "2099-12-31"
        with patch("app.nlu.semantic_repair.daily_budget._utc_date_str", return_value=tomorrow):
            # First call on new day resets the counter
            assert budget.try_consume() is True
            assert budget.count == 1

    def test_limit_property(self):
        budget = GptDailyBudget(limit=42)
        assert budget.limit == 42


class TestGptDailyBudgetSkipInRepairService:
    """Verify that daily budget exceeded triggers gpt_skipped_reason in repair_service."""

    def test_budget_exceeded_returns_skipped_reason(self):
        from app.config.semantic_repair import SemanticRepairConfig
        from app.nlu.intent_resolution.intent import Intent
        from app.nlu.intent_resolution.intent_result import IntentResult
        from app.nlu.nlu_result import NLUResult
        from app.nlu.semantic_repair.repair_service import GptRepairService
        from app.state_machine.models.conversation_state import ConversationState

        cfg = SemanticRepairConfig(
            phase=2,
            model="gpt-4o-mini",
            timeout_seconds=0.1,
            call_mode="eligible_only",
            daily_budget=1,
        )
        svc = GptRepairService(config=cfg)
        # Exhaust the budget
        svc._daily_budget.try_consume()  # 1 allowed call consumed

        nlu = NLUResult(
            effective_intent=Intent.UNKNOWN,
            intent_confidence=0.1,
            raw_text="I want a burger please",
            normalized_text="I want a burger please",
        )
        ir = IntentResult(intent=Intent.UNKNOWN, raw_text="I want a burger please")

        analysis, result = svc.run(
            nlu=nlu,
            intent_result=ir,
            state=ConversationState.IDLE,
        )
        from app.nlu.semantic_repair.gpt_repair_result import GPT_NOT_CALLED
        assert result is GPT_NOT_CALLED
        assert analysis.skipped_reason == "daily_budget_exceeded"
