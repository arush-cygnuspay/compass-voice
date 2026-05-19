# tests/services/test_smart_turn_planner.py
"""Tests for SmartTurnPlanner service and policy.

Covers:
  TC-01  Feature flag disabled → plan_smart_turn returns None
  TC-02  Missing OPENAI_API_KEY → returns skipped plan (not None)
  TC-03  Daily budget exceeded → returns skipped plan
  TC-04  GPT call times out → plan_smart_turn returns None
  TC-05  GPT returns invalid JSON → returns unclear plan (not None)
  TC-06  Happy-path compound utterance → parsed plan returned
  TC-07  Correction phrase → correction plan with SmartTurnCorrection
  TC-08  validate_smart_plan blocks low-confidence plan
  TC-09  validate_smart_plan blocks item not in menu_context
  TC-10  validate_smart_plan passes when all gates clear
  TC-11  should_use_smart_planner: correction prefix → True
  TC-12  should_use_smart_planner: terminal state → False
  TC-13  should_use_smart_planner: compound utterance in IDLE → True
  TC-14  should_use_smart_planner: low confidence WAITING_FOR_MODIFIER → True
  TC-15  plan is never applied by the service (no state/cart mutation)
  TC-16  SmartTurnPlan.to_dict() produces JSON-serialisable output
  TC-17  _parse_plan handles markdown fences gracefully
  TC-18  _parse_plan clamps quantity to safe range
  TC-19  validate_smart_plan: correction decision without correction field → blocked
"""
from __future__ import annotations

import json
import os
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure tests run without the openai package installed
# ---------------------------------------------------------------------------
if "openai" not in sys.modules:
    _fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *a, **kw):
            pass

    _fake_openai.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    sys.modules["openai"] = _fake_openai


from app.services.smart_turn_planner import (
    SMART_TURN_NOT_CALLED,
    SmartTurnCorrection,
    SmartTurnItem,
    SmartTurnModifier,
    SmartTurnPlan,
    SmartTurnSide,
    _build_user_message,
    _parse_plan,
    plan_smart_turn,
)
from app.services.smart_turn_policy import (
    ValidationResult,
    _has_correction_signal,
    _is_compound_utterance,
    build_menu_context_for_turn,
    should_use_smart_planner,
    validate_smart_plan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _EnvPatch:
    """Context manager that sets env vars for a test and restores originals."""

    _KEYS = frozenset({
        "SMART_TURN_PLANNER_ENABLED",
        "OPENAI_API_KEY",
        "SMART_TURN_PLANNER_TIMEOUT_MS",
        "SMART_TURN_PLANNER_DAILY_BUDGET",
        "OPENAI_MODEL_FAST",
    })

    def __init__(self, overrides: dict[str, str]) -> None:
        self._overrides = overrides
        self._saved: dict[str, str | None] = {}

    def __enter__(self):
        for key in self._KEYS:
            self._saved[key] = os.environ.get(key)
            os.environ.pop(key, None)
        for key, value in self._overrides.items():
            os.environ[key] = value
        return self

    def __exit__(self, *_):
        for key in self._KEYS:
            original = self._saved.get(key)
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _good_json_response(
    decision: str = "add_items",
    confidence: float = 0.9,
    items: list | None = None,
    correction: dict | None = None,
    reason: str = "compound utterance",
) -> str:
    data: dict = {
        "decision": decision,
        "confidence": confidence,
        "items": items or [],
        "correction": correction,
        "response": None,
        "reason": reason,
    }
    return json.dumps(data)


# ---------------------------------------------------------------------------
# TC-01: Feature flag disabled
# ---------------------------------------------------------------------------

class TestFeatureFlagDisabled(unittest.TestCase):

    def test_plan_smart_turn_returns_none_when_disabled(self):
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "false",
            "OPENAI_API_KEY": "sk-test-key",
        }):
            result = plan_smart_turn(
                transcript="burger with mozzarella",
                state="IDLE",
                local_intent="ADD_ITEM",
                local_confidence=0.6,
                menu_context=["Burger"],
                cart_snapshot=[],
                trigger_reason="compound_utterance",
            )
        self.assertIsNone(result)

    def test_plan_smart_turn_returns_none_when_not_set(self):
        with _EnvPatch({"OPENAI_API_KEY": "sk-test-key"}):
            result = plan_smart_turn(
                transcript="burger with mozzarella",
                state="IDLE",
                local_intent="ADD_ITEM",
                local_confidence=0.6,
                menu_context=["Burger"],
                cart_snapshot=[],
            )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# TC-02: Missing OPENAI_API_KEY
# ---------------------------------------------------------------------------

class TestMissingApiKey(unittest.TestCase):

    def test_returns_skipped_plan_not_none(self):
        with _EnvPatch({"SMART_TURN_PLANNER_ENABLED": "true"}):
            # OPENAI_API_KEY not set
            result = plan_smart_turn(
                transcript="burger with mozzarella",
                state="IDLE",
                local_intent="ADD_ITEM",
                local_confidence=0.6,
                menu_context=["Burger"],
                cart_snapshot=[],
                trigger_reason="compound_utterance",
            )
        self.assertIsNotNone(result)
        assert result is not None  # type narrowing
        self.assertFalse(result.gpt_called)
        self.assertEqual(result.skipped_reason, "missing_api_key")
        self.assertEqual(result.decision, "no_action")


# ---------------------------------------------------------------------------
# TC-03: Daily budget exceeded
# ---------------------------------------------------------------------------

class TestDailyBudgetExceeded(unittest.TestCase):

    def test_returns_skipped_plan_when_budget_zero(self):
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test-key",
            "SMART_TURN_PLANNER_DAILY_BUDGET": "1",
        }):
            # Patch the budget to simulate exceeded state
            from app.services import smart_turn_planner as _mod
            mock_budget = MagicMock()
            mock_budget.try_consume.return_value = False
            with patch.object(_mod, "_get_budget", return_value=mock_budget):
                result = plan_smart_turn(
                    transcript="burger with mozzarella",
                    state="IDLE",
                    local_intent="ADD_ITEM",
                    local_confidence=0.6,
                    menu_context=["Burger"],
                    cart_snapshot=[],
                )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.gpt_called)
        self.assertEqual(result.skipped_reason, "daily_budget_exceeded")


# ---------------------------------------------------------------------------
# TC-04: GPT call times out
# ---------------------------------------------------------------------------

class TestGptTimeout(unittest.TestCase):

    def test_returns_none_on_timeout(self):
        from concurrent.futures import TimeoutError as FuturesTimeout

        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test-key",
            "SMART_TURN_PLANNER_TIMEOUT_MS": "1",
        }):
            from app.services import smart_turn_planner as _mod

            mock_budget = MagicMock()
            mock_budget.try_consume.return_value = True

            mock_future = MagicMock()
            mock_future.result.side_effect = FuturesTimeout()

            mock_executor = MagicMock()
            mock_executor.submit.return_value = mock_future

            with patch.object(_mod, "_get_budget", return_value=mock_budget), \
                 patch.object(_mod, "_get_executor", return_value=mock_executor):
                result = plan_smart_turn(
                    transcript="burger with mozzarella",
                    state="IDLE",
                    local_intent="ADD_ITEM",
                    local_confidence=0.6,
                    menu_context=["Burger"],
                    cart_snapshot=[],
                    trigger_reason="compound_utterance",
                )

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# TC-05: GPT returns invalid JSON
# ---------------------------------------------------------------------------

class TestInvalidJson(unittest.TestCase):

    def test_returns_unclear_plan_not_none(self):
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test-key",
        }):
            from app.services import smart_turn_planner as _mod

            mock_budget = MagicMock()
            mock_budget.try_consume.return_value = True

            mock_future = MagicMock()
            mock_future.result.return_value = "this is not json {{{"

            mock_executor = MagicMock()
            mock_executor.submit.return_value = mock_future

            with patch.object(_mod, "_get_budget", return_value=mock_budget), \
                 patch.object(_mod, "_get_executor", return_value=mock_executor):
                result = plan_smart_turn(
                    transcript="aaasdfkjhqwerty",
                    state="IDLE",
                    local_intent="ADD_ITEM",
                    local_confidence=0.3,
                    menu_context=[],
                    cart_snapshot=[],
                )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.gpt_called)
        self.assertEqual(result.decision, "unclear")


# ---------------------------------------------------------------------------
# TC-06: Happy-path compound utterance
# ---------------------------------------------------------------------------

class TestHappyPathCompound(unittest.TestCase):

    def test_compound_add_items_parsed_correctly(self):
        raw_json = _good_json_response(
            decision="add_items",
            confidence=0.92,
            items=[
                {
                    "item_name": "Chicken Burger",
                    "quantity": 1,
                    "size": None,
                    "variant": None,
                    "modifiers": [{"name": "Mozzarella", "operation": "add"}],
                    "sides": [{"name": "Coke", "quantity": 1}],
                }
            ],
            reason="compound utterance with modifier and side",
        )

        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test-key",
        }):
            from app.services import smart_turn_planner as _mod

            mock_budget = MagicMock()
            mock_budget.try_consume.return_value = True

            mock_future = MagicMock()
            mock_future.result.return_value = raw_json

            mock_executor = MagicMock()
            mock_executor.submit.return_value = mock_future

            with patch.object(_mod, "_get_budget", return_value=mock_budget), \
                 patch.object(_mod, "_get_executor", return_value=mock_executor):
                result = plan_smart_turn(
                    transcript="chicken burger with mozzarella and a coke",
                    state="IDLE",
                    local_intent="ADD_ITEM",
                    local_confidence=0.72,
                    menu_context=["Chicken Burger", "Coke"],
                    cart_snapshot=[],
                    trigger_reason="compound_utterance",
                )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.gpt_called)
        self.assertEqual(result.decision, "add_items")
        self.assertAlmostEqual(result.confidence, 0.92, places=2)
        self.assertEqual(len(result.items), 1)
        item = result.items[0]
        self.assertEqual(item.item_name, "Chicken Burger")
        self.assertEqual(len(item.modifiers), 1)
        self.assertEqual(item.modifiers[0].name, "Mozzarella")
        self.assertEqual(len(item.sides), 1)
        self.assertEqual(item.sides[0].name, "Coke")
        self.assertEqual(result.trigger_reason, "compound_utterance")


# ---------------------------------------------------------------------------
# TC-07: Correction phrase
# ---------------------------------------------------------------------------

class TestCorrectionPhrase(unittest.TestCase):

    def test_correction_plan_parsed(self):
        raw_json = _good_json_response(
            decision="correction",
            confidence=0.88,
            items=[],
            correction={
                "original_text": "Bourbon Burger",
                "corrected_text": "Chicken Burger",
            },
            reason="user corrected item",
        )

        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test-key",
        }):
            from app.services import smart_turn_planner as _mod

            mock_budget = MagicMock()
            mock_budget.try_consume.return_value = True

            mock_future = MagicMock()
            mock_future.result.return_value = raw_json

            mock_executor = MagicMock()
            mock_executor.submit.return_value = mock_future

            with patch.object(_mod, "_get_budget", return_value=mock_budget), \
                 patch.object(_mod, "_get_executor", return_value=mock_executor):
                result = plan_smart_turn(
                    transcript="actually I said chicken burger",
                    state="IDLE",
                    local_intent="ADD_ITEM",
                    local_confidence=0.5,
                    menu_context=["Bourbon Burger", "Chicken Burger"],
                    cart_snapshot=["Bourbon Burger"],
                    trigger_reason="correction_phrase",
                )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.decision, "correction")
        self.assertIsNotNone(result.correction)
        assert result.correction is not None
        self.assertEqual(result.correction.original_text, "Bourbon Burger")
        self.assertEqual(result.correction.corrected_text, "Chicken Burger")


# ---------------------------------------------------------------------------
# TC-08: validate_smart_plan blocks low-confidence plan
# ---------------------------------------------------------------------------

class TestValidateSmartPlanLowConfidence(unittest.TestCase):

    def test_low_confidence_is_blocked(self):
        plan = SmartTurnPlan(
            decision="add_items",
            confidence=0.5,   # below 0.75 threshold
            items=(SmartTurnItem(item_name="Burger"),),
            gpt_called=True,
            trigger_reason="compound_utterance",
        )
        result = validate_smart_plan(
            plan,
            menu_context=["Burger"],
            cart_snapshot=[],
            state="IDLE",
            local_intent="ADD_ITEM",
        )
        self.assertFalse(result.is_safe)
        self.assertEqual(result.block_reason, "low_confidence")


# ---------------------------------------------------------------------------
# TC-09: validate_smart_plan blocks item not in menu_context
# ---------------------------------------------------------------------------

class TestValidateSmartPlanItemNotInMenu(unittest.TestCase):

    def test_item_not_in_menu_context_is_blocked(self):
        plan = SmartTurnPlan(
            decision="add_items",
            confidence=0.9,
            items=(SmartTurnItem(item_name="Unicorn Burger"),),
            gpt_called=True,
            trigger_reason="compound_utterance",
        )
        result = validate_smart_plan(
            plan,
            menu_context=["Chicken Burger", "Bourbon Burger"],
            cart_snapshot=[],
            state="IDLE",
            local_intent="ADD_ITEM",
        )
        self.assertFalse(result.is_safe)
        self.assertEqual(result.block_reason, "item_not_in_menu_context")


# ---------------------------------------------------------------------------
# TC-10: validate_smart_plan passes when all gates clear
# ---------------------------------------------------------------------------

class TestValidateSmartPlanAllGatesPass(unittest.TestCase):

    def test_valid_plan_is_safe(self):
        plan = SmartTurnPlan(
            decision="add_items",
            confidence=0.85,
            items=(SmartTurnItem(
                item_name="Chicken Burger",
                modifiers=(SmartTurnModifier(name="Mozzarella"),),
            ),),
            gpt_called=True,
            trigger_reason="compound_utterance",
        )
        result = validate_smart_plan(
            plan,
            menu_context=["Chicken Burger", "Mozzarella"],
            cart_snapshot=[],
            state="IDLE",
            local_intent="ADD_ITEM",
        )
        self.assertTrue(result.is_safe)
        self.assertEqual(result.reason, "all_gates_passed")

    def test_empty_menu_context_skips_item_gate(self):
        """When caller provides no menu_context, gate 5 is skipped (permissive)."""
        plan = SmartTurnPlan(
            decision="add_items",
            confidence=0.85,
            items=(SmartTurnItem(item_name="Whatever Item"),),
            gpt_called=True,
        )
        result = validate_smart_plan(
            plan,
            menu_context=[],   # no context → gate skipped
            cart_snapshot=[],
            state="IDLE",
            local_intent="ADD_ITEM",
        )
        self.assertTrue(result.is_safe)


# ---------------------------------------------------------------------------
# TC-11 / TC-12 / TC-13 / TC-14: should_use_smart_planner
# ---------------------------------------------------------------------------

class TestShouldUseSmartPlanner(unittest.TestCase):

    def test_correction_prefix_triggers_planner(self):
        ok, reason = should_use_smart_planner(
            transcript="actually I want the chicken burger",
            state="IDLE",
            local_intent="ADD_ITEM",
            local_confidence=0.6,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "correction_phrase")

    def test_terminal_state_never_triggers(self):
        for state in ("COMPLETED", "ERROR_RECOVERY", "TRANSFERRING_TO_HUMAN_AGENT",
                      "WAITING_FOR_PAYMENT", "CONFIRMING_ORDER"):
            ok, reason = should_use_smart_planner(
                transcript="burger with mozzarella",
                state=state,
                local_intent="ADD_ITEM",
                local_confidence=0.3,
            )
            self.assertFalse(ok, f"should NOT trigger for state={state}")
            self.assertIn("state", reason)

    def test_compound_utterance_in_idle_triggers(self):
        ok, reason = should_use_smart_planner(
            transcript="chicken burger with mozzarella and a coke",
            state="IDLE",
            local_intent="ADD_ITEM",
            local_confidence=0.8,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "compound_utterance")

    def test_low_confidence_waiting_for_modifier_triggers(self):
        ok, reason = should_use_smart_planner(
            transcript="macarola cheese",
            state="WAITING_FOR_MODIFIER",
            local_intent="ADD_ITEM",
            local_confidence=0.4,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "low_confidence_waiting_state")

    def test_high_confidence_simple_utterance_does_not_trigger(self):
        ok, reason = should_use_smart_planner(
            transcript="bourbon burger",
            state="IDLE",
            local_intent="ADD_ITEM",
            local_confidence=0.92,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "local_path_sufficient")

    def test_empty_transcript_does_not_trigger(self):
        ok, reason = should_use_smart_planner(
            transcript="",
            state="IDLE",
            local_intent="ADD_ITEM",
            local_confidence=0.9,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "empty_transcript")

    def test_unknown_state_does_not_trigger(self):
        ok, reason = should_use_smart_planner(
            transcript="burger with mozzarella and a coke",
            state="SOME_UNKNOWN_STATE",
            local_intent="ADD_ITEM",
            local_confidence=0.3,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "state_not_eligible")


# ---------------------------------------------------------------------------
# TC-15: Plan never applied by the service
# ---------------------------------------------------------------------------

class TestPlanNeverApplied(unittest.TestCase):
    """The plan is a value object — it has no mutate methods and cannot touch
    cart, session, or FSM state.  This test documents the invariant."""

    def test_plan_has_no_apply_method(self):
        plan = SmartTurnPlan(decision="add_items", gpt_called=True, confidence=0.9)
        self.assertFalse(hasattr(plan, "apply"))
        self.assertFalse(hasattr(plan, "execute"))
        self.assertFalse(hasattr(plan, "mutate"))

    def test_plan_is_frozen(self):
        plan = SmartTurnPlan(decision="add_items", gpt_called=True, confidence=0.9)
        with self.assertRaises((AttributeError, TypeError)):
            plan.decision = "correction"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TC-16: SmartTurnPlan.to_dict() is JSON-serialisable
# ---------------------------------------------------------------------------

class TestToDict(unittest.TestCase):

    def test_full_plan_serialises(self):
        plan = SmartTurnPlan(
            decision="add_items",
            confidence=0.85,
            items=(
                SmartTurnItem(
                    item_name="Chicken Burger",
                    quantity=1,
                    modifiers=(SmartTurnModifier(name="Mozzarella", operation="add"),),
                    sides=(SmartTurnSide(name="Coke", quantity=1),),
                ),
            ),
            correction=None,
            response=None,
            reason="compound utterance",
            trigger_reason="compound_utterance",
            latency_ms=345.2,
            gpt_called=True,
            model="gpt-4o-mini",
        )
        d = plan.to_dict()
        # Must serialize without error
        serialised = json.dumps(d)
        parsed = json.loads(serialised)
        self.assertEqual(parsed["decision"], "add_items")
        self.assertEqual(len(parsed["items"]), 1)
        self.assertEqual(parsed["items"][0]["item_name"], "Chicken Burger")
        self.assertEqual(parsed["items"][0]["modifiers"][0]["name"], "Mozzarella")

    def test_sentinel_serialises(self):
        d = SMART_TURN_NOT_CALLED.to_dict()
        _ = json.dumps(d)  # must not raise

    def test_plan_with_correction_serialises(self):
        plan = SmartTurnPlan(
            decision="correction",
            confidence=0.88,
            correction=SmartTurnCorrection(
                original_text="Bourbon Burger",
                corrected_text="Chicken Burger",
            ),
            gpt_called=True,
        )
        d = plan.to_dict()
        _ = json.dumps(d)
        self.assertIsNotNone(d["correction"])
        self.assertEqual(d["correction"]["original_text"], "Bourbon Burger")


# ---------------------------------------------------------------------------
# TC-17: _parse_plan handles markdown fences
# ---------------------------------------------------------------------------

class TestParsePlanMarkdownFences(unittest.TestCase):

    def test_strips_code_fences(self):
        raw = "```json\n" + _good_json_response(
            decision="add_items",
            confidence=0.8,
            items=[{"item_name": "Burger", "quantity": 1, "size": None,
                    "variant": None, "modifiers": [], "sides": []}],
        ) + "\n```"
        plan = _parse_plan(raw, trigger_reason="test", model="gpt-4o-mini", latency_ms=100.0)
        self.assertEqual(plan.decision, "add_items")
        self.assertEqual(plan.items[0].item_name, "Burger")

    def test_plain_code_fences(self):
        raw = "```\n" + _good_json_response() + "\n```"
        plan = _parse_plan(raw, trigger_reason="test", model="gpt-4o-mini", latency_ms=100.0)
        self.assertEqual(plan.decision, "add_items")


# ---------------------------------------------------------------------------
# TC-18: _parse_plan clamps quantity
# ---------------------------------------------------------------------------

class TestParsePlanQuantityClamping(unittest.TestCase):

    def test_negative_quantity_becomes_one(self):
        raw = json.dumps({
            "decision": "add_items",
            "confidence": 0.9,
            "items": [{"item_name": "Burger", "quantity": -5}],
            "correction": None,
            "response": None,
            "reason": "test",
        })
        plan = _parse_plan(raw, trigger_reason="test", model="gpt-4o-mini", latency_ms=10.0)
        self.assertEqual(plan.items[0].quantity, 1)

    def test_huge_quantity_clamped_to_20(self):
        raw = json.dumps({
            "decision": "add_items",
            "confidence": 0.9,
            "items": [{"item_name": "Burger", "quantity": 9999}],
            "correction": None,
            "response": None,
            "reason": "test",
        })
        plan = _parse_plan(raw, trigger_reason="test", model="gpt-4o-mini", latency_ms=10.0)
        self.assertEqual(plan.items[0].quantity, 20)

    def test_item_with_empty_name_dropped(self):
        raw = json.dumps({
            "decision": "add_items",
            "confidence": 0.9,
            "items": [
                {"item_name": "", "quantity": 1},       # should be dropped
                {"item_name": "Burger", "quantity": 1},  # should be kept
            ],
            "correction": None,
            "response": None,
            "reason": "test",
        })
        plan = _parse_plan(raw, trigger_reason="test", model="gpt-4o-mini", latency_ms=10.0)
        self.assertEqual(len(plan.items), 1)
        self.assertEqual(plan.items[0].item_name, "Burger")


# ---------------------------------------------------------------------------
# TC-19: validate_smart_plan: correction without correction field
# ---------------------------------------------------------------------------

class TestValidateSmartPlanCorrectionGate(unittest.TestCase):

    def test_correction_decision_without_field_is_blocked(self):
        plan = SmartTurnPlan(
            decision="correction",
            confidence=0.9,
            correction=None,   # missing field
            gpt_called=True,
        )
        result = validate_smart_plan(
            plan,
            menu_context=["Burger"],
            cart_snapshot=[],
            state="IDLE",
            local_intent="ADD_ITEM",
        )
        self.assertFalse(result.is_safe)
        self.assertEqual(result.block_reason, "missing_correction_field")

    def test_gpt_not_called_is_blocked(self):
        plan = SmartTurnPlan(
            decision="add_items",
            confidence=0.9,
            items=(SmartTurnItem(item_name="Burger"),),
            gpt_called=False,  # sentinel / skip
        )
        result = validate_smart_plan(
            plan,
            menu_context=["Burger"],
            cart_snapshot=[],
            state="IDLE",
            local_intent="ADD_ITEM",
        )
        self.assertFalse(result.is_safe)
        self.assertEqual(result.block_reason, "gpt_not_called")


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_has_correction_signal_various_prefixes(self):
        for phrase in [
            "actually I meant the chicken",
            "scratch that I want a coke",
            "no I said mozzarella",
            "i mean the bourbon burger",
            "change it to a veggie burger",
        ]:
            with self.subTest(phrase=phrase):
                self.assertTrue(_has_correction_signal(phrase.lower()))

    def test_has_correction_signal_false_for_normal(self):
        self.assertFalse(_has_correction_signal("bourbon burger"))
        self.assertFalse(_has_correction_signal("yes please"))

    def test_is_compound_utterance_detects_and(self):
        self.assertTrue(_is_compound_utterance("burger and a coke"))

    def test_is_compound_utterance_detects_with(self):
        self.assertTrue(_is_compound_utterance("chicken burger with mozzarella"))

    def test_is_compound_utterance_detects_comma(self):
        self.assertTrue(_is_compound_utterance("burger, fries, coke"))

    def test_is_compound_utterance_false_for_simple(self):
        self.assertFalse(_is_compound_utterance("bourbon burger"))

    def test_build_menu_context_clips_to_max(self):
        names = [f"Item {i}" for i in range(20)]
        ctx = build_menu_context_for_turn(item_names=names, max_items=5)
        self.assertEqual(len(ctx), 5)

    def test_build_user_message_does_not_include_api_key(self):
        msg = _build_user_message(
            transcript="burger",
            state="IDLE",
            local_intent="ADD_ITEM",
            local_confidence=0.8,
            menu_context=["Burger"],
            cart_snapshot=[],
            last_cart_diff=None,
            previous_turns=[],
        )
        self.assertNotIn("sk-", msg)
        self.assertNotIn("api_key", msg.lower())


if __name__ == "__main__":
    unittest.main()
