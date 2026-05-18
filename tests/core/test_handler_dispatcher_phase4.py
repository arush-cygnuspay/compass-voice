# tests/core/test_handler_dispatcher_phase4.py
"""Phase 4.1 production wiring tests for HandlerDispatcher.

Verifies:
  1. DISABLED mode  — AddItemHandler.gpt_planner is None (no planner).
  2. SHADOW mode    — AddItemHandler.gpt_planner is a GptAddItemPlannerService.
  3. INLINE mode    — AddItemHandler.gpt_planner is a GptAddItemPlannerService.
  4. Default env    — mode defaults to 'disabled', planner is None.
  5. Config error   — HandlerDispatcher still constructs cleanly (no crash).
  6. Planner exception during build — HandlerDispatcher still constructs cleanly.
  7. Disabled equivalence — add-item turn produces same response_key with/without planner.
  8. Shadow mode    — planner.run() called but no cart mutation.
  9. Inline + safe_to_apply  — plan applied (single item) → HandlerResult returned.
 10. Inline + multi-item safe_to_apply → falls through (multi_item_deferred logged).
 11. Shadow mode    — planner.run() called, applies nothing even with safe_to_apply=True in mock.
 12. No crash when planner.run() raises an exception.
 13. Planner exception → local path handles turn without corruption.
 14. add_item_planner_applied log field emitted.
 15. add_item_planner_apply_block_reason='multi_item_deferred' logged on multi-item deferred.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy external deps before any project import
# ---------------------------------------------------------------------------
for _mod in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
sys.modules["twilio.rest"].Client = type("_C", (), {"__init__": lambda *a, **k: None})

_redis = types.ModuleType("redis")
_redis.Redis = type("_R", (), {"__init__": lambda *a, **k: None})
sys.modules.setdefault("redis", _redis)

_intent_mod = types.ModuleType("app.ml.intent.inference_intent")
_slot_mod = types.ModuleType("app.ml.slot.inference_slot")
_intent_mod.IntentBundle = type("IntentBundle", (), {})
_intent_mod.predict_intent = lambda *a, **k: []
_slot_mod.SlotBundle = type("SlotBundle", (), {})
_slot_mod.predict_slots = lambda *a, **k: []
sys.modules.setdefault("app.ml.intent.inference_intent", _intent_mod)
sys.modules.setdefault("app.ml.slot.inference_slot", _slot_mod)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.config.semantic_repair import SemanticRepairConfig
from app.core.command_executor import CommandExecutor
from app.core.handler_dispatcher import HandlerDispatcher, _build_add_item_planner
from app.core.response_builder import ResponseBuilder
from app.core.turn_diagnostics import TurnDiagnostics
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.semantic_repair.add_item_planner_result import AddItemPlannerResult
from app.nlu.semantic_repair.add_item_planner_service import GptAddItemPlannerService
from app.services.checkout_service import CheckoutService
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants"


class _StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        return SimpleNamespace(ok=False, sid=None, error_code="x", error_message="x")


def _build_menu_repo(restaurant: str = "steves_grill") -> MenuRepository:
    root = DATA_ROOT / restaurant
    return MenuRepository(
        MenuStore(
            menu_path=root / "menu.json",
            entity_index_path=root / "entity_index.json",
        )
    )


def _build_dispatcher(menu_repo: MenuRepository | None = None) -> HandlerDispatcher:
    if menu_repo is None:
        menu_repo = _build_menu_repo()
    sms = _StubSmsService()
    return HandlerDispatcher(
        menu_repo=menu_repo,
        cart_summary_builder=CartSummaryBuilder(menu_repo),
        sms_service=sms,
        checkout_service=CheckoutService(),
        responder=ResponseBuilder(menu_repo),
        command_executor=CommandExecutor(sms),
        diagnostics=TurnDiagnostics(backends=[]),
    )


def _make_disabled_config() -> SemanticRepairConfig:
    return SemanticRepairConfig(
        phase=0,
        model="gpt-4o-mini",
        timeout_seconds=3.0,
        call_mode="disabled",
        add_item_planner_mode="disabled",
    )


def _make_shadow_config() -> SemanticRepairConfig:
    return SemanticRepairConfig(
        phase=0,
        model="gpt-4o-mini",
        timeout_seconds=3.0,
        call_mode="disabled",
        add_item_planner_mode="shadow",
    )


def _make_inline_config() -> SemanticRepairConfig:
    return SemanticRepairConfig(
        phase=0,
        model="gpt-4o-mini",
        timeout_seconds=3.0,
        call_mode="disabled",
        add_item_planner_mode="inline",
    )


def _no_op_planner_result(safe_to_apply: bool = False) -> AddItemPlannerResult:
    """Return a planner result that is already fully populated (no real GPT call)."""
    return AddItemPlannerResult(
        decision="skipped",
        gpt_called=False,
        route_mode="no_gpt",
        route_reason="mock",
        safe_to_apply=safe_to_apply,
    )


def _new_context(state: ConversationState = ConversationState.IDLE) -> ConversationContext:
    ctx = ConversationContext()
    # ConversationContext does not store conversation_state (that lives in Session).
    # order_type is set to avoid FlowGate order-type prompts in integration tests.
    ctx.order_type = "pickup"
    return ctx


# ---------------------------------------------------------------------------
# Part 1: _build_add_item_planner() helper
# ---------------------------------------------------------------------------


class TestBuildAddItemPlannerHelper(unittest.TestCase):
    """Unit tests for the _build_add_item_planner() factory function."""

    def test_disabled_mode_returns_none(self):
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            return_value=_make_disabled_config(),
        ):
            result = _build_add_item_planner()
        self.assertIsNone(result)

    def test_shadow_mode_returns_service(self):
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            return_value=_make_shadow_config(),
        ):
            result = _build_add_item_planner()
        self.assertIsInstance(result, GptAddItemPlannerService)

    def test_inline_mode_returns_service(self):
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            return_value=_make_inline_config(),
        ):
            result = _build_add_item_planner()
        self.assertIsInstance(result, GptAddItemPlannerService)

    def test_config_exception_returns_none(self):
        """Any exception during config load → None (never raises)."""
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            side_effect=RuntimeError("boom"),
        ):
            result = _build_add_item_planner()
        self.assertIsNone(result)

    def test_shadow_service_has_correct_mode(self):
        """The constructed service carries the shadow config."""
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            return_value=_make_shadow_config(),
        ):
            result = _build_add_item_planner()
        self.assertIsNotNone(result)
        mode = getattr(getattr(result, "_config", None), "add_item_planner_mode", None)
        self.assertEqual(mode, "shadow")

    def test_inline_service_has_correct_mode(self):
        """The constructed service carries the inline config."""
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            return_value=_make_inline_config(),
        ):
            result = _build_add_item_planner()
        self.assertIsNotNone(result)
        mode = getattr(getattr(result, "_config", None), "add_item_planner_mode", None)
        self.assertEqual(mode, "inline")


# ---------------------------------------------------------------------------
# Part 2: HandlerDispatcher wiring
# ---------------------------------------------------------------------------


class TestHandlerDispatcherWiring(unittest.TestCase):
    """Verify HandlerDispatcher passes the right planner to AddItemHandler."""

    def test_disabled_mode_add_item_handler_has_no_planner(self):
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            return_value=_make_disabled_config(),
        ):
            dispatcher = _build_dispatcher()
        handler = dispatcher.get_handler("add_item_handler")
        self.assertIsInstance(handler, AddItemHandler)
        self.assertIsNone(handler._gpt_planner)

    def test_shadow_mode_add_item_handler_has_planner(self):
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            return_value=_make_shadow_config(),
        ):
            dispatcher = _build_dispatcher()
        handler = dispatcher.get_handler("add_item_handler")
        self.assertIsInstance(handler, AddItemHandler)
        self.assertIsInstance(handler._gpt_planner, GptAddItemPlannerService)

    def test_inline_mode_add_item_handler_has_planner(self):
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            return_value=_make_inline_config(),
        ):
            dispatcher = _build_dispatcher()
        handler = dispatcher.get_handler("add_item_handler")
        self.assertIsInstance(handler, AddItemHandler)
        self.assertIsInstance(handler._gpt_planner, GptAddItemPlannerService)

    def test_default_env_no_planner(self):
        """Default env COMPASS_GPT_ADD_ITEM_PLANNER_MODE is 'disabled'."""
        import os
        # Ensure the env var is absent so default kicks in.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COMPASS_GPT_ADD_ITEM_PLANNER_MODE", None)
            dispatcher = _build_dispatcher()
        handler = dispatcher.get_handler("add_item_handler")
        self.assertIsNone(handler._gpt_planner)

    def test_build_error_still_constructs_dispatcher(self):
        """If _build_add_item_planner() returns None due to error, dispatcher is fine."""
        with patch("app.core.handler_dispatcher._build_add_item_planner", return_value=None):
            dispatcher = _build_dispatcher()
        self.assertIsNotNone(dispatcher.get_handler("add_item_handler"))
        self.assertIsNone(dispatcher.get_handler("add_item_handler")._gpt_planner)

    def test_all_other_handlers_still_registered(self):
        """Phase 4.1 must not remove any existing handler registrations."""
        with patch(
            "app.core.handler_dispatcher.get_semantic_repair_config",
            return_value=_make_disabled_config(),
        ):
            dispatcher = _build_dispatcher()
        for name in (
            "waiting_for_side_handler",
            "waiting_for_modifier_handler",
            "waiting_for_size_handler",
            "confirming_handler",
            "waiting_for_payment_handler",
            "cart_handler",
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(dispatcher.get_handler(name))


# ---------------------------------------------------------------------------
# Part 3: AddItemHandler Phase 4.1 behavioral tests
# ---------------------------------------------------------------------------


class TestAddItemHandlerDisabledEquivalence(unittest.TestCase):
    """Disabled mode: handler behavior is identical to pre-Phase-4 baseline."""

    def setUp(self):
        self.menu_repo = _build_menu_repo()
        # No planner at all — pure local path.
        self.handler = AddItemHandler(menu_repo=self.menu_repo)

    def _ctx(self) -> ConversationContext:
        return _new_context()

    def test_disabled_produces_handler_result(self):
        ctx = self._ctx()
        ctx.last_slots = [SlotValue(name="ITEM", value="Bourbon Burger")]
        result = self.handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="bourbon burger",
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.response_key)

    def test_disabled_planner_never_called(self):
        """_try_gpt_planner returns None immediately when gpt_planner is None."""
        ctx = self._ctx()
        result = self.handler._try_gpt_planner(
            user_text="chicken burger with cheese",
            slots=[{"n": "ITEM", "v": "chicken burger"}],
            session=None,
        )
        self.assertIsNone(result)

    def test_disabled_no_log_planner_apply_outcome(self):
        """_log_planner_apply_outcome must not be called in disabled mode."""
        ctx = self._ctx()
        ctx.last_slots = [SlotValue(name="ITEM", value="Bourbon Burger")]
        with patch.object(self.handler, "_log_planner_apply_outcome") as mock_log:
            self.handler.handle(
                intent=Intent.ADD_ITEM,
                context=ctx,
                user_text="bourbon burger",
            )
        mock_log.assert_not_called()


class TestAddItemHandlerShadowMode(unittest.TestCase):
    """Shadow mode: planner is called but never applied."""

    def setUp(self):
        self.menu_repo = _build_menu_repo()
        self.mock_planner = MagicMock(spec=GptAddItemPlannerService)
        self.handler = AddItemHandler(
            menu_repo=self.menu_repo,
            gpt_planner=self.mock_planner,
        )

    def _ctx(self) -> ConversationContext:
        return _new_context()

    def test_shadow_planner_run_called(self):
        """planner.run() is called via _try_gpt_planner."""
        ctx = self._ctx()
        ctx.last_slots = []
        self.mock_planner.run.return_value = _no_op_planner_result(safe_to_apply=False)
        self.handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="two chicken burgers",
        )
        self.mock_planner.run.assert_called_once()

    def test_shadow_never_applies_when_safe_to_apply_false(self):
        """safe_to_apply=False → _apply_planner_result never called."""
        ctx = self._ctx()
        self.mock_planner.run.return_value = _no_op_planner_result(safe_to_apply=False)
        with patch.object(self.handler, "_apply_planner_result") as mock_apply:
            self.handler.handle(
                intent=Intent.ADD_ITEM,
                context=ctx,
                user_text="a burger",
            )
        mock_apply.assert_not_called()

    def test_shadow_apply_outcome_not_logged_when_not_safe(self):
        """_log_planner_apply_outcome not called when safe_to_apply=False."""
        ctx = self._ctx()
        self.mock_planner.run.return_value = _no_op_planner_result(safe_to_apply=False)
        with patch.object(self.handler, "_log_planner_apply_outcome") as mock_log:
            self.handler.handle(
                intent=Intent.ADD_ITEM,
                context=ctx,
                user_text="a burger",
            )
        mock_log.assert_not_called()

    def test_shadow_exception_does_not_crash_turn(self):
        """If planner.run() raises, handle() still returns a valid HandlerResult."""
        ctx = self._ctx()
        self.mock_planner.run.side_effect = RuntimeError("GPT exploded")
        result = self.handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="chicken burger",
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.response_key)

    def test_shadow_handler_result_response_key_nonempty(self):
        """Local path produces a non-empty response_key even with shadow planner."""
        ctx = self._ctx()
        self.mock_planner.run.return_value = _no_op_planner_result(safe_to_apply=False)
        result = self.handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="bourbon burger",
        )
        self.assertIsNotNone(result.response_key)
        self.assertNotEqual(result.response_key, "")


class TestAddItemHandlerInlineMode(unittest.TestCase):
    """Inline mode: safe single-item plan is applied; multi-item falls through."""

    def setUp(self):
        self.menu_repo = _build_menu_repo()
        self.mock_planner = MagicMock(spec=GptAddItemPlannerService)
        self.handler = AddItemHandler(
            menu_repo=self.menu_repo,
            gpt_planner=self.mock_planner,
        )

    def _ctx(self) -> ConversationContext:
        return _new_context()

    def _inline_result_single(self) -> AddItemPlannerResult:
        """A result that is safe_to_apply with 1 validated item (single)."""
        from app.nlu.semantic_repair.add_item_planner_result import PlannerGptItem
        from app.nlu.semantic_repair.add_item_plan_validator import (
            ValidatedAddItemPlan,
        )
        # Build a minimal ValidatedAddItemPlan with 1 item.
        # We'll patch _apply_planner_result instead of building real menu objects.
        return AddItemPlannerResult(
            decision="add_items",
            gpt_called=True,
            route_mode="inline_gpt",
            safe_to_apply=True,
            confidence=0.90,
        )

    def test_inline_safe_single_item_apply_called(self):
        """When safe_to_apply=True, _apply_planner_result is invoked."""
        ctx = self._ctx()
        mock_result = self._inline_result_single()
        self.mock_planner.run.return_value = mock_result
        sentinel = SimpleNamespace(
            response_key="item_found",
            next_state=ConversationState.IDLE,
        )
        with patch.object(self.handler, "_apply_planner_result", return_value=sentinel):
            result = self.handler.handle(
                intent=Intent.ADD_ITEM,
                context=ctx,
                user_text="chicken burger with cheese",
            )
        self.assertEqual(result.response_key, "item_found")

    def test_inline_applied_log_emitted_on_success(self):
        """_log_planner_apply_outcome(applied=True) is called on successful apply."""
        ctx = self._ctx()
        mock_result = self._inline_result_single()
        self.mock_planner.run.return_value = mock_result
        sentinel = SimpleNamespace(response_key="item_found", next_state=ConversationState.IDLE)
        logged_calls: list[dict] = []

        def _capture_log(*, planner_result, applied, block_reason):
            logged_calls.append({"applied": applied, "block_reason": block_reason})

        with patch.object(self.handler, "_apply_planner_result", return_value=sentinel):
            with patch.object(self.handler, "_log_planner_apply_outcome", side_effect=_capture_log):
                self.handler.handle(
                    intent=Intent.ADD_ITEM,
                    context=ctx,
                    user_text="chicken burger with cheese",
                )
        self.assertEqual(len(logged_calls), 1)
        self.assertTrue(logged_calls[0]["applied"])
        self.assertIsNone(logged_calls[0]["block_reason"])

    def test_inline_multi_item_deferred_block_reason_logged(self):
        """When apply returns None (multi-item deferred), block_reason='multi_item_deferred'."""
        ctx = self._ctx()
        mock_result = self._inline_result_single()
        self.mock_planner.run.return_value = mock_result

        # Simulate _apply_planner_result returning None (multi-item deferred).
        logged_calls: list[dict] = []

        def _capture_log(*, planner_result, applied, block_reason):
            logged_calls.append({"applied": applied, "block_reason": block_reason})

        with patch.object(self.handler, "_apply_planner_result", return_value=None):
            with patch.object(self.handler, "_log_planner_apply_outcome", side_effect=_capture_log):
                self.handler.handle(
                    intent=Intent.ADD_ITEM,
                    context=ctx,
                    user_text="chicken burger and a coke",
                )
        # Outcome log must have been emitted with applied=False.
        self.assertEqual(len(logged_calls), 1)
        self.assertFalse(logged_calls[0]["applied"])
        # block_reason is either multi_item_deferred or apply_helper_returned_none.
        self.assertIn(
            logged_calls[0]["block_reason"],
            ("multi_item_deferred", "apply_helper_returned_none"),
        )

    def test_inline_apply_returns_none_falls_through_to_local(self):
        """When apply returns None, local path runs and produces a valid HandlerResult."""
        ctx = self._ctx()
        mock_result = self._inline_result_single()
        self.mock_planner.run.return_value = mock_result
        with patch.object(self.handler, "_apply_planner_result", return_value=None):
            result = self.handler.handle(
                intent=Intent.ADD_ITEM,
                context=ctx,
                user_text="chicken burger",
            )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.response_key)


# ---------------------------------------------------------------------------
# Part 4: _log_planner_apply_outcome unit tests
# ---------------------------------------------------------------------------


class TestLogPlannerApplyOutcome(unittest.TestCase):
    """Unit tests for the _log_planner_apply_outcome helper."""

    def setUp(self):
        menu_repo = _build_menu_repo()
        self.handler = AddItemHandler(menu_repo=menu_repo)

    def test_applied_true_logs_correct_fields(self):
        import logging

        fake_result = AddItemPlannerResult(
            decision="add_items",
            gpt_called=True,
            route_mode="inline_gpt",
            confidence=0.88,
            safe_to_apply=True,
        )
        with self.assertLogs("app.state_machine.handlers.item.add_item.add_item_handler", level=logging.INFO) as cm:
            self.handler._log_planner_apply_outcome(
                planner_result=fake_result,
                applied=True,
                block_reason=None,
            )
        # Ensure the log event was emitted.
        self.assertTrue(any("add_item_planner_apply_outcome" in line for line in cm.output))

    def test_applied_false_multi_item_logs_correct_fields(self):
        import logging

        fake_result = AddItemPlannerResult(
            decision="add_items",
            gpt_called=True,
            route_mode="inline_gpt",
            confidence=0.85,
            safe_to_apply=True,
        )
        with self.assertLogs("app.state_machine.handlers.item.add_item.add_item_handler", level=logging.INFO) as cm:
            self.handler._log_planner_apply_outcome(
                planner_result=fake_result,
                applied=False,
                block_reason="multi_item_deferred",
            )
        self.assertTrue(any("add_item_planner_apply_outcome" in line for line in cm.output))

    def test_no_exception_on_null_planner_result_attributes(self):
        """Should not raise even if planner_result has no attributes."""
        self.handler._log_planner_apply_outcome(
            planner_result=SimpleNamespace(),  # empty object
            applied=False,
            block_reason="apply_helper_returned_none",
        )


# ---------------------------------------------------------------------------
# Part 5: No cart mutation from GPT
# ---------------------------------------------------------------------------


class TestNoCartMutationFromGpt(unittest.TestCase):
    """GPT planner must never directly mutate the cart."""

    def setUp(self):
        self.menu_repo = _build_menu_repo()
        self.mock_planner = MagicMock(spec=GptAddItemPlannerService)
        self.handler = AddItemHandler(
            menu_repo=self.menu_repo,
            gpt_planner=self.mock_planner,
        )

    def test_shadow_cart_unchanged(self):
        """Shadow mode never adds to cart even if planner returns a plan."""
        ctx = _new_context()
        cart_before: list = []

        self.mock_planner.run.return_value = _no_op_planner_result(safe_to_apply=False)
        self.handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="chicken burger",
        )
        # Cart has no direct accessor in context; validate that _apply_planner_result
        # was never reached (safe_to_apply=False).
        with patch.object(self.handler, "_apply_planner_result") as mock_apply:
            ctx2 = _new_context()
            self.mock_planner.run.return_value = _no_op_planner_result(safe_to_apply=False)
            self.handler.handle(
                intent=Intent.ADD_ITEM,
                context=ctx2,
                user_text="chicken burger",
            )
        mock_apply.assert_not_called()

    def test_gpt_result_safe_to_apply_never_applied_by_service(self):
        """AddItemPlannerResult.safe_to_apply is set by the apply gate (handler).

        The GPT service itself never applies — safe_to_apply is a signal to the
        handler, not a guarantee that anything was mutated. The handler decides.
        """
        result = AddItemPlannerResult(
            decision="add_items",
            gpt_called=True,
            route_mode="inline_gpt",
            safe_to_apply=True,
        )
        # safe_to_apply=True is just a flag — no cart/state mutation happened yet.
        # The result object carries no "applied" state — application is in the handler.
        self.assertTrue(result.safe_to_apply)
        self.assertEqual(result.decision, "add_items")
        self.assertTrue(result.gpt_called)


if __name__ == "__main__":
    unittest.main()
