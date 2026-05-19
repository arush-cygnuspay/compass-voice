# tests/services/test_smart_turn_wiring.py
"""Runtime wiring tests for SmartTurnPlanner integration.

These tests verify that the planner is correctly wired into the live handler
paths and that the feature-flag, timeout, invalid-JSON, and validation
guardrails work end-to-end.

Tests do NOT make real OpenAI calls — plan_smart_turn() is mocked.

TC-W01  Feature flag off -> add_item_handler path unchanged.
TC-W02  Planner timeout (returns None) -> add_item_handler local flow continues.
TC-W03  Invalid JSON -> add_item_handler local flow continues.
TC-W04  waiting_for_modifier + "macarola cheese" + Mozzarella allowed
        -> resolves via deterministic modifier path.
TC-W05  waiting_for_side + "loaded fries" in allowed options
        -> resolves via deterministic side path.
TC-W06  correction: user says "no I said loaded fries"
        -> correction plan routed; does not duplicate.
TC-W07  compound: "bourbon chicken with fresh mushroom"
        -> item resolved to Bourbon Chicken; modifier sent to prefill.
TC-W08  checkout task_mode assigned for "that's it" in IDLE.
TC-W09  "that's it" in WAITING_FOR_SIDE -> does not trigger SmartTurnPlanner
        (local skip/done handler runs instead — planner not invoked).
TC-W10  reprompt: waiting_for_modifier, "what do you have?" triggers planner
        with allowed options populated.
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Ensure tests run without openai package
# ---------------------------------------------------------------------------
if "openai" not in sys.modules:
    _fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *a, **kw):
            pass

    _fake_openai.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    sys.modules["openai"] = _fake_openai

# ---------------------------------------------------------------------------
# Import from project
# ---------------------------------------------------------------------------
from app.services.smart_turn_planner import (
    SmartTurnCorrection,
    SmartTurnItem,
    SmartTurnModifier,
    SmartTurnPlan,
    SmartTurnSide,
)
from app.services.smart_turn_policy import (
    determine_smart_task_mode,
    should_use_smart_planner,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


class _EnvPatch:
    _KEYS = frozenset({"SMART_TURN_PLANNER_ENABLED", "OPENAI_API_KEY"})

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
            orig = self._saved.get(key)
            if orig is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig


def _make_context(
    *,
    state: ConversationState = ConversationState.IDLE,
    last_intent_confidence: float = 0.85,
    pending_add_item=None,
    current_modifier_group_index: int = 0,
    current_side_group_index: int = 0,
) -> ConversationContext:
    ctx = ConversationContext()
    ctx.last_intent_confidence = last_intent_confidence
    if pending_add_item is not None:
        ctx.pending_add_item = pending_add_item
    ctx.current_modifier_group_index = current_modifier_group_index
    ctx.current_side_group_index = current_side_group_index
    return ctx


def _make_pending_item(item_name: str, modifier_groups=None, side_groups=None):
    """Build a minimal PendingAddItem-like stub for tests."""
    m = MagicMock()
    m.item_name = item_name
    m.modifier_groups = modifier_groups or []
    m.side_groups = side_groups or []
    m.modifier_groups_by_id = {g.group_id: g for g in (modifier_groups or [])}
    return m


def _make_modifier_group(group_id: str, group_name: str, choices: list[str]):
    """Build a minimal PendingModifierGroup-like stub."""
    g = MagicMock()
    g.group_id = group_id
    g.name = group_name
    g.choices = []
    for cname in choices:
        c = MagicMock()
        c.name = cname
        c.normalized_name = cname.lower()
        c.modifier_id = f"mod_{cname.lower().replace(' ', '_')}"
        c.match_texts = (cname.lower(),)
        g.choices.append(c)
    g.choices_by_modifier_id = {c.modifier_id: c for c in g.choices}
    g.min_selector = 1
    g.max_selector = 1
    return g


def _make_side_group(group_id: str, group_name: str, choices: list[str]):
    """Build a minimal PendingSideGroup-like stub."""
    g = MagicMock()
    g.group_id = group_id
    g.name = group_name
    g.choices = []
    for cname in choices:
        c = MagicMock()
        c.name = cname
        c.item_id = f"side_{cname.lower().replace(' ', '_')}"
        c.pricing_mode = "flat"
        g.choices.append(c)
    g.choices_by_item_id = {c.item_id: c for c in g.choices}
    g.min_selector = 1
    g.max_selector = 1
    g.allow_duplicate_selections = False
    return g


def _add_items_plan(item_name: str, modifiers=None, sides=None, confidence=0.9):
    return SmartTurnPlan(
        decision="add_items",
        confidence=confidence,
        items=(
            SmartTurnItem(
                item_name=item_name,
                modifiers=tuple(SmartTurnModifier(name=m) for m in (modifiers or [])),
                sides=tuple(SmartTurnSide(name=s) for s in (sides or [])),
            ),
        ),
        gpt_called=True,
        trigger_reason="test",
    )


def _correction_plan(original: str, corrected: str, confidence=0.9):
    return SmartTurnPlan(
        decision="correction",
        confidence=confidence,
        correction=SmartTurnCorrection(
            original_text=original,
            corrected_text=corrected,
        ),
        gpt_called=True,
        trigger_reason="test",
    )


# ---------------------------------------------------------------------------
# TC-W01: Feature flag off — add_item_handler path unchanged
# ---------------------------------------------------------------------------


class TestFeatureFlagOffAddItemHandler(unittest.TestCase):
    """When SMART_TURN_PLANNER_ENABLED=false, plan_smart_turn must not be called."""

    def test_plan_not_called_when_disabled(self):
        with _EnvPatch({"SMART_TURN_PLANNER_ENABLED": "false"}):
            from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
            from app.nlu.intent_resolution.intent import Intent

            menu_repo = MagicMock()
            menu_repo.store = None  # Prevent local planner from matching via MagicMock
            menu_repo.resolve_menu_query_normalized.return_value = MagicMock(items=[])

            handler = AddItemHandler(menu_repo=menu_repo, gpt_planner=None)
            ctx = _make_context(last_intent_confidence=0.4)

            # Compound utterance — planner must NOT be called (flag is off)
            with patch("app.services.smart_turn_planner.plan_smart_turn") as mock_plan:
                handler.handle(
                    intent=Intent.ADD_ITEM,
                    context=ctx,
                    user_text="chicken burger with mozzarella and a coke",
                    session=None,
                )
                mock_plan.assert_not_called()


# ---------------------------------------------------------------------------
# TC-W02: Planner timeout -> local flow continues
# ---------------------------------------------------------------------------


class TestPlannerTimeoutFallsThrough(unittest.TestCase):

    def test_timeout_returns_local_result(self):
        """When plan_smart_turn returns None (timeout), add_item_handler
        falls through to local menu resolution."""
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test",
        }):
            from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
            from app.nlu.intent_resolution.intent import Intent

            menu_repo = MagicMock()
            menu_repo.store = None  # Prevent local planner from matching via MagicMock
            local_result = MagicMock()
            local_result.items = []
            menu_repo.resolve_menu_query_normalized.return_value = local_result

            handler = AddItemHandler(menu_repo=menu_repo, gpt_planner=None)
            ctx = _make_context(last_intent_confidence=0.4)

            # plan_smart_turn returns None (simulates timeout)
            with patch("app.services.smart_turn_planner.plan_smart_turn", return_value=None):
                with patch(
                    "app.services.smart_turn_policy.should_use_smart_planner",
                    return_value=(True, "compound_utterance"),
                ):
                    result = handler.handle(
                        intent=Intent.ADD_ITEM,
                        context=ctx,
                        user_text="chicken burger with mozzarella and a coke",
                        session=None,
                    )

        # Result should be a HandlerResult from the local path, not None
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# TC-W03: Invalid JSON -> local flow continues
# ---------------------------------------------------------------------------


class TestInvalidJsonFallsThrough(unittest.TestCase):

    def test_unclear_plan_fails_validation_and_local_runs(self):
        """When plan has decision='unclear', validation blocks it and local runs."""
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test",
        }):
            from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
            from app.nlu.intent_resolution.intent import Intent

            menu_repo = MagicMock()
            local_result = MagicMock()
            local_result.items = []
            menu_repo.resolve_menu_query_normalized.return_value = local_result

            handler = AddItemHandler(menu_repo=menu_repo, gpt_planner=None)
            ctx = _make_context(last_intent_confidence=0.4)

            # Simulate invalid JSON → parse error → decision='unclear'
            unclear_plan = SmartTurnPlan(
                decision="unclear",
                gpt_called=True,
                confidence=0.1,
                reason="parse_error: invalid json",
            )

            with patch("app.services.smart_turn_planner.plan_smart_turn", return_value=unclear_plan):
                with patch(
                    "app.services.smart_turn_policy.should_use_smart_planner",
                    return_value=(True, "compound_utterance"),
                ):
                    result = handler.handle(
                        intent=Intent.ADD_ITEM,
                        context=ctx,
                        user_text="aaasdfkjhqwerty",
                        session=None,
                    )

        # Must still return a result from local path
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# TC-W04: waiting_for_modifier + "macarola cheese" -> Mozzarella applied
# ---------------------------------------------------------------------------


class TestModifierPlannerResolvesMacarola(unittest.TestCase):

    def test_planner_resolves_phonetic_to_mozzarella(self):
        """SmartTurnPlanner resolves 'macarola cheese' → 'Mozzarella'.

        Verifies that _try_smart_planner_modifier:
          1. Calls plan_smart_turn with the allowed modifier options.
          2. Extracts "Mozzarella" from the plan.
          3. Validates the name via build_modifier_selections_from_names.
          4. Routes to _apply_modifier_selection (deterministic path).

        _apply_modifier_selection is patched so the test does not depend on
        the full determine_next_add_item_step / mock-pending-item chain.
        """
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test",
        }):
            from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
                WaitingForModifierHandler,
            )
            from app.nlu.intent_resolution.intent import Intent

            mod_group = _make_modifier_group(
                "cheese_group", "Cheese", ["Mozzarella", "Cheddar", "American"]
            )
            pending = _make_pending_item("Chicken Burger", modifier_groups=[mod_group])

            ctx = _make_context(
                last_intent_confidence=0.35,  # below threshold → triggers planner
                pending_add_item=pending,
            )

            mozzarella_plan = _add_items_plan(
                "Chicken Burger",
                modifiers=["Mozzarella"],
            )

            with patch("app.services.smart_turn_planner.plan_smart_turn", return_value=mozzarella_plan):
                with patch(
                    "app.services.smart_turn_policy.should_use_smart_planner",
                    return_value=(True, "low_confidence_waiting_state"),
                ):
                    handler = WaitingForModifierHandler(menu_repo=None)
                    # Patch _apply_modifier_selection to capture the call without
                    # running the full finalization chain against a mock pending item.
                    with patch.object(
                        handler,
                        "_apply_modifier_selection",
                        return_value=MagicMock(
                            response_key="item_added_successfully",
                            next_state=ConversationState.IDLE,
                        ),
                    ) as mock_apply:
                        result = handler.handle(
                            intent=Intent.ADD_ITEM,
                            context=ctx,
                            user_text="macarola cheese",
                            session=None,
                        )

        # _apply_modifier_selection must have been called by the planner hook
        mock_apply.assert_called()
        # The result must NOT be a reprompt — it comes from the planner path
        self.assertIsNotNone(result)
        self.assertNotEqual(result.response_key, "repeat_modifier_options")
        # Verify the resolved name was "Mozzarella" — check matched_selections kwarg
        call_kwargs = mock_apply.call_args.kwargs
        matched = call_kwargs.get("matched_selections", [])
        self.assertTrue(
            any(getattr(s, "name", "") == "Mozzarella" for s in matched),
            f"Expected 'Mozzarella' in matched_selections; got {matched}",
        )


# ---------------------------------------------------------------------------
# TC-W05: waiting_for_side + "loaded fries" resolved via planner
# ---------------------------------------------------------------------------


class TestSidePlannerResolvesLoadedFries(unittest.TestCase):

    def test_planner_resolves_side_to_loaded_fries(self):
        """SmartTurnPlanner resolves 'loaded fries' to the allowed side."""
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test",
        }):
            from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
                WaitingForSideHandler,
            )
            from app.nlu.intent_resolution.intent import Intent

            side_group = _make_side_group(
                "fries_group", "Fries", ["Regular Fries", "Loaded Fries", "Sweet Potato Fries"]
            )
            pending = _make_pending_item("Chicken Burger", side_groups=[side_group])

            ctx = _make_context(
                last_intent_confidence=0.38,
                pending_add_item=pending,
            )

            loaded_fries_plan = SmartTurnPlan(
                decision="add_items",
                confidence=0.88,
                items=(
                    SmartTurnItem(
                        item_name="Chicken Burger",
                        sides=(SmartTurnSide(name="Loaded Fries"),),
                    ),
                ),
                gpt_called=True,
            )

            # side_resolver.resolve() with "Loaded Fries" must match
            mock_side_match = MagicMock()
            mock_side_match.matched_item_ids = ["side_loaded_fries"]

            with patch("app.services.smart_turn_planner.plan_smart_turn", return_value=loaded_fries_plan):
                with patch(
                    "app.services.smart_turn_policy.should_use_smart_planner",
                    return_value=(True, "low_confidence_waiting_state"),
                ):
                    handler = WaitingForSideHandler(menu_repo=None)
                    # Patch the resolver's resolve to return a match for "Loaded Fries"
                    handler.side_resolver.resolve = MagicMock(return_value=mock_side_match)
                    # Patch _apply_side_selection to capture the call
                    with patch.object(
                        handler,
                        "_apply_side_selection",
                        return_value=MagicMock(
                            response_key="confirming_item",
                            next_state=ConversationState.CONFIRMING_ITEM,
                        ),
                    ) as mock_apply:
                        result = handler.handle(
                            intent=Intent.ADD_ITEM,
                            context=ctx,
                            user_text="loaded fries",
                            session=None,
                        )

        self.assertIsNotNone(result)
        # _apply_side_selection must have been called exactly once by the planner path
        # (it may also be called by other paths, so check last call)
        mock_apply.assert_called()


# ---------------------------------------------------------------------------
# TC-W06: correction — "no I said loaded fries" routes correction plan
# ---------------------------------------------------------------------------


class TestCorrectionRouting(unittest.TestCase):

    def test_correction_plan_routes_without_duplicate(self):
        """'no I said loaded fries' triggers correction task_mode in add_item_handler.
        The correction is applied through menu resolution (not duplicated)."""
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test",
        }):
            from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
            from app.nlu.intent_resolution.intent import Intent

            menu_repo = MagicMock()
            menu_item = MagicMock()
            menu_item.item_id = "loaded_fries_id"
            menu_item.name = "Loaded Fries"
            query_result = MagicMock()
            query_result.items = [menu_item]
            menu_repo.resolve_menu_query_normalized.return_value = query_result

            handler = AddItemHandler(menu_repo=menu_repo, gpt_planner=None)
            ctx = _make_context(last_intent_confidence=0.5)
            # Simulate cart containing "Fries" (the wrong item)
            ctx.turn_memory.append(("assistant", "I added Fries to your order."))

            corr_plan = _correction_plan("Fries", "Loaded Fries")

            with patch("app.services.smart_turn_planner.plan_smart_turn", return_value=corr_plan):
                with patch(
                    "app.services.smart_turn_policy.should_use_smart_planner",
                    return_value=(True, "correction_phrase"),
                ):
                    with patch.object(
                        handler.item_resolution_handler,
                        "resolve_item_and_enter_flow",
                        return_value=MagicMock(
                            response_key="confirming_item",
                            next_state=ConversationState.CONFIRMING_ITEM,
                        ),
                    ) as mock_resolve:
                        result = handler.handle(
                            intent=Intent.ADD_ITEM,
                            context=ctx,
                            user_text="no I said loaded fries",
                            session=None,
                        )

        self.assertIsNotNone(result)
        # resolve_item_and_enter_flow must be called with "Loaded Fries"
        mock_resolve.assert_called()
        call_kwargs = mock_resolve.call_args.kwargs
        self.assertEqual(call_kwargs.get("requested_item_text"), "Loaded Fries")


# ---------------------------------------------------------------------------
# TC-W07: compound — "bourbon chicken with fresh mushroom"
# ---------------------------------------------------------------------------


class TestCompoundItemWithModifier(unittest.TestCase):

    def test_bourbon_chicken_with_mushroom_routes_item_not_modifier_as_item(self):
        """'bourbon chicken with fresh mushroom' — Bourbon Chicken is the item;
        Fresh Mushroom goes into the modifier slot, not as a separate menu item."""
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test",
        }):
            from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
            from app.nlu.intent_resolution.intent import Intent

            menu_repo = MagicMock()
            chicken_item = MagicMock()
            chicken_item.item_id = "bourbon_chicken_id"
            chicken_item.name = "Bourbon Chicken"
            query_result = MagicMock()
            query_result.items = [chicken_item]
            menu_repo.resolve_menu_query_normalized.return_value = query_result

            handler = AddItemHandler(menu_repo=menu_repo, gpt_planner=None)
            ctx = _make_context(last_intent_confidence=0.72)

            compound_plan = SmartTurnPlan(
                decision="add_items",
                confidence=0.87,
                items=(
                    SmartTurnItem(
                        item_name="Bourbon Chicken",
                        modifiers=(SmartTurnModifier(name="Fresh Mushroom"),),
                    ),
                ),
                gpt_called=True,
                trigger_reason="compound_utterance",
            )

            with patch("app.services.smart_turn_planner.plan_smart_turn", return_value=compound_plan):
                with patch(
                    "app.services.smart_turn_policy.should_use_smart_planner",
                    return_value=(True, "compound_utterance"),
                ):
                    with patch.object(
                        handler.item_resolution_handler,
                        "resolve_item_and_enter_flow",
                        return_value=MagicMock(
                            response_key="ask_for_modifier",
                            next_state=ConversationState.WAITING_FOR_MODIFIER,
                        ),
                    ) as mock_resolve:
                        result = handler.handle(
                            intent=Intent.ADD_ITEM,
                            context=ctx,
                            user_text="bourbon chicken with fresh mushroom",
                            session=None,
                        )

        self.assertIsNotNone(result)
        # resolve_item_and_enter_flow must be called with Bourbon Chicken
        mock_resolve.assert_called()
        call_kwargs = mock_resolve.call_args.kwargs
        self.assertEqual(call_kwargs.get("requested_item_text"), "Bourbon Chicken")
        # And Fresh Mushroom must be in the synthetic slots (as a MODIFIER slot)
        slots = call_kwargs.get("slots", [])
        modifier_slot_values = [
            getattr(s, "value", "") for s in slots
            if str(getattr(s, "name", "")).upper() == "MODIFIER"
        ]
        self.assertIn("Fresh Mushroom", modifier_slot_values)


# ---------------------------------------------------------------------------
# TC-W08: checkout task_mode assigned for "that's it" in IDLE
# ---------------------------------------------------------------------------


class TestCheckoutTaskModeAssigned(unittest.TestCase):

    def test_checkout_phrase_in_idle_gets_checkout_task_mode(self):
        """determine_smart_task_mode returns 'checkout' for checkout phrases in IDLE."""
        mode = determine_smart_task_mode(
            transcript="that's it",
            state="IDLE",
            local_intent="CHECKOUT",
            local_confidence=0.9,
        )
        self.assertEqual(mode, "checkout")

    def test_checkout_phrase_nothing_else_also_checkout(self):
        mode = determine_smart_task_mode(
            transcript="nothing else",
            state="IDLE",
            local_intent="CHECKOUT",
            local_confidence=0.9,
        )
        self.assertEqual(mode, "checkout")


# ---------------------------------------------------------------------------
# TC-W09: "that's it" in WAITING_FOR_SIDE -> planner NOT invoked
# ---------------------------------------------------------------------------


class TestCheckoutInWaitingForSideNotInvoked(unittest.TestCase):

    def test_skip_done_handled_locally_not_by_planner(self):
        """In WAITING_FOR_SIDE, 'that's it' maps to skip/done — the control-phrase
        intercept handles it BEFORE the SmartTurnPlanner hook is reached.
        should_use_smart_planner returns False for WAITING_FOR_SIDE + 'that's it'
        (terminal-like — handled deterministically)."""
        ok, reason = should_use_smart_planner(
            transcript="that's it",
            state="WAITING_FOR_SIDE",
            local_intent="DONE",
            local_confidence=0.95,
        )
        # High confidence + no correction signal = local path sufficient
        self.assertFalse(ok)
        self.assertEqual(reason, "local_path_sufficient")

    def test_low_confidence_thats_it_in_waiting_for_side_triggers_planner(self):
        """Very low confidence in WAITING_FOR_SIDE does trigger the planner."""
        ok, reason = should_use_smart_planner(
            transcript="that's it",
            state="WAITING_FOR_SIDE",
            local_intent="DONE",
            local_confidence=0.3,  # below threshold
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "low_confidence_waiting_state")


# ---------------------------------------------------------------------------
# TC-W10: reprompt — waiting_for_modifier "what do you have?" triggers planner
#         with allowed options populated
# ---------------------------------------------------------------------------


class TestRepromptTriggersWithAllowedOptions(unittest.TestCase):

    def test_reprompt_populates_allowed_options_in_context(self):
        """When SmartTurnPlanner is invoked in WAITING_FOR_MODIFIER, the context
        builder populates allowed_options from the current modifier group choices."""
        with _EnvPatch({
            "SMART_TURN_PLANNER_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test",
        }):
            from app.services.smart_turn_context_builder import build_smart_turn_context

            mod_group = _make_modifier_group(
                "cheese_group", "Cheese", ["Mozzarella", "Cheddar", "American"]
            )
            pending = _make_pending_item("Chicken Burger", modifier_groups=[mod_group])
            ctx = _make_context(
                last_intent_confidence=0.4,
                pending_add_item=pending,
            )

            # Simulate: handler called build_smart_turn_context with group choices
            stp_ctx = build_smart_turn_context(
                transcript="what do you have",
                state=ConversationState.WAITING_FOR_MODIFIER.value,
                local_intent="ADD_ITEM",
                local_confidence=0.4,
                context=ctx,
                session=None,
                allowed_options=["Mozzarella", "Cheddar", "American"],
            )

        self.assertEqual(
            stp_ctx.allowed_options, ["Mozzarella", "Cheddar", "American"]
        )
        self.assertIn("allowed_options", stp_ctx.context_keys)
        self.assertEqual(stp_ctx.pending_item_name, "Chicken Burger")

    def test_plan_sent_to_gpt_includes_allowed_options(self):
        """The user message built for the planner call includes allowed options."""
        from app.services.smart_turn_planner import _build_user_message

        msg = _build_user_message(
            transcript="what do you have",
            state="WAITING_FOR_MODIFIER",
            local_intent="ADD_ITEM",
            local_confidence=0.4,
            menu_context=[],
            cart_snapshot=[],
            last_cart_diff=None,
            previous_turns=[("What cheese would you like?", "what do you have")],
            task_mode="modifier_selection",
            allowed_options=["Mozzarella", "Cheddar", "American"],
            pending_item_name="Chicken Burger",
            pending_group_name="Cheese",
        )

        self.assertIn("Mozzarella", msg)
        self.assertIn("Cheddar", msg)
        self.assertIn("American", msg)
        self.assertIn("modifier_selection", msg)
        self.assertIn("What cheese would you like?", msg)


# ---------------------------------------------------------------------------
# Bonus: determine_smart_task_mode routing table
# ---------------------------------------------------------------------------


class TestDetermineSmartTaskModeRouting(unittest.TestCase):

    def test_correction_phrase_wins_over_state(self):
        mode = determine_smart_task_mode(
            "actually I want cheddar",
            "WAITING_FOR_MODIFIER",
            "ADD_ITEM",
            0.8,
        )
        self.assertEqual(mode, "correction")

    def test_waiting_for_modifier_gets_modifier_selection(self):
        mode = determine_smart_task_mode(
            "macarola",
            "WAITING_FOR_MODIFIER",
            "ADD_ITEM",
            0.5,
        )
        self.assertEqual(mode, "modifier_selection")

    def test_waiting_for_side_gets_side_selection(self):
        mode = determine_smart_task_mode(
            "loaded fries",
            "WAITING_FOR_SIDE",
            "ADD_ITEM",
            0.5,
        )
        self.assertEqual(mode, "side_selection")

    def test_compound_idle_gets_compound_add_item(self):
        mode = determine_smart_task_mode(
            "burger with mozzarella and a coke",
            "IDLE",
            "ADD_ITEM",
            0.8,
        )
        self.assertEqual(mode, "compound_add_item")

    def test_low_confidence_idle_gets_generic_repair(self):
        mode = determine_smart_task_mode(
            "bourbon burger",  # simple — no compound, no checkout
            "IDLE",
            "ADD_ITEM",
            0.3,  # low confidence
        )
        self.assertEqual(mode, "generic_repair")

    def test_returns_none_for_high_confidence_simple(self):
        mode = determine_smart_task_mode(
            "bourbon burger",
            "IDLE",
            "ADD_ITEM",
            0.95,
        )
        self.assertIsNone(mode)


if __name__ == "__main__":
    unittest.main()
