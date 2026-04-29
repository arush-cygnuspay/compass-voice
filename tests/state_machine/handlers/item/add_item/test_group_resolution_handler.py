# tests/state_machine/handlers/item/add_item/test_group_resolution_handler.py
"""Unit tests for GroupResolutionHandler._step_to_result and _match_debug_payload,
plus subclass identity checks for all three concrete handlers."""
from __future__ import annotations

import types
import unittest

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.add_item_flow import (
    AddItemCommand,
    AddItemNextStep,
    ReadyToFinalize,
)
from app.state_machine.handlers.item.add_item.group_resolution_handler import (
    GroupResolutionHandler,
)
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Minimal concrete subclass (handle() is abstract on BaseHandler)
# ---------------------------------------------------------------------------

class _ConcreteHandler(GroupResolutionHandler):
    def handle(self, intent, context, user_text, session=None):
        return HandlerResult(next_state=ConversationState.IDLE, response_key="noop")


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

def _make_context(*, has_pending: bool = True) -> types.SimpleNamespace:
    pending = None
    if has_pending:
        pending = types.SimpleNamespace(item_name="Zinger Burger")
    return types.SimpleNamespace(
        pending_add_item=pending,
        quantity=1,
    )


def _make_command() -> AddItemCommand:
    return AddItemCommand(
        item_id="item_1",
        quantity=1,
        variant_id=None,
        sides={},
        side_variants={},
        modifiers={},
    )


def _make_ready_to_finalize() -> ReadyToFinalize:
    return ReadyToFinalize(command=_make_command())


def _make_next_step(
    *,
    state: ConversationState = ConversationState.WAITING_FOR_MODIFIER,
    key: str = "ask_modifier",
    payload: dict | None = None,
) -> AddItemNextStep:
    return AddItemNextStep(next_state=state, response_key=key, response_payload=payload)


# ---------------------------------------------------------------------------
# Tests: missing pending context
# ---------------------------------------------------------------------------

class MissingPendingContextTests(unittest.TestCase):
    def setUp(self):
        self.handler = _ConcreteHandler()

    def test_returns_error_recovery_when_pending_is_none(self):
        ctx = _make_context(has_pending=False)
        result = self.handler._step_to_result(ctx, _make_ready_to_finalize())
        self.assertEqual(result.next_state, ConversationState.ERROR_RECOVERY)
        self.assertEqual(result.response_key, "item_context_missing")

    def test_error_recovery_also_for_non_finalize_step(self):
        ctx = _make_context(has_pending=False)
        result = self.handler._step_to_result(ctx, _make_next_step())
        self.assertEqual(result.next_state, ConversationState.ERROR_RECOVERY)
        self.assertEqual(result.response_key, "item_context_missing")


# ---------------------------------------------------------------------------
# Tests: ReadyToFinalize path
# ---------------------------------------------------------------------------

class ReadyToFinalizeTests(unittest.TestCase):
    def setUp(self):
        self.handler = _ConcreteHandler()
        self.ctx = _make_context()

    def test_basic_finalization(self):
        step = _make_ready_to_finalize()
        result = self.handler._step_to_result(self.ctx, step)
        self.assertEqual(result.next_state, ConversationState.IDLE)
        self.assertEqual(result.response_key, "item_added_successfully")
        self.assertTrue(result.reset_context)
        self.assertEqual(result.command, step.command.to_dict())
        self.assertEqual(result.response_payload["item_name"], "Zinger Burger")
        self.assertEqual(result.response_payload["quantity"], 1)

    def test_matched_names_injected(self):
        result = self.handler._step_to_result(
            self.ctx,
            _make_ready_to_finalize(),
            matched_names=["extra bacon"],
        )
        self.assertEqual(result.response_payload["matched_names"], ["extra bacon"])

    def test_unmatched_names_injected(self):
        result = self.handler._step_to_result(
            self.ctx,
            _make_ready_to_finalize(),
            unmatched_names=["mystery topping"],
        )
        self.assertEqual(result.response_payload["unmatched_names"], ["mystery topping"])

    def test_match_debug_merged(self):
        result = self.handler._step_to_result(
            self.ctx,
            _make_ready_to_finalize(),
            match_debug={"scorer": "token", "score": 0.9},
        )
        self.assertEqual(result.response_payload["scorer"], "token")
        self.assertEqual(result.response_payload["score"], 0.9)

    def test_no_matched_names_key_when_empty(self):
        result = self.handler._step_to_result(self.ctx, _make_ready_to_finalize())
        self.assertNotIn("matched_names", result.response_payload)
        self.assertNotIn("unmatched_names", result.response_payload)

    def test_quantity_zero_defaults_to_one(self):
        ctx = _make_context()
        ctx.quantity = 0
        result = self.handler._step_to_result(ctx, _make_ready_to_finalize())
        self.assertEqual(result.response_payload["quantity"], 1)

    def test_quantity_two_preserved(self):
        ctx = _make_context()
        ctx.quantity = 2
        result = self.handler._step_to_result(ctx, _make_ready_to_finalize())
        self.assertEqual(result.response_payload["quantity"], 2)


# ---------------------------------------------------------------------------
# Tests: non-ReadyToFinalize (AddItemNextStep) path
# ---------------------------------------------------------------------------

class NextStepPathTests(unittest.TestCase):
    def setUp(self):
        self.handler = _ConcreteHandler()
        self.ctx = _make_context()

    def test_no_kwargs_preserves_original_payload_reference(self):
        payload = {"group_name": "Sauce"}
        step = _make_next_step(payload=payload)
        result = self.handler._step_to_result(self.ctx, step)
        self.assertIs(result.response_payload, payload)

    def test_no_kwargs_none_payload_stays_none(self):
        step = _make_next_step(payload=None)
        result = self.handler._step_to_result(self.ctx, step)
        self.assertIsNone(result.response_payload)

    def test_state_and_key_forwarded(self):
        step = _make_next_step(
            state=ConversationState.WAITING_FOR_SIDE,
            key="ask_side",
            payload={"a": 1},
        )
        result = self.handler._step_to_result(
            self.ctx, step, matched_names=["fries"]
        )
        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(result.response_key, "ask_side")

    def test_matched_names_injected_into_copy(self):
        orig = {"group_name": "Sauce"}
        step = _make_next_step(payload=orig)
        result = self.handler._step_to_result(
            self.ctx, step, matched_names=["extra bacon"]
        )
        self.assertEqual(result.response_payload["matched_names"], ["extra bacon"])
        # original must not be mutated
        self.assertNotIn("matched_names", orig)

    def test_unmatched_names_injected_into_copy(self):
        orig = {"group_name": "Sauce"}
        step = _make_next_step(payload=orig)
        result = self.handler._step_to_result(
            self.ctx, step, unmatched_names=["mystery"]
        )
        self.assertEqual(result.response_payload["unmatched_names"], ["mystery"])
        self.assertNotIn("unmatched_names", orig)

    def test_match_debug_merged_into_copy(self):
        orig = {"group_name": "Sauce"}
        step = _make_next_step(payload=orig)
        result = self.handler._step_to_result(
            self.ctx, step, match_debug={"scorer": "seq"}
        )
        self.assertEqual(result.response_payload["scorer"], "seq")
        self.assertNotIn("scorer", orig)

    def test_none_payload_with_matched_names_produces_dict(self):
        step = _make_next_step(payload=None)
        result = self.handler._step_to_result(
            self.ctx, step, matched_names=["chips"]
        )
        self.assertEqual(result.response_payload["matched_names"], ["chips"])


# ---------------------------------------------------------------------------
# Tests: _match_debug_payload static helper
# ---------------------------------------------------------------------------

class MatchDebugPayloadTests(unittest.TestCase):
    def test_none_returns_empty_dict(self):
        self.assertEqual(GroupResolutionHandler._match_debug_payload(None), {})

    def test_empty_dict_returns_empty_dict(self):
        self.assertEqual(GroupResolutionHandler._match_debug_payload({}), {})

    def test_populated_dict_returns_shallow_copy(self):
        d = {"key": "value", "score": 0.8}
        result = GroupResolutionHandler._match_debug_payload(d)
        self.assertEqual(result, d)
        self.assertIsNot(result, d)

    def test_mutation_of_result_does_not_affect_original(self):
        d = {"key": "value"}
        result = GroupResolutionHandler._match_debug_payload(d)
        result["extra"] = "injected"
        self.assertNotIn("extra", d)


# ---------------------------------------------------------------------------
# Tests: subclass identity
# ---------------------------------------------------------------------------

class SubclassIdentityTests(unittest.TestCase):
    def test_waiting_for_modifier_handler_is_group_resolution_handler(self):
        from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
            WaitingForModifierHandler,
        )
        self.assertTrue(issubclass(WaitingForModifierHandler, GroupResolutionHandler))

    def test_waiting_for_side_handler_is_group_resolution_handler(self):
        from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
            WaitingForSideHandler,
        )
        self.assertTrue(issubclass(WaitingForSideHandler, GroupResolutionHandler))

    def test_waiting_for_size_handler_is_group_resolution_handler(self):
        from app.state_machine.handlers.item.add_item.waiting_for_size_handler import (
            WaitingForSizeHandler,
        )
        self.assertTrue(issubclass(WaitingForSizeHandler, GroupResolutionHandler))

    def test_handlers_do_not_define_own_step_to_result(self):
        from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
            WaitingForModifierHandler,
        )
        from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
            WaitingForSideHandler,
        )
        from app.state_machine.handlers.item.add_item.waiting_for_size_handler import (
            WaitingForSizeHandler,
        )
        for cls in (WaitingForModifierHandler, WaitingForSideHandler, WaitingForSizeHandler):
            with self.subTest(cls=cls.__name__):
                self.assertNotIn(
                    "_step_to_result",
                    cls.__dict__,
                    f"{cls.__name__} still owns _step_to_result; it should inherit from GroupResolutionHandler.",
                )

    def test_handlers_do_not_define_own_match_debug_payload(self):
        from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
            WaitingForModifierHandler,
        )
        from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
            WaitingForSideHandler,
        )
        for cls in (WaitingForModifierHandler, WaitingForSideHandler):
            with self.subTest(cls=cls.__name__):
                self.assertNotIn(
                    "_match_debug_payload",
                    cls.__dict__,
                    f"{cls.__name__} still owns _match_debug_payload; it should inherit from GroupResolutionHandler.",
                )


if __name__ == "__main__":
    unittest.main()
