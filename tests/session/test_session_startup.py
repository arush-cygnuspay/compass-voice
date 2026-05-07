# tests/session/test_session_startup.py
"""
Verify that the startup flow no longer asks about landline/mobile and
that new sessions begin directly at the pickup/delivery prompt.
"""
from __future__ import annotations

import sys
import types
import unittest

# Stub redis before any app.session.repository import
_redis_module = types.ModuleType("redis")
_redis_module.Redis = type("_Redis", (), {"__init__": lambda *a, **k: None, "get": lambda *a, **k: None})
sys.modules.setdefault("redis", _redis_module)

from app.session.session import Session
from app.session.repository import _new_order_type_session
from app.state_machine.models.conversation_state import ConversationState


class SessionStartupStateTests(unittest.TestCase):
    def test_session_default_state_is_waiting_for_order_type(self):
        session = Session(session_id="s1", restaurant_id="demo")
        self.assertEqual(
            session.conversation_state,
            ConversationState.WAITING_FOR_ORDER_TYPE,
        )

    def test_enum_does_not_contain_dead_device_type_states(self):
        state_names = {s.name for s in ConversationState}
        self.assertNotIn("WAITING_FOR_CALLER_DEVICE_TYPE", state_names)
        self.assertNotIn("WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION", state_names)

    def test_new_order_type_session_starts_at_order_type(self):
        session = _new_order_type_session("call-abc", "demo")
        self.assertEqual(
            session.conversation_state,
            ConversationState.WAITING_FOR_ORDER_TYPE,
        )

    def test_new_session_caller_device_type_defaults_to_phone(self):
        session = _new_order_type_session("call-abc", "demo")
        self.assertEqual(session.conversation_context.caller_device_type, "phone")

    def test_new_session_order_type_is_unset(self):
        session = _new_order_type_session("call-abc", "demo")
        self.assertIsNone(session.conversation_context.order_type)


if __name__ == "__main__":
    unittest.main()
