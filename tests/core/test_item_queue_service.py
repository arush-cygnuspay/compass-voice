"""Smoke tests for ItemQueueService (Commit 3 extraction)."""
import unittest

from app.core.item_queue_service import ItemQueueService
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState


class _StubExecutor:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, session, command):
        self.calls.append(command)
        return {}


class ItemQueueServiceSmokeTests(unittest.TestCase):
    def test_try_drain_returns_none_when_response_key_is_not_item_added(self):
        service = ItemQueueService(handlers={}, command_executor=_StubExecutor())
        session = Session(session_id="s1", restaurant_id="steves_grill")
        session.conversation_state = ConversationState.IDLE

        result = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="ask_for_quantity",
        )
        out = service.try_drain(session=session, current_result=result)
        self.assertIsNone(out)

    def test_try_drain_returns_none_when_state_is_not_idle(self):
        service = ItemQueueService(handlers={}, command_executor=_StubExecutor())
        session = Session(session_id="s1", restaurant_id="steves_grill")
        session.conversation_state = ConversationState.WAITING_FOR_MODIFIER

        result = HandlerResult(
            next_state=ConversationState.WAITING_FOR_MODIFIER,
            response_key="item_added_successfully",
        )
        out = service.try_drain(session=session, current_result=result)
        self.assertIsNone(out)

    def test_try_drain_returns_none_when_queue_is_empty(self):
        service = ItemQueueService(handlers={}, command_executor=_StubExecutor())
        session = Session(session_id="s1", restaurant_id="steves_grill")
        session.conversation_state = ConversationState.IDLE
        # pending_item_queue is empty by default

        result = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
        )
        out = service.try_drain(session=session, current_result=result)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
