import unittest

from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handlers.system.waiting_for_caller_device_type_handler import (
    HUMAN_AGENT_TRANSFER_NUMBER,
    WaitingForCallerDeviceTypeHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class WaitingForCallerDeviceTypeHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = WaitingForCallerDeviceTypeHandler()

    def test_landline_starts_pickup_only_confirmation(self):
        session = Session(session_id="call-1", restaurant_id="demo")
        session.conversation_state = ConversationState.WAITING_FOR_CALLER_DEVICE_TYPE
        context = ConversationContext()

        result = self.handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="landline",
            session=session,
        )

        self.assertEqual(
            result.next_state,
            ConversationState.WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION,
        )
        self.assertEqual(result.response_key, "confirm_landline_pickup_only")
        self.assertEqual(context.caller_device_type, "landline")

    def test_landline_confirmation_yes_transfers_to_human_agent(self):
        session = Session(session_id="call-1", restaurant_id="demo")
        session.conversation_state = ConversationState.WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION
        context = ConversationContext()
        context.caller_device_type = "landline"

        result = self.handler.handle(
            intent=Intent.AFFIRM,
            context=context,
            user_text="yes proceed",
            session=session,
        )

        self.assertEqual(
            result.next_state,
            ConversationState.TRANSFERRING_TO_HUMAN_AGENT,
        )
        self.assertEqual(result.response_key, "transferring_to_human_agent")
        self.assertEqual(
            result.command,
            {
                "type": "transfer_call",
                "transfer_number": HUMAN_AGENT_TRANSFER_NUMBER,
            },
        )

    def test_landline_confirmation_no_ends_call(self):
        session = Session(session_id="call-1", restaurant_id="demo")
        session.conversation_state = ConversationState.WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION
        context = ConversationContext()
        context.caller_device_type = "landline"

        result = self.handler.handle(
            intent=Intent.DENY,
            context=context,
            user_text="no",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.COMPLETED)
        self.assertEqual(result.response_key, "landline_pickup_declined")


if __name__ == "__main__":
    unittest.main()
