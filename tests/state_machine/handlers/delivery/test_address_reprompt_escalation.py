import logging
import unittest

from app.cart.cart_item import CartItem
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.session.session import Session
from app.state_machine.handlers.delivery.waiting_for_delivery_address_collection_handler import (
    ADDRESS_FIELD_MAX_REPROMPTS,
    WaitingForDeliveryAddressCollectionHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class _StubCartSummaryBuilder:
    def build(self, cart):
        return {"items": [{"name": "Burger", "quantity": 1}], "total": "$10.00"}


class _StubCheckoutService:
    def ensure_payment_link_for_voice_session(self, *args, **kwargs):
        raise NotImplementedError


def _make_handler() -> WaitingForDeliveryAddressCollectionHandler:
    return WaitingForDeliveryAddressCollectionHandler(
        cart_summary_builder=_StubCartSummaryBuilder(),
        checkout_service=_StubCheckoutService(),
    )


def _make_session() -> Session:
    session = Session(session_id="delivery-1", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION
    session.cart.add_item(
        CartItem.create(
            item_id="burger",
            quantity=1,
            variant_id=None,
            sides={},
            side_variants={},
            modifiers={},
        )
    )
    return session


def _make_context(prompt_field: str) -> ConversationContext:
    context = ConversationContext()
    context.current_prompt_field = prompt_field
    context.delivery_address.area = "Downtown"
    context.delivery_address.postal_code = "12345"
    return context


def _set_last_nlu(context: ConversationContext, intent: Intent) -> None:
    context.set_last_nlu(
        user_text="",
        nlu=NLUResult(
            effective_intent=intent,
            intent_confidence=0.2,
            raw_text="",
            normalized_text="",
        ),
    )


class _EventLogCapture(logging.Handler):
    def __init__(self, target_name: str) -> None:
        super().__init__(level=logging.INFO)
        self.target_name = target_name
        self.events: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == self.target_name:
            self.events.append(record)


class AddressRepromptEscalationTests(unittest.TestCase):
    def test_two_invalid_house_numbers_stay_in_state_and_emit_attempt_count_2(self):
        handler = _make_handler()
        session = _make_session()
        context = _make_context("delivery_house_number")
        _set_last_nlu(context, Intent.UNKNOWN)

        first = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="i can't remember",
            session=session,
        )
        self.assertEqual(first.response_key, "repeat_delivery_house_number")
        self.assertEqual(first.response_payload["attempt_count"], 1)
        self.assertEqual(
            first.next_state, ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
        )

        second = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="i can't remember",
            session=session,
        )
        self.assertEqual(second.response_key, "repeat_delivery_house_number")
        self.assertEqual(second.response_payload["attempt_count"], 2)
        self.assertEqual(
            second.next_state, ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
        )

    def test_three_invalid_house_numbers_escalate_to_human_agent_with_log(self):
        handler = _make_handler()
        session = _make_session()
        context = _make_context("delivery_house_number")
        _set_last_nlu(context, Intent.UNKNOWN)

        capture = _EventLogCapture("address_field_max_attempts_exceeded")
        target_logger = logging.getLogger(
            "app.state_machine.control_intent_resolver"
        )
        previous_level = target_logger.level
        target_logger.addHandler(capture)
        target_logger.setLevel(logging.INFO)

        try:
            for _ in range(ADDRESS_FIELD_MAX_REPROMPTS):
                result = handler.handle(
                    intent=Intent.UNKNOWN,
                    context=context,
                    user_text="i can't remember",
                    session=session,
                )
        finally:
            target_logger.removeHandler(capture)
            target_logger.setLevel(previous_level)

        self.assertEqual(
            result.next_state, ConversationState.TRANSFERRING_TO_HUMAN_AGENT,
        )
        self.assertEqual(result.response_key, "address_collection_giving_up")
        self.assertEqual(result.response_payload["field_name"], "delivery_house_number")
        self.assertEqual(
            result.response_payload["attempt_count"], ADDRESS_FIELD_MAX_REPROMPTS,
        )
        self.assertEqual(len(capture.events), 1)
        self.assertEqual(
            getattr(capture.events[0], "field_name", None),
            "delivery_house_number",
        )
        self.assertEqual(
            getattr(capture.events[0], "attempts", None),
            ADDRESS_FIELD_MAX_REPROMPTS,
        )

    def test_successful_capture_resets_attempt_counter(self):
        handler = _make_handler()
        session = _make_session()
        context = _make_context("delivery_house_number")
        _set_last_nlu(context, Intent.UNKNOWN)

        # One bad attempt.
        handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="i can't remember",
            session=session,
        )
        self.assertEqual(context.reprompt_count("delivery_house_number"), 1)

        # Now a valid number.
        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="142",
            session=session,
        )
        self.assertEqual(result.response_key, "confirm_delivery_house_number")
        self.assertEqual(context.reprompt_count("delivery_house_number"), 0)

    def test_cancel_clears_reprompt_attempts(self):
        handler = _make_handler()
        session = _make_session()
        context = _make_context("delivery_house_number")
        _set_last_nlu(context, Intent.UNKNOWN)

        handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="i can't remember",
            session=session,
        )
        handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="i can't remember",
            session=session,
        )
        self.assertEqual(context.reprompt_count("delivery_house_number"), 2)

        # Cancel mid-flow.
        result = handler.handle(
            intent=Intent.CANCEL_ORDER,
            context=context,
            user_text="cancel",
            session=session,
        )
        self.assertEqual(result.response_key, "confirm_order_summary")
        self.assertEqual(len(context.reprompt_attempts), 0)


if __name__ == "__main__":
    unittest.main()
