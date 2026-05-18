"""Behavior-parity tests for the delivery-address SKIP consolidation."""
import unittest

from app.cart.cart_item import CartItem
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.session.session import Session
from app.state_machine.handlers.delivery.waiting_for_delivery_address_collection_handler import (
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
    session = Session(session_id="dlv-1", restaurant_id="steves_grill")
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


def _make_context_for_secondary_step(intent: Intent = Intent.UNKNOWN) -> ConversationContext:
    context = ConversationContext()
    context.current_prompt_field = "delivery_secondary_address"
    context.delivery_address.area = "Downtown"
    context.delivery_address.postal_code = "12345"
    context.delivery_address.house_number = "42"
    context.delivery_address.street = "Main Street"
    context.set_last_nlu(
        user_text="",
        nlu=NLUResult(
            effective_intent=intent,
            intent_confidence=0.0,
            raw_text="",
            normalized_text="",
        ),
    )
    return context


class DeliveryAddressIntentRoutingTests(unittest.TestCase):
    """Verify the secondary-address skip path through the resolver
    produces the same outcome as the legacy ``is_optional_skip_response``
    path."""

    def test_no_apartment_exact_match_skips_via_handler_set(self):
        handler = _make_handler()
        session = _make_session()
        # Use intent=UNKNOWN so we hit the OPTIONAL_NONE_WORDS exact-match
        # path (legacy `text in self.OPTIONAL_NONE_WORDS`).
        context = _make_context_for_secondary_step(intent=Intent.UNKNOWN)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="no apartment",
            session=session,
        )

        # Should have moved past the secondary step (either to payment
        # or to error_recovery if checkout fails). The key signal is
        # that we didn't loop back to ask for the secondary address.
        self.assertNotEqual(result.response_key, "confirm_delivery_secondary_address")
        self.assertNotEqual(result.response_key, "ask_for_delivery_secondary_address")
        self.assertIsNone(context.delivery_address.secondary_address)

    def test_skip_via_phrase_fallback_resolves_to_skip_kind(self):
        handler = _make_handler()
        session = _make_session()
        context = _make_context_for_secondary_step(intent=Intent.UNKNOWN)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="skip",
            session=session,
        )

        self.assertNotEqual(result.response_key, "confirm_delivery_secondary_address")
        self.assertIsNone(context.delivery_address.secondary_address)

    def test_deny_intent_skips_secondary_address(self):
        handler = _make_handler()
        session = _make_session()
        # Intent.DENY should also be treated as a skip (parity with
        # legacy `is_optional_skip_response` check `intent in DENY_INTENTS`).
        context = _make_context_for_secondary_step(intent=Intent.DENY)

        result = handler.handle(
            intent=Intent.DENY,
            context=context,
            user_text="i dont have one",  # not in OPTIONAL_NONE_WORDS or _SKIP_PHRASES
            session=session,
        )

        self.assertNotEqual(result.response_key, "confirm_delivery_secondary_address")
        self.assertIsNone(context.delivery_address.secondary_address)

    def test_real_apartment_value_does_not_trigger_skip(self):
        handler = _make_handler()
        session = _make_session()
        context = _make_context_for_secondary_step(intent=Intent.UNKNOWN)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="apartment 4B",
            session=session,
        )

        # Real apartment value should be captured, not skipped.
        self.assertEqual(result.response_key, "confirm_delivery_secondary_address")
        self.assertEqual(context.delivery_address.secondary_address, "apartment 4B")


if __name__ == "__main__":
    unittest.main()
