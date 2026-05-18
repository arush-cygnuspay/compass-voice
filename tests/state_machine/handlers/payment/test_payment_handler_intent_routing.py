"""Behavior-parity tests for the payment-mode wait choice consolidation.

Asserts that the post-refactor handlers produce HandlerResults that
match the legacy ``detect_payment_wait_mode_choice`` branch outputs
verbatim.
"""
import unittest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.session.session import Session
from app.state_machine.handlers.payment.payment_flow_support import (
    append_payment_event,
)
from app.state_machine.handlers.payment.waiting_for_checkout_completion_handler import (
    WaitingForCheckoutCompletionHandler,
)
from app.state_machine.handlers.payment.waiting_for_payment_handler import (
    WaitingForPaymentHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class _StubCheckoutService:
    pass


class _StubCartSummaryBuilder:
    def build(self, cart):
        return {}


def _payment_session() -> Session:
    session = Session(session_id="pay-1", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_PAYMENT
    return session


def _checkout_session() -> Session:
    session = Session(session_id="chk-1", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
    return session


def _make_context() -> ConversationContext:
    context = ConversationContext()
    context.set_last_nlu(
        user_text="",
        nlu=NLUResult(
            effective_intent=Intent.UNKNOWN,
            intent_confidence=0.0,
            raw_text="",
            normalized_text="",
        ),
    )
    return context


# Expected HandlerResults captured verbatim from the legacy
# detect_payment_wait_mode_choice branches before the refactor.

class PaymentHandlerIntentRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = WaitingForPaymentHandler(
            cart_summary_builder=_StubCartSummaryBuilder(),
            checkout_service=_StubCheckoutService(),
        )

    def test_stay_on_the_line_routes_to_stay_on_call_branch(self):
        session = _payment_session()
        context = _make_context()

        result = self.handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="stay on the line please",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_PAYMENT)
        self.assertEqual(result.response_key, "payment_wait_stay_on_call")
        self.assertEqual(
            result.response_payload,
            append_payment_event(
                None,
                event_name="payment_wait_mode_selected",
                metadata={"mode": "stay_on_call"},
            ),
        )
        self.assertEqual(
            context.delivery_address.payment_wait_mode, "stay_on_call"
        )
        self.assertEqual(
            context.delivery_address.payment_session_state,
            "waiting_payment_stay_on_call",
        )

    def test_after_call_phrase_routes_to_after_call_branch(self):
        session = _payment_session()
        context = _make_context()

        result = self.handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="ill do it later",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_PAYMENT)
        self.assertEqual(result.response_key, "payment_after_call_selected")
        self.assertEqual(
            result.response_payload,
            append_payment_event(
                None,
                event_name="payment_after_call_selected",
                metadata={"mode": "after_call"},
            ),
        )
        self.assertEqual(
            context.delivery_address.payment_wait_mode, "after_call"
        )
        self.assertEqual(
            context.delivery_address.payment_session_state,
            "waiting_payment_after_call",
        )

    def test_cannot_open_link_routes_to_after_call_with_fallback_response(self):
        session = _payment_session()
        context = _make_context()

        result = self.handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="i cant open the link",
            session=session,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_PAYMENT)
        # VOICE_PAYMENT_FALLBACK_AVAILABLE defaults to False so the
        # link_after_call_fallback response_key is selected.
        self.assertEqual(result.response_key, "payment_link_after_call_fallback")
        self.assertEqual(
            result.response_payload,
            append_payment_event(
                None,
                event_name="payment_after_call_selected",
                metadata={"reason": "cannot_open_link"},
            ),
        )
        self.assertEqual(
            context.delivery_address.payment_wait_mode, "after_call"
        )


class CheckoutCompletionHandlerIntentRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = WaitingForCheckoutCompletionHandler(
            checkout_service=_StubCheckoutService(),
        )

    def test_stay_on_the_line_routes_to_checkout_stay_on_call_branch(self):
        session = _checkout_session()
        context = _make_context()

        result = self.handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="stay with me",
            session=session,
        )

        self.assertEqual(
            result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
        )
        self.assertEqual(result.response_key, "checkout_wait_stay_on_call")
        self.assertEqual(
            result.response_payload,
            append_payment_event(
                None,
                event_name="payment_wait_mode_selected",
                metadata={"mode": "stay_on_call"},
            ),
        )

    def test_after_call_phrase_routes_to_checkout_after_call_branch(self):
        session = _checkout_session()
        context = _make_context()

        result = self.handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="not now",
            session=session,
        )

        self.assertEqual(
            result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
        )
        self.assertEqual(result.response_key, "checkout_after_call_selected")

    def test_cannot_open_link_routes_to_checkout_link_after_call_fallback(self):
        session = _checkout_session()
        context = _make_context()

        result = self.handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="cannot open the message",
            session=session,
        )

        self.assertEqual(
            result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
        )
        self.assertEqual(result.response_key, "checkout_link_after_call_fallback")


class PaymentHandlerLegacyParityTests(unittest.TestCase):
    """Parameterized regression diff: every utterance that the legacy
    ``detect_payment_wait_mode_choice`` would have routed to a wait-mode
    branch must produce the same HandlerResult through the new path.
    """

    PAYMENT_FIXTURES: tuple[tuple[str, str, str], ...] = (
        # (user_text, expected_response_key, expected_mode)
        ("stay on the line", "payment_wait_stay_on_call", "stay_on_call"),
        ("stay with me", "payment_wait_stay_on_call", "stay_on_call"),
        ("hold on the line", "payment_wait_stay_on_call", "stay_on_call"),
        ("after this call", "payment_after_call_selected", "after_call"),
        ("ill do it later", "payment_after_call_selected", "after_call"),
        ("not now", "payment_after_call_selected", "after_call"),
        ("later", "payment_after_call_selected", "after_call"),
        ("cant open the link", "payment_link_after_call_fallback", "after_call"),
        ("i cannot open the message", "payment_link_after_call_fallback", "after_call"),
    )

    def test_payment_handler_routes_legacy_phrases_identically(self):
        handler = WaitingForPaymentHandler(
            cart_summary_builder=_StubCartSummaryBuilder(),
            checkout_service=_StubCheckoutService(),
        )

        for text, expected_key, expected_mode in self.PAYMENT_FIXTURES:
            with self.subTest(text=text):
                session = _payment_session()
                context = _make_context()
                result = handler.handle(
                    intent=Intent.UNKNOWN,
                    context=context,
                    user_text=text,
                    session=session,
                )
                self.assertEqual(
                    result.next_state, ConversationState.WAITING_FOR_PAYMENT
                )
                self.assertEqual(result.response_key, expected_key)
                self.assertEqual(
                    context.delivery_address.payment_wait_mode, expected_mode
                )


if __name__ == "__main__":
    unittest.main()
