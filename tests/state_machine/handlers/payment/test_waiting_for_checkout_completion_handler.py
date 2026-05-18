import unittest
import sys
import types


twilio_module = types.ModuleType("twilio")
twilio_base_module = types.ModuleType("twilio.base")
twilio_base_exceptions_module = types.ModuleType("twilio.base.exceptions")
twilio_rest_module = types.ModuleType("twilio.rest")


class _TwilioRestException(Exception):
    pass


class _TwilioClient:
    def __init__(self, *args, **kwargs):
        pass


twilio_base_exceptions_module.TwilioRestException = _TwilioRestException
twilio_rest_module.Client = _TwilioClient

sys.modules.setdefault("twilio", twilio_module)
sys.modules.setdefault("twilio.base", twilio_base_module)
sys.modules.setdefault("twilio.base.exceptions", twilio_base_exceptions_module)
sys.modules.setdefault("twilio.rest", twilio_rest_module)

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import NLUResult
from app.session.session import Session
from app.state_machine.handlers.payment.waiting_for_checkout_completion_handler import (
    WaitingForCheckoutCompletionHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class StubCheckoutService:
    def __init__(self, *, paid: bool):
        self.paid = paid

    def verify_payment_by_order_number(self, order_number: str) -> dict:
        return {
            "ok": True,
            "paid": self.paid,
            "payment_completed": self.paid,
            "status": "completed" if self.paid else "pending",
            "reference": "ref-123" if self.paid else None,
            "session": None,
            "error": None,
        }


def _make_session() -> Session:
    session = Session(session_id="call-1", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
    return session


def _make_context() -> ConversationContext:
    context = ConversationContext()
    context.delivery_address.order_number = "1234567"
    context.delivery_address.customer_phone_number = "+15555550123"
    context.delivery_address.address_form_link = "https://example.com/checkout"
    return context


def _set_last_nlu(context: ConversationContext, intent: Intent, confidence: float = 0.2) -> None:
    context.set_last_nlu(
        user_text="",
        nlu=NLUResult(
            effective_intent=intent,
            intent_confidence=confidence,
            raw_text="",
            normalized_text="",
        ),
    )


class WaitingForCheckoutCompletionHandlerTests(unittest.TestCase):
    def test_payment_done_completes_only_after_verified_provider_confirmation(self):
        handler = WaitingForCheckoutCompletionHandler(StubCheckoutService(paid=True))

        result = handler.handle(
            intent=Intent.PAYMENT_DONE,
            context=_make_context(),
            user_text="i paid",
            session=_make_session(),
        )

        self.assertEqual(result.next_state, ConversationState.COMPLETED)
        self.assertEqual(result.response_key, "order_completed")
        self.assertEqual(result.response_payload["order_number"], "1234567")
        self.assertEqual(result.command, {"type": "CLEAR_CART"})

    def test_payment_done_stays_waiting_when_provider_has_not_confirmed_payment(self):
        handler = WaitingForCheckoutCompletionHandler(StubCheckoutService(paid=False))

        result = handler.handle(
            intent=Intent.PAYMENT_DONE,
            context=_make_context(),
            user_text="i paid",
            session=_make_session(),
        )

        self.assertEqual(
            result.next_state,
            ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
        )
        self.assertEqual(result.response_key, "payment_not_confirmed_yet")

    def test_natural_affirmation_checks_payment_status(self):
        handler = WaitingForCheckoutCompletionHandler(StubCheckoutService(paid=False))
        context = _make_context()
        _set_last_nlu(context, Intent.UNKNOWN)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="go ahead",
            session=_make_session(),
        )

        self.assertEqual(result.response_key, "payment_not_confirmed_yet")

    def test_deny_like_reply_keeps_waiting_for_checkout(self):
        handler = WaitingForCheckoutCompletionHandler(StubCheckoutService(paid=False))
        context = _make_context()
        _set_last_nlu(context, Intent.UNKNOWN)

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="not yet",
            session=_make_session(),
        )

        self.assertEqual(result.response_key, "waiting_for_checkout_completion")
        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)


    def test_repeat_that_repeats_checkout_instruction_without_advancing_state(self):
        handler = WaitingForCheckoutCompletionHandler(StubCheckoutService(paid=False))
        context = _make_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="repeat that",
            session=_make_session(),
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)
        self.assertEqual(result.response_key, "waiting_for_checkout_completion")

    def test_cancel_does_not_silently_cancel_checkout_flow(self):
        handler = WaitingForCheckoutCompletionHandler(StubCheckoutService(paid=False))
        context = _make_context()

        result = handler.handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="cancel",
            session=_make_session(),
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)
        self.assertEqual(result.response_key, "cannot_cancel_during_checkout")

    def test_resend_link_still_works_during_checkout_wait(self):
        handler = WaitingForCheckoutCompletionHandler(StubCheckoutService(paid=False))
        context = _make_context()

        result = handler.handle(
            intent=Intent.PAYMENT_REQUEST,
            context=context,
            user_text="resend link",
            session=_make_session(),
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_CHECKOUT_COMPLETION)
        self.assertEqual(result.response_key, "checkout_link_resent")


if __name__ == "__main__":
    unittest.main()


