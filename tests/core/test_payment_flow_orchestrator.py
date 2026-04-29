"""Smoke tests for PaymentFlowOrchestrator (Commit 4 extraction)."""
import sys
import types
import unittest


for _name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
sys.modules["twilio.rest"].Client = type("_Client", (), {"__init__": lambda *a, **k: None})
_redis_module = types.ModuleType("redis")
_redis_module.Redis = type("_Redis", (), {"__init__": lambda *a, **k: None})
sys.modules.setdefault("redis", _redis_module)


from app.core.payment_flow_orchestrator import (
    CHECKOUT_PENDING_REMINDER_INTERVAL_SECONDS,
    PAYMENT_PENDING_REMINDER_INTERVAL_SECONDS,
    PaymentFlowOrchestrator,
)
from app.core.payment_response_classifier import PaymentResponseClassifier
from app.state_machine.models.conversation_state import ConversationState


class PendingPromptIntervalTests(unittest.TestCase):
    def test_interval_for_checkout_completion_state(self):
        self.assertEqual(
            PaymentFlowOrchestrator._pending_prompt_interval_for_state(
                ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
            ),
            CHECKOUT_PENDING_REMINDER_INTERVAL_SECONDS,
        )

    def test_interval_for_waiting_for_payment_state(self):
        self.assertEqual(
            PaymentFlowOrchestrator._pending_prompt_interval_for_state(
                ConversationState.WAITING_FOR_PAYMENT
            ),
            PAYMENT_PENDING_REMINDER_INTERVAL_SECONDS,
        )

    def test_interval_zero_for_other_states(self):
        for state in ConversationState:
            if state in {
                ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                ConversationState.WAITING_FOR_PAYMENT,
            }:
                continue
            with self.subTest(state=state):
                self.assertEqual(
                    PaymentFlowOrchestrator._pending_prompt_interval_for_state(state),
                    0.0,
                )


class IsPaymentPendingResponseTests(unittest.TestCase):
    """These tests previously targeted PaymentFlowOrchestrator._is_payment_pending_response.
    The method has been extracted to PaymentResponseClassifier; tests now target
    the shared classifier directly."""

    def test_true_for_waiting_for_payment_keys(self):
        self.assertTrue(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="waiting_for_payment",
            )
        )
        self.assertTrue(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_not_confirmed_yet",
            )
        )

    def test_true_for_waiting_for_checkout_completion_keys(self):
        self.assertTrue(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                response_key="waiting_for_checkout_completion",
            )
        )

    def test_false_for_unrelated_state_or_key(self):
        self.assertFalse(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.IDLE,
                response_key="waiting_for_payment",
            )
        )
        self.assertFalse(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="ask_for_quantity",
            )
        )


if __name__ == "__main__":
    unittest.main()
