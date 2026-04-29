# tests/core/test_payment_response_classifier.py
"""Unit tests for PaymentResponseClassifier — the shared payment-pending
response classification utility extracted from PaymentFlowOrchestrator and
SessionResponseWriter."""
import unittest

from app.core.payment_response_classifier import PaymentResponseClassifier
from app.state_machine.models.conversation_state import ConversationState


class PaymentResponseClassifierTruthTableTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # WAITING_FOR_PAYMENT state
    # ------------------------------------------------------------------

    def test_waiting_for_payment_key_in_payment_state(self):
        self.assertTrue(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="waiting_for_payment",
            )
        )

    def test_payment_not_confirmed_yet_in_payment_state(self):
        self.assertTrue(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="payment_not_confirmed_yet",
            )
        )

    def test_unrelated_key_in_payment_state_returns_false(self):
        self.assertFalse(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="ask_for_quantity",
            )
        )

    # ------------------------------------------------------------------
    # WAITING_FOR_CHECKOUT_COMPLETION state
    # ------------------------------------------------------------------

    def test_checkout_completion_key_in_checkout_state(self):
        self.assertTrue(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                response_key="waiting_for_checkout_completion",
            )
        )

    def test_payment_not_confirmed_yet_in_checkout_state(self):
        self.assertTrue(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                response_key="payment_not_confirmed_yet",
            )
        )

    def test_unrelated_key_in_checkout_state_returns_false(self):
        self.assertFalse(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
                response_key="waiting_for_payment",
            )
        )

    # ------------------------------------------------------------------
    # Non-payment states always return False regardless of key
    # ------------------------------------------------------------------

    def test_false_for_idle_state(self):
        self.assertFalse(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.IDLE,
                response_key="waiting_for_payment",
            )
        )

    def test_false_for_every_non_payment_state(self):
        non_payment_states = [
            s for s in ConversationState
            if s not in {
                ConversationState.WAITING_FOR_PAYMENT,
                ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
            }
        ]
        for state in non_payment_states:
            with self.subTest(state=state):
                self.assertFalse(
                    PaymentResponseClassifier.is_payment_pending_response(
                        state=state,
                        response_key="waiting_for_payment",
                    )
                )

    # ------------------------------------------------------------------
    # Empty / None / blank input
    # ------------------------------------------------------------------

    def test_empty_response_key_returns_false(self):
        self.assertFalse(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="",
            )
        )

    # ------------------------------------------------------------------
    # Cross-state: payment key in wrong state returns False
    # ------------------------------------------------------------------

    def test_checkout_key_in_payment_state_returns_false(self):
        self.assertFalse(
            PaymentResponseClassifier.is_payment_pending_response(
                state=ConversationState.WAITING_FOR_PAYMENT,
                response_key="waiting_for_checkout_completion",
            )
        )


class PaymentResponseClassifierImportTests(unittest.TestCase):
    """Ensure both consumer modules import and call the shared classifier
    rather than keeping their own private copy."""

    def test_session_response_writer_has_no_private_is_payment_pending(self):
        from app.core.session_response_writer import SessionResponseWriter
        self.assertFalse(
            hasattr(SessionResponseWriter, "_is_payment_pending_response"),
            "SessionResponseWriter still owns a private _is_payment_pending_response; "
            "it should delegate to PaymentResponseClassifier.",
        )

    def test_payment_flow_orchestrator_has_no_private_is_payment_pending(self):
        from app.core.payment_flow_orchestrator import PaymentFlowOrchestrator
        self.assertFalse(
            hasattr(PaymentFlowOrchestrator, "_is_payment_pending_response"),
            "PaymentFlowOrchestrator still owns a private _is_payment_pending_response; "
            "it should delegate to PaymentResponseClassifier.",
        )

    def test_session_response_writer_imports_classifier(self):
        import app.core.session_response_writer as mod
        self.assertIn(
            "PaymentResponseClassifier",
            dir(mod),
            "PaymentResponseClassifier should be importable from session_response_writer module.",
        )

    def test_payment_flow_orchestrator_imports_classifier(self):
        import app.core.payment_flow_orchestrator as mod
        self.assertIn(
            "PaymentResponseClassifier",
            dir(mod),
            "PaymentResponseClassifier should be importable from payment_flow_orchestrator module.",
        )


if __name__ == "__main__":
    unittest.main()
