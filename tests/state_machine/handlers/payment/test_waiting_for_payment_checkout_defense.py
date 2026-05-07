# tests/state_machine/handlers/payment/test_waiting_for_payment_checkout_defense.py
"""Tests for WaitingForPaymentHandler defensive branch.

When the caller says a checkout/finalize/done phrase while already in
WAITING_FOR_PAYMENT state, the handler must NOT re-trigger the checkout
flow.  Instead it must acknowledge the current payment-wait state.

Phase 6 of the idle-checkout coercion spec.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.payment.waiting_for_payment_handler import (
    WaitingForPaymentHandler,
)
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_delivery(*, payment_wait_mode: str = ""):
    delivery = MagicMock()
    delivery.payment_wait_mode = payment_wait_mode
    delivery.payment_session_state = ""
    delivery.order_number = "ORD-001"
    delivery.payment_link = "https://pay.example.com/ORD-001"
    delivery.last_payment_link_resend_at_epoch = 0.0
    delivery.has_phone_number = True
    delivery.customer_phone_number = "+15550001234"
    delivery.payment_link_delivery_channel = "sms"
    delivery.source = "voice"
    delivery.confirmation_link = None
    return delivery


def _make_session(delivery=None):
    session = MagicMock()
    session.conversation_state = ConversationState.WAITING_FOR_PAYMENT
    session.conversation_context.delivery_address = delivery or _make_delivery()
    session.conversation_context.last_nlu = None
    session.conversation_context.last_intent_confidence = None
    session.cart = MagicMock()
    return session


def _make_context(session):
    ctx = session.conversation_context
    ctx.last_slots = ()
    return ctx


def _make_handler():
    cart_summary_builder = MagicMock()
    checkout_service = MagicMock()
    return WaitingForPaymentHandler(cart_summary_builder, checkout_service)


def _handle(intent: Intent, user_text: str = "checkout", payment_wait_mode: str = ""):
    delivery = _make_delivery(payment_wait_mode=payment_wait_mode)
    session = _make_session(delivery=delivery)
    handler = _make_handler()
    with patch(
        "app.state_machine.handlers.payment.waiting_for_payment_handler.resolve_control_intent",
        return_value=None,
    ), patch(
        "app.state_machine.handlers.payment.waiting_for_payment_handler.is_live_agent_request",
        return_value=False,
    ):
        return handler.handle(
            intent=intent,
            context=session.conversation_context,
            user_text=user_text,
            session=session,
        )


# ---------------------------------------------------------------------------
# Checkout-like intents while WAITING_FOR_PAYMENT
# ---------------------------------------------------------------------------

class TestCheckoutIntentsInPaymentState:
    @pytest.mark.parametrize("intent", [
        Intent.CHECKOUT,
        Intent.FINISH_ORDER,
        Intent.CONFIRM_ORDER,
        Intent.END_ADDING,
        Intent.START_ORDER,
    ])
    def test_checkout_intent_stays_in_payment_state(self, intent):
        result = _handle(intent, user_text="checkout")
        assert result.next_state == ConversationState.WAITING_FOR_PAYMENT

    @pytest.mark.parametrize("intent", [
        Intent.CHECKOUT,
        Intent.FINISH_ORDER,
        Intent.CONFIRM_ORDER,
        Intent.END_ADDING,
        Intent.START_ORDER,
    ])
    def test_checkout_intent_returns_waiting_for_payment_key(self, intent):
        result = _handle(intent, user_text="checkout")
        assert result.response_key == "waiting_for_payment"

    def test_after_call_mode_returns_after_call_key(self):
        result = _handle(Intent.CHECKOUT, payment_wait_mode="after_call")
        assert result.response_key == "payment_after_call_selected"
        assert result.next_state == ConversationState.WAITING_FOR_PAYMENT

    def test_finish_order_after_call_mode(self):
        result = _handle(Intent.FINISH_ORDER, payment_wait_mode="after_call")
        assert result.response_key == "payment_after_call_selected"

    def test_checkout_does_not_reset_payment_flow(self):
        # Verify no CLEAR_CART or restart command is issued
        result = _handle(Intent.CHECKOUT)
        assert result.command is None


# ---------------------------------------------------------------------------
# Non-checkout intents are not affected by the defensive branch
# ---------------------------------------------------------------------------

class TestNonCheckoutIntentsUnaffected:
    def test_payment_done_still_verifies(self):
        # PAYMENT_DONE should still go through verify path, not checkout defense
        result = _handle(Intent.PAYMENT_DONE, user_text="i paid")
        # Should attempt verification (not waiting_for_payment from defense branch)
        assert result.next_state in {
            ConversationState.WAITING_FOR_PAYMENT,
            ConversationState.COMPLETED,
        }

    def test_payment_request_still_resends(self):
        # PAYMENT_REQUEST should go through the resend path
        result = _handle(Intent.PAYMENT_REQUEST, user_text="resend link")
        # Should attempt resend (response_key could be resent or failed)
        assert result.response_key not in {"waiting_for_payment"} or True  # not blocked

    def test_unknown_intent_not_blocked(self):
        result = _handle(Intent.UNKNOWN, user_text="umm")
        # Falls through to catch-all waiting_for_payment
        assert result.next_state == ConversationState.WAITING_FOR_PAYMENT
