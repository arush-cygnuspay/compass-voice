# app/core/payment_response_classifier.py
"""Shared payment-pending response classification.

Previously duplicated as a private static method on both
``PaymentFlowOrchestrator`` and ``SessionResponseWriter``.  Extracted here
so both modules share a single source of truth with no circular imports.
"""
from __future__ import annotations

from app.state_machine.models.conversation_state import ConversationState

_PAYMENT_PENDING_KEYS: frozenset[str] = frozenset(
    {"waiting_for_payment", "payment_not_confirmed_yet"}
)
_CHECKOUT_PENDING_KEYS: frozenset[str] = frozenset(
    {"waiting_for_checkout_completion", "payment_not_confirmed_yet"}
)


class PaymentResponseClassifier:
    """Stateless classifier: maps (state, response_key) → is-payment-pending."""

    @staticmethod
    def is_payment_pending_response(
        *,
        state: ConversationState,
        response_key: str,
    ) -> bool:
        """Return True when *response_key* is a payment-pending prompt in *state*.

        Empty / None inputs always return False.
        """
        if not response_key:
            return False
        if state == ConversationState.WAITING_FOR_PAYMENT:
            return response_key in _PAYMENT_PENDING_KEYS
        if state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION:
            return response_key in _CHECKOUT_PENDING_KEYS
        return False
