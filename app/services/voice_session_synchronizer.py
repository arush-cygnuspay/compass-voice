# app/services/voice_session_synchronizer.py
"""Synchronise checkout/payment state into the voice (Twilio) session.

Extracted verbatim from CheckoutService._sync_voice_session_from_checkout.
The lazy import pattern is preserved to avoid circular dependencies at
module-load time (checkout_service is imported by API routes which are
imported before the full app graph is available).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.models.checkout_session import CheckoutSession
    from app.models.payment_link_session import PaymentLinkSession

logger = logging.getLogger(__name__)

PAYMENT_FAILURE_STATUSES = {"cancelled", "canceled", "expired", "failed", "declined"}


class VoiceSessionSynchronizer:
    """Push checkout/payment state changes into the voice session store.

    Parameters
    ----------
    find_latest_payment_link_session:
        Callable ``(checkout_token: str) -> PaymentLinkSession | None``.
        Typically ``CheckoutSessionRepository.find_latest_payment_link_session``.
    failure_statuses:
        Set of payment status strings that represent terminal failures.
    """

    def __init__(
        self,
        find_latest_payment_link_session: Callable[[str], "PaymentLinkSession | None"],
        *,
        failure_statuses: frozenset[str] = frozenset(PAYMENT_FAILURE_STATUSES),
    ) -> None:
        self._find_latest_payment_link_session = find_latest_payment_link_session
        self._failure_statuses = failure_statuses

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(
        self,
        checkout_session: "CheckoutSession",
        *,
        mark_completed: bool = False,
    ) -> None:
        """Update the voice session that corresponds to *checkout_session*.

        No-ops silently when:
        - The checkout session has no ``call_sid``.
        - The voice session cannot be loaded (stale / missing).
        - The voice-session imports are unavailable (import guard).
        """
        call_sid = (checkout_session.call_sid or "").strip()
        if not call_sid:
            return

        try:
            from app.session.repository import load_existing_session, save_session
            from app.state_machine.models.conversation_state import ConversationState
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not import voice-session helpers: %s", exc)
            return

        voice_session = load_existing_session(call_sid, checkout_session.restaurant_id)
        if voice_session is None:
            return

        context = voice_session.conversation_context
        delivery = context.delivery_address

        # ---- address fields ----------------------------------------
        if checkout_session.customer_phone_number:
            delivery.customer_phone_number = checkout_session.customer_phone_number
        if checkout_session.order_number:
            delivery.order_number = checkout_session.order_number
        if checkout_session.confirmation_link:
            delivery.confirmation_link = checkout_session.confirmation_link
        if checkout_session.area:
            delivery.area = checkout_session.area
        if checkout_session.postal_code:
            delivery.postal_code = checkout_session.postal_code
        if checkout_session.house_number:
            delivery.house_number = checkout_session.house_number
        if checkout_session.street:
            delivery.street = checkout_session.street
        if checkout_session.secondary_address is not None:
            delivery.secondary_address = checkout_session.secondary_address
        if checkout_session.city:
            delivery.city = checkout_session.city
        if checkout_session.state:
            delivery.state = checkout_session.state
        if checkout_session.full_address_raw:
            delivery.full_address_raw = checkout_session.full_address_raw

        # ---- payment link ------------------------------------------
        latest_payment_link = self._find_latest_payment_link_session(
            checkout_session.token
        )
        if latest_payment_link and latest_payment_link.public_link_url:
            delivery.payment_link = latest_payment_link.public_link_url
            delivery.confirmation_link = latest_payment_link.public_link_url
            if checkout_session.address_required:
                delivery.checkout_status = "checkout_sent"

        # ---- payment status ----------------------------------------
        payment_status = str(checkout_session.last_payment_status or "").strip().lower()
        if checkout_session.payment_completed:
            delivery.payment_status = "payment_confirmed"
            delivery.payment_reference = checkout_session.payment_reference
            if delivery.address_form_link or checkout_session.address_required:
                delivery.checkout_status = "checkout_opened"
        elif payment_status in self._failure_statuses:
            delivery.payment_status = "payment_failed"
            if delivery.address_form_link or checkout_session.address_required:
                delivery.checkout_status = "checkout_opened"
        elif checkout_session.payment_started or payment_status:
            delivery.payment_status = "payment_pending"
            if checkout_session.address_completed or delivery.form_completed:
                delivery.checkout_status = "checkout_opened"
        elif delivery.address_form_link:
            delivery.checkout_status = "checkout_sent"

        # ---- address completion ------------------------------------
        if checkout_session.address_completed:
            delivery.source = "sms_form"
            delivery.form_completed = True
            delivery.collected = True
            delivery.confirmed = True
            context.delivery_address_confirmed = True
            if not checkout_session.payment_completed:
                delivery.checkout_status = "checkout_opened"

        # ---- order completion --------------------------------------
        if mark_completed or checkout_session.payment_completed:
            # reset_session_scope() replaces context.delivery_address with a fresh
            # DeliveryAddress(), discarding all the fields we just synced above.
            # Restore the synced object afterward so consumers still see order_number,
            # form_completed, confirmation_link, etc.
            context.reset_session_scope()
            context.delivery_address = delivery
            voice_session.cart.clear()
            voice_session.conversation_state = ConversationState.COMPLETED
            voice_session.last_response_key = "order_completed"
            voice_session.last_response_payload = {
                "order_number": checkout_session.order_number,
                "payment_reference": checkout_session.payment_reference,
            }

        save_session(voice_session)
