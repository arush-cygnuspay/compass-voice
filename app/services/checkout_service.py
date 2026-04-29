# app/services/checkout_service.py
"""Thin orchestration layer for checkout use cases.

All persistence, payment polling, voice-session synchronisation, and
order-number generation are delegated to focused sub-components.
This module retains the module-level constants (CHECKOUT_DATA_DIR etc.)
so that existing tests can patch them via ``patch.object(checkout_service_module, ...)``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

from app.config.payment import get_payment_config
from app.models.checkout_session import CheckoutSession
from app.models.payment_link_session import PaymentLinkSession
from app.repositories.checkout_session_repository import (
    CheckoutNotFoundError,
    CheckoutExpiredError,
    CheckoutSessionRepository,
)
from app.services.live_call_service import LiveCallService
from app.services.order_number_generator import OrderNumberGenerator
from app.services.payment.datacap_payment_links_service import DatacapPaymentLinksService
from app.services.payment_polling_orchestrator import (
    PaymentPollingOrchestrator,
    PAYMENT_FAILURE_STATUSES,
)
from app.services.sms_service import SmsSendRequest, SmsService
from app.services.voice_session_synchronizer import VoiceSessionSynchronizer

logger = logging.getLogger(__name__)

# Re-export errors so callers importing from this module keep working.
__all__ = [
    "CheckoutService",
    "CheckoutNotFoundError",
    "CheckoutExpiredError",
    "ReverseGeocodeError",
]

# ---------------------------------------------------------------------------
# Module-level constants — kept here so tests can patch them via
#   patch.object(checkout_service_module, "CHECKOUT_DATA_DIR", ...)
# Sourced from PaymentConfig; this module no longer calls os.getenv directly.
# ---------------------------------------------------------------------------

def _init_module_constants() -> None:
    global PAYMENT_POLL_INTERVAL_SECONDS, PAYMENT_POLL_MAX_DURATION_SECONDS
    global CHECKOUT_DATA_DIR, PAYMENT_LINK_SESSION_DATA_DIR
    global CHECKOUT_INDEX_DIR, CHECKOUT_ORDER_INDEX_DIR
    global PAYMENT_LINK_INDEX_DIR, PAYMENT_LINK_BY_CHECKOUT_INDEX_DIR
    global PAYMENT_LINK_BY_REQUEST_INDEX_DIR
    global PUBLIC_CHECKOUT_BASE_URL, REVERSE_GEOCODE_URL, REVERSE_GEOCODE_USER_AGENT
    cfg = get_payment_config()

    PAYMENT_POLL_INTERVAL_SECONDS = cfg.payment_poll_interval_seconds
    PAYMENT_POLL_MAX_DURATION_SECONDS = cfg.payment_poll_max_duration_seconds

    CHECKOUT_DATA_DIR = cfg.checkout_data_dir
    CHECKOUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    PAYMENT_LINK_SESSION_DATA_DIR = cfg.payment_link_session_data_dir
    PAYMENT_LINK_SESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)

    CHECKOUT_INDEX_DIR = CHECKOUT_DATA_DIR / "_indexes"
    CHECKOUT_ORDER_INDEX_DIR = CHECKOUT_INDEX_DIR / "by_order_number"
    PAYMENT_LINK_INDEX_DIR = PAYMENT_LINK_SESSION_DATA_DIR / "_indexes"
    PAYMENT_LINK_BY_CHECKOUT_INDEX_DIR = PAYMENT_LINK_INDEX_DIR / "latest_by_checkout_token"
    PAYMENT_LINK_BY_REQUEST_INDEX_DIR = PAYMENT_LINK_INDEX_DIR / "by_request_id"

    for _directory in (
        CHECKOUT_INDEX_DIR,
        CHECKOUT_ORDER_INDEX_DIR,
        PAYMENT_LINK_INDEX_DIR,
        PAYMENT_LINK_BY_CHECKOUT_INDEX_DIR,
        PAYMENT_LINK_BY_REQUEST_INDEX_DIR,
    ):
        _directory.mkdir(parents=True, exist_ok=True)

    PUBLIC_CHECKOUT_BASE_URL = cfg.public_checkout_base_url
    REVERSE_GEOCODE_URL = cfg.reverse_geocode_url
    REVERSE_GEOCODE_USER_AGENT = cfg.reverse_geocode_user_agent


_init_module_constants()

RESTAURANT_DATA_ROOT = Path("app/data/restaurants")


class ReverseGeocodeError(Exception):
    pass


class CheckoutService:
    """Thin coordinator for checkout use cases.

    Constructor accepts optional pre-built components so tests can inject
    fakes/stubs without touching module globals.  All parameters have
    sensible defaults so ``CheckoutService()`` still works with no arguments.
    """

    def __init__(
        self,
        *,
        repository: CheckoutSessionRepository | None = None,
        payment_provider: DatacapPaymentLinksService | None = None,
        sms_service: SmsService | None = None,
        live_call_service: LiveCallService | None = None,
        order_number_generator: OrderNumberGenerator | None = None,
        polling_orchestrator: PaymentPollingOrchestrator | None = None,
        voice_synchronizer: VoiceSessionSynchronizer | None = None,
    ) -> None:
        # Persistence
        self.repository = repository or CheckoutSessionRepository(
            data_dir=CHECKOUT_DATA_DIR,
            payment_link_session_dir=PAYMENT_LINK_SESSION_DATA_DIR,
        )

        # External service dependencies (kept as direct attributes so test
        # code can do ``service.sms_service = Dummy()`` after construction).
        self.payment_provider = payment_provider or DatacapPaymentLinksService()
        self.sms_service = sms_service or SmsService()
        self.live_call_service = live_call_service or LiveCallService()

        # Order-number minting
        self.order_number_generator = order_number_generator or OrderNumberGenerator()

        # Payment polling (wired after payment_provider is set so the verify
        # callable can reference self.verify_payment_with_provider).
        self.polling_orchestrator = polling_orchestrator or PaymentPollingOrchestrator(
            verify_fn=self.verify_payment_with_provider,
            poll_interval=PAYMENT_POLL_INTERVAL_SECONDS,
            poll_max_duration=PAYMENT_POLL_MAX_DURATION_SECONDS,
            failure_statuses=frozenset(PAYMENT_FAILURE_STATUSES),
        )

        # Voice-session sync
        self.voice_synchronizer = voice_synchronizer or VoiceSessionSynchronizer(
            find_latest_payment_link_session=self.repository.find_latest_payment_link_session,
            failure_statuses=frozenset(PAYMENT_FAILURE_STATUSES),
        )

    # ------------------------------------------------------------------
    # Delegated persistence helpers (kept so existing callers compile)
    # ------------------------------------------------------------------

    def save_session(self, session: CheckoutSession) -> None:
        self.repository.save_session(session)

    def save_payment_link_session(self, payment_link_session: PaymentLinkSession) -> None:
        self.repository.save_payment_link_session(payment_link_session)

    def get_session(self, token: str) -> CheckoutSession:
        return self.repository.get_session(token)

    def find_session_by_order_number(
        self, order_number: str | None
    ) -> CheckoutSession | None:
        return self.repository.find_session_by_order_number(order_number)

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def build_checkout_url(self, token: str) -> str:
        return f"{PUBLIC_CHECKOUT_BASE_URL}/{token}"

    # ------------------------------------------------------------------
    # Session serialisation
    # ------------------------------------------------------------------

    def serialize_session(self, session: CheckoutSession) -> dict[str, Any]:
        payload = session.to_dict()
        latest_payment_link = self.repository.find_latest_payment_link_session(session.token)
        live_cart_url = self.build_checkout_url(session.token)

        payload["payment_link_url"] = (
            latest_payment_link.public_link_url if latest_payment_link else None
        )
        payload["payment_link_embedded_url"] = (
            latest_payment_link.public_link_embedded_url
            if latest_payment_link
            else None
        )
        payload["payment_link_qr_code_url"] = (
            latest_payment_link.public_link_qr_code_url if latest_payment_link else None
        )
        payload["payment_link_status"] = (
            latest_payment_link.status if latest_payment_link else None
        )
        payload["payment_link_request_id"] = (
            latest_payment_link.request_id if latest_payment_link else None
        )
        payload["live_cart_url"] = live_cart_url
        payload["status_tracking_url"] = session.confirmation_link or live_cart_url
        payload["payment_retry_available"] = bool(session.can_retry_payment)
        payload["order_state"] = session.status

        return payload

    # ------------------------------------------------------------------
    # Session creation / mutation
    # ------------------------------------------------------------------

    def create_session(
        self,
        *,
        restaurant_id: str,
        call_sid: str | None,
        order_number: str | None,
        customer_phone_number: str | None,
        address_required: bool,
        area: str | None,
        postal_code: str | None,
        order_summary: dict[str, Any] | None = None,
    ) -> CheckoutSession:
        if not order_number or not order_number.isdigit():
            order_number = self.order_number_generator.generate(restaurant_id)
        session = CheckoutSession.new(
            restaurant_id=restaurant_id,
            call_sid=call_sid,
            order_number=order_number,
            customer_phone_number=customer_phone_number,
            address_required=address_required,
            area=area,
            postal_code=postal_code,
            order_summary=order_summary,
        )
        session.confirmation_link = self._resolve_confirmation_link(
            restaurant_id=restaurant_id,
            order_number=session.order_number,
        )
        self.repository.save_session(session)
        return session

    def update_address(
        self,
        *,
        token: str,
        house_number: str,
        street: str,
        secondary_address: str | None,
    ) -> CheckoutSession:
        session = self.repository.get_session(token)
        session.house_number = house_number.strip()
        session.street = street.strip()
        session.secondary_address = (secondary_address or "").strip() or None
        session.address_source = "manual"
        session.mark_address_completed()
        self.repository.save_session(session)
        self.voice_synchronizer.sync(session)
        return session

    def update_location(
        self,
        *,
        token: str,
        latitude: float | None,
        longitude: float | None,
        permission_granted: bool | None,
    ) -> CheckoutSession:
        session = self.repository.get_session(token)
        session.latitude = latitude
        session.longitude = longitude
        session.location_permission_granted = permission_granted

        if permission_granted and latitude is not None and longitude is not None:
            resolved = self._reverse_geocode(latitude=latitude, longitude=longitude)
            session.house_number = resolved.get("house_number")
            session.street = resolved.get("street")
            session.secondary_address = None
            session.area = resolved.get("area") or session.area
            session.postal_code = resolved.get("postal_code") or session.postal_code
            session.city = resolved.get("city")
            session.state = resolved.get("state")
            session.full_address_raw = resolved.get("display_name")
            session.address_source = "device_location"
            session.mark_address_completed()

        self.repository.save_session(session)
        self.voice_synchronizer.sync(session)
        return session

    def complete_checkout(
        self,
        token: str,
        *,
        payment_reference: str | None = None,
    ) -> CheckoutSession:
        session = self.repository.get_session(token)
        session.mark_payment_completed(reference=payment_reference)
        self.repository.save_session(session)
        self.voice_synchronizer.sync(session, mark_completed=True)
        return session

    # ------------------------------------------------------------------
    # Payment flow
    # ------------------------------------------------------------------

    def ensure_payment_link(
        self,
        *,
        restaurant_id: str,
        call_sid: str | None,
        order_number: str | None,
        customer_phone_number: str | None,
        address_required: bool,
        area: str | None,
        postal_code: str | None,
        order_summary: dict[str, Any] | None,
        house_number: str | None = None,
        street: str | None = None,
        secondary_address: str | None = None,
        city: str | None = None,
        state: str | None = None,
        full_address_raw: str | None = None,
        address_source: str | None = None,
    ) -> dict[str, Any]:
        session = self._upsert_checkout_session(
            restaurant_id=restaurant_id,
            call_sid=call_sid,
            order_number=order_number,
            customer_phone_number=customer_phone_number,
            address_required=address_required,
            area=area,
            postal_code=postal_code,
            order_summary=order_summary,
            house_number=house_number,
            street=street,
            secondary_address=secondary_address,
            city=city,
            state=state,
            full_address_raw=full_address_raw,
            address_source=address_source,
        )

        latest = self.repository.find_latest_payment_link_session(session.token)
        if session.payment_completed:
            return {
                "ok": True,
                "checkout_token": session.token,
                "order_number": session.order_number,
                "payment_completed": True,
                "status": "completed",
                "confirmation_link": session.confirmation_link,
                "redirect_url": latest.public_link_url if latest else None,
                "embedded_url": latest.public_link_embedded_url if latest else None,
                "qr_code_url": latest.public_link_qr_code_url if latest else None,
                "provider_reference": session.payment_reference,
                "payment_link_session_id": latest.id if latest else None,
                "request_id": latest.request_id if latest else None,
            }

        if latest and latest.public_link_url:
            status_lower = str(latest.status or "").lower()
            if status_lower in DatacapPaymentLinksService.PAID_STATUSES:
                result = self.verify_payment_with_provider(session.token)
                if result.get("payment_completed"):
                    latest = self.repository.find_latest_payment_link_session(session.token)
                    return {
                        "ok": True,
                        "checkout_token": session.token,
                        "order_number": session.order_number,
                        "payment_completed": True,
                        "status": result.get("status") or "completed",
                        "confirmation_link": session.confirmation_link,
                        "redirect_url": latest.public_link_url if latest else None,
                        "embedded_url": latest.public_link_embedded_url if latest else None,
                        "qr_code_url": latest.public_link_qr_code_url if latest else None,
                        "provider_reference": result.get("reference"),
                        "payment_link_session_id": latest.id if latest else None,
                        "request_id": latest.request_id if latest else None,
                    }

            if status_lower not in PAYMENT_FAILURE_STATUSES:
                if not session.payment_started:
                    session.mark_payment_started()
                    self.repository.save_session(session)
                self.polling_orchestrator.start(session.token)
                return {
                    "ok": True,
                    "checkout_token": session.token,
                    "order_number": session.order_number,
                    "payment_completed": False,
                    "status": latest.status,
                    "confirmation_link": session.confirmation_link,
                    "redirect_url": latest.public_link_url,
                    "embedded_url": latest.public_link_embedded_url,
                    "qr_code_url": latest.public_link_qr_code_url,
                    "provider_reference": latest.provider_reference,
                    "payment_link_session_id": latest.id,
                    "request_id": latest.request_id,
                }

        amount = self._payment_amount_from_summary(order_summary)
        result = self.start_payment(token=session.token, amount=amount)
        result.update(
            {
                "checkout_token": session.token,
                "order_number": session.order_number,
                "payment_completed": False,
                "confirmation_link": session.confirmation_link,
            }
        )
        return result

    def start_payment(
        self,
        *,
        token: str,
        amount: str,
    ) -> dict[str, Any]:
        session = self.repository.get_session(token)

        if session.address_required and not session.address_completed:
            raise ValueError("Address must be completed before payment.")

        credit_mid = os.getenv("DATACAP_CREDIT_MERCHANT_ID", "").strip()
        if not credit_mid:
            raise ValueError("DATACAP_CREDIT_MERCHANT_ID is not configured.")

        payment_link_session = self.payment_provider.create_payment_link(
            checkout_token=session.token,
            invoice_no=session.order_number,
            amount=amount,
            credit_mid=credit_mid,
        )

        session.mark_payment_started()
        self.repository.save_session(session)
        self.repository.save_payment_link_session(payment_link_session)
        logger.info(
            "Started payment link for order=%s token=%s request_id=%s",
            session.order_number,
            session.token,
            payment_link_session.request_id,
        )
        self.polling_orchestrator.start(session.token)

        return {
            "ok": True,
            "provider": "datacap_payment_links",
            "status": payment_link_session.status,
            "confirmation_link": session.confirmation_link,
            "redirect_url": payment_link_session.public_link_url,
            "embedded_url": payment_link_session.public_link_embedded_url,
            "qr_code_url": payment_link_session.public_link_qr_code_url,
            "provider_reference": payment_link_session.provider_reference,
            "payment_link_session_id": payment_link_session.id,
            "request_id": payment_link_session.request_id,
        }

    def verify_payment_with_provider(self, token: str) -> dict[str, Any]:
        try:
            session = self.repository.get_session(token)
        except (CheckoutNotFoundError, CheckoutExpiredError) as exc:
            return {
                "ok": False,
                "paid": False,
                "payment_completed": False,
                "status": "",
                "reference": None,
                "session": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if session.payment_completed:
            return {
                "ok": True,
                "paid": True,
                "payment_completed": True,
                "status": "completed",
                "reference": session.payment_reference,
                "session": self.serialize_session(session),
                "error": None,
            }

        if not session.payment_started and not session.can_retry_payment:
            return {
                "ok": True,
                "paid": False,
                "payment_completed": False,
                "status": "not_started",
                "reference": None,
                "session": self.serialize_session(session),
                "error": None,
            }

        payment_link_session = self.repository.find_latest_payment_link_session(token)
        if not payment_link_session or not payment_link_session.request_id:
            if session.can_retry_payment:
                return {
                    "ok": True,
                    "paid": False,
                    "payment_completed": False,
                    "status": session.last_payment_status or session.status,
                    "reference": None,
                    "session": self.serialize_session(session),
                    "error": None,
                }
            return {
                "ok": False,
                "paid": False,
                "payment_completed": False,
                "status": "",
                "reference": None,
                "session": self.serialize_session(session),
                "error": "No payment link session / request_id on file.",
            }

        status_info = self.payment_provider.get_payment_status(
            request_id=payment_link_session.request_id
        )

        if status_info.get("ok"):
            payment_link_session.status = (
                status_info.get("status") or payment_link_session.status
            )
            if status_info.get("raw"):
                payment_link_session.raw_response = status_info["raw"]
            self.repository.save_payment_link_session(payment_link_session)

        if not status_info.get("ok"):
            return {
                "ok": False,
                "paid": False,
                "payment_completed": False,
                "status": status_info.get("status", ""),
                "reference": None,
                "session": self.serialize_session(session),
                "error": status_info.get("error"),
            }

        if status_info.get("paid"):
            updated = self.handle_payment_completed(
                order_number=session.order_number,
                payment_reference=status_info.get("reference")
                or payment_link_session.request_id,
            ) or session
            return {
                "ok": True,
                "paid": True,
                "payment_completed": True,
                "status": status_info.get("status"),
                "reference": updated.payment_reference,
                "session": self.serialize_session(updated),
                "error": None,
            }

        status_lower = str(status_info.get("status") or "").lower()
        if status_lower in PAYMENT_FAILURE_STATUSES:
            session.mark_payment_retryable(status_lower)
            self.repository.save_session(session)
            self.voice_synchronizer.sync(session)
            logger.info(
                "Payment requires retry for order=%s token=%s status=%s",
                session.order_number,
                session.token,
                status_lower,
            )
            return {
                "ok": True,
                "paid": False,
                "payment_completed": False,
                "status": status_lower,
                "reference": None,
                "session": self.serialize_session(session),
                "error": None,
            }

        if status_lower and status_lower != (session.last_payment_status or "").lower():
            session.last_payment_status = status_lower
            self.repository.save_session(session)
        self.voice_synchronizer.sync(session)
        return {
            "ok": True,
            "paid": False,
            "payment_completed": False,
            "status": status_info.get("status"),
            "reference": None,
            "session": self.serialize_session(session),
            "error": None,
        }

    def verify_payment_by_order_number(self, order_number: str) -> dict[str, Any]:
        session = self.repository.find_session_by_order_number(order_number)
        if session is None:
            return {
                "ok": False,
                "paid": False,
                "payment_completed": False,
                "status": "",
                "reference": None,
                "session": None,
                "error": f"No checkout session found for order_number={order_number}",
            }
        return self.verify_payment_with_provider(session.token)

    def handle_payment_completed(
        self,
        *,
        order_number: str,
        payment_reference: str | None = None,
    ) -> CheckoutSession | None:
        session = self.repository.find_session_by_order_number(order_number)
        if session is None:
            logger.warning(
                "handle_payment_completed: no session found for order_number=%s",
                order_number,
            )
            return None

        if session.payment_completed:
            logger.info(
                "handle_payment_completed: order %s already completed, skipping",
                order_number,
            )
            self.voice_synchronizer.sync(session, mark_completed=True)
            return session

        session.mark_payment_completed(reference=payment_reference)
        self.repository.save_session(session)
        self.voice_synchronizer.sync(session, mark_completed=True)
        logger.info(
            "Payment completed for order %s (token=%s)", order_number, session.token
        )

        # ---- SMS dispatch ------------------------------------------
        phone_number = (session.customer_phone_number or "").strip()
        if phone_number:
            latest_payment = self.repository.find_latest_payment_link_session(session.token)
            order_link = (
                (latest_payment.public_link_url if latest_payment else None)
                or session.confirmation_link
                or ""
            )
            from app.services.sms_exceptions import SmsError
            try:
                sms_result = self.sms_service.send(
                    SmsSendRequest(
                        template="order_confirmation",
                        phone_number=phone_number,
                        order_number=order_number,
                        link=order_link,
                    )
                )
                logger.info(
                    "Order-confirmation SMS sent to %s (sid=%s)",
                    phone_number,
                    sms_result.sid,
                )
            except SmsError as exc:
                logger.error(
                    "Failed to send order-confirmation SMS: code=%s msg=%s",
                    getattr(exc, "error_code", None),
                    exc,
                )
        else:
            surface = "chat_ui" if not session.call_sid else "twilio"
            logger.info(
                "phone_number_unavailable",
                extra={
                    "event_name": "phone_number_unavailable",
                    "surface": surface,
                    "consumer": "checkout_service.handle_payment_completed",
                    "session_token": session.token,
                },
            )

        # ---- live-call announcement --------------------------------
        if self.live_call_service.announce_order_completed(
            call_sid=session.call_sid,
            order_number=session.order_number,
        ):
            logger.info(
                "Live call completion announcement sent for call_sid=%s order=%s",
                session.call_sid,
                session.order_number,
            )

        return session

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upsert_checkout_session(
        self,
        *,
        restaurant_id: str,
        call_sid: str | None,
        order_number: str | None,
        customer_phone_number: str | None,
        address_required: bool,
        area: str | None,
        postal_code: str | None,
        order_summary: dict[str, Any] | None,
        house_number: str | None,
        street: str | None,
        secondary_address: str | None,
        city: str | None,
        state: str | None,
        full_address_raw: str | None,
        address_source: str | None,
    ) -> CheckoutSession:
        session = (
            self.repository.find_session_by_order_number(order_number)
            if order_number
            else None
        )
        if session is None:
            session = self.create_session(
                restaurant_id=restaurant_id,
                call_sid=call_sid,
                order_number=order_number,
                customer_phone_number=customer_phone_number,
                address_required=address_required,
                area=area,
                postal_code=postal_code,
                order_summary=order_summary,
            )
        else:
            if call_sid:
                session.call_sid = call_sid
            if customer_phone_number:
                session.customer_phone_number = customer_phone_number
            session.address_required = address_required
            if area:
                session.area = area
            if postal_code:
                session.postal_code = postal_code
            if order_summary:
                session.order_summary = order_summary

        if house_number:
            session.house_number = house_number.strip()
        if street:
            session.street = street.strip()
        if secondary_address is not None:
            session.secondary_address = secondary_address.strip() or None
        if city:
            session.city = city.strip()
        if state:
            session.state = state.strip()
        if full_address_raw:
            session.full_address_raw = full_address_raw.strip()
        if address_source:
            session.address_source = address_source

        session.confirmation_link = self._resolve_confirmation_link(
            restaurant_id=restaurant_id,
            order_number=session.order_number,
        )

        if not session.address_required or (session.house_number and session.street):
            session.mark_address_completed()

        self.repository.save_session(session)
        return session

    def _resolve_confirmation_link(
        self,
        *,
        restaurant_id: str,
        order_number: str | None,
    ) -> str | None:
        env_template = os.getenv("COMPASS_ORDER_CONFIRMATION_LINK_TEMPLATE", "").strip()
        if env_template:
            return self._format_link_template(env_template, order_number=order_number)

        restaurant_config = self._load_restaurant_config(restaurant_id)
        links = restaurant_config.get("links") or {}
        contact = restaurant_config.get("contact") or {}

        for candidate in (
            links.get("order_confirmation"),
            links.get("confirmation"),
            links.get("menu"),
            contact.get("website"),
        ):
            resolved = self._format_link_template(candidate, order_number=order_number)
            if resolved:
                return resolved

        return None

    def _load_restaurant_config(self, restaurant_id: str) -> dict[str, Any]:
        path = RESTAURANT_DATA_ROOT / restaurant_id / "restaurant.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse restaurant config %s: %s", path.name, exc)
            return {}

    def _format_link_template(
        self, template: str | None, *, order_number: str | None
    ) -> str | None:
        value = (template or "").strip()
        if not value:
            return None
        try:
            return value.format(order_number=order_number or "")
        except Exception:
            return value

    def _reverse_geocode(self, *, latitude: float, longitude: float) -> dict[str, Any]:
        params = urlencode(
            {
                "lat": latitude,
                "lon": longitude,
                "format": "jsonv2",
                "addressdetails": 1,
            }
        )
        url = f"{REVERSE_GEOCODE_URL}?{params}"
        request = Request(
            url,
            headers={
                "User-Agent": REVERSE_GEOCODE_USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ReverseGeocodeError(f"Reverse geocoding failed: {exc}") from exc

        address = payload.get("address") or {}
        return {
            "house_number": address.get("house_number") or address.get("house"),
            "street": (
                address.get("road")
                or address.get("pedestrian")
                or address.get("street")
                or address.get("residential")
                or address.get("neighbourhood")
            ),
            "area": (
                address.get("suburb")
                or address.get("neighbourhood")
                or address.get("city_district")
                or address.get("town")
                or address.get("city")
                or address.get("county")
            ),
            "city": address.get("city") or address.get("town") or address.get("village"),
            "state": address.get("state"),
            "postal_code": address.get("postcode"),
            "display_name": payload.get("display_name"),
        }

    def _payment_amount_from_summary(
        self, order_summary: dict[str, Any] | None
    ) -> str:
        summary = order_summary or {}
        for key in ("total", "total_price", "grand_total"):
            raw_value = summary.get(key)
            if raw_value:
                return str(raw_value).replace("$", "").strip()
        return "0.00"
