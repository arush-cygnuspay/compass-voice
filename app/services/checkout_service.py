from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.models.checkout_session import CheckoutSession
from app.models.payment_link_session import PaymentLinkSession
from app.services.live_call_service import LiveCallService
from app.services.payment.datacap_payment_links_service import DatacapPaymentLinksService
from app.services.sms_service import SmsSendRequest, SmsService

logger = logging.getLogger(__name__)


PAYMENT_POLL_INTERVAL_SECONDS = float(os.getenv("COMPASS_PAYMENT_POLL_INTERVAL", "6"))
PAYMENT_POLL_MAX_DURATION_SECONDS = float(
    os.getenv("COMPASS_PAYMENT_POLL_MAX_DURATION", "900")
)

CHECKOUT_DATA_DIR = Path(
    os.getenv("COMPASS_CHECKOUT_DATA_DIR", "app/data/checkout_sessions")
)
CHECKOUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

PAYMENT_LINK_SESSION_DATA_DIR = Path(
    os.getenv("COMPASS_PAYMENT_LINK_SESSION_DATA_DIR", "app/data/payment_link_sessions")
)
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

PUBLIC_CHECKOUT_BASE_URL = os.getenv(
    "COMPASS_PUBLIC_CHECKOUT_BASE_URL",
    "https://voice.cygnuscompass.com/checkout",
).rstrip("/")

REVERSE_GEOCODE_URL = os.getenv(
    "COMPASS_REVERSE_GEOCODE_URL",
    "https://nominatim.openstreetmap.org/reverse",
).strip()
REVERSE_GEOCODE_USER_AGENT = os.getenv(
    "COMPASS_REVERSE_GEOCODE_USER_AGENT",
    "CompassCheckout/1.0 (support@cygnuspayments.com)",
).strip()

PAYMENT_FAILURE_STATUSES = {"cancelled", "canceled", "expired", "failed", "declined"}
RESTAURANT_DATA_ROOT = Path("app/data/restaurants")


class CheckoutNotFoundError(Exception):
    pass


class CheckoutExpiredError(Exception):
    pass


class ReverseGeocodeError(Exception):
    pass


def _lookup_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CheckoutService:
    def __init__(self) -> None:
        self.data_dir = CHECKOUT_DATA_DIR
        self.payment_link_session_dir = PAYMENT_LINK_SESSION_DATA_DIR
        self.payment_provider = DatacapPaymentLinksService()
        self.sms_service = SmsService()
        self.live_call_service = LiveCallService()
        self._active_pollers: set[str] = set()
        self._pollers_lock = threading.Lock()

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
        self.save_session(session)
        return session

    def build_checkout_url(self, token: str) -> str:
        return f"{PUBLIC_CHECKOUT_BASE_URL}/{token}"

    def _path_for_token(self, token: str) -> Path:
        return self.data_dir / f"{token}.json"

    def _payment_link_session_path(self, session_id: str) -> Path:
        return self.payment_link_session_dir / f"{session_id}.json"

    def _order_index_path(self, order_number: str) -> Path:
        return CHECKOUT_ORDER_INDEX_DIR / f"{_lookup_key(order_number)}.json"

    def _payment_link_latest_index_path(self, checkout_token: str) -> Path:
        return PAYMENT_LINK_BY_CHECKOUT_INDEX_DIR / f"{_lookup_key(checkout_token)}.json"

    def _payment_link_request_index_path(self, request_id: str) -> Path:
        return PAYMENT_LINK_BY_REQUEST_INDEX_DIR / f"{_lookup_key(request_id)}.json"

    def _write_json(
        self,
        path: Path,
        payload: dict[str, Any],
        *,
        compact: bool = False,
    ) -> None:
        if compact:
            text = json.dumps(payload, separators=(",", ":"))
        else:
            text = json.dumps(payload, indent=2)
        path.write_text(text, encoding="utf-8")

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse JSON file %s: %s", path.name, exc)
            return None

    def _restaurant_config_path(self, restaurant_id: str) -> Path:
        return RESTAURANT_DATA_ROOT / restaurant_id / "restaurant.json"

    def _load_restaurant_config(self, restaurant_id: str) -> dict[str, Any]:
        return self._read_json(self._restaurant_config_path(restaurant_id)) or {}

    def _format_link_template(self, template: str | None, *, order_number: str | None) -> str | None:
        value = (template or "").strip()
        if not value:
            return None
        try:
            return value.format(order_number=order_number or "")
        except Exception:
            return value

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

    def _load_checkout_session_file(self, path: Path) -> CheckoutSession | None:
        data = self._read_json(path)
        if not data:
            return None
        try:
            return CheckoutSession.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse checkout session %s: %s", path.name, exc)
            return None

    def _load_payment_link_session_file(self, path: Path) -> PaymentLinkSession | None:
        data = self._read_json(path)
        if not data:
            return None
        try:
            return PaymentLinkSession.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse payment link session %s: %s", path.name, exc)
            return None

    def _save_session_indexes(self, session: CheckoutSession) -> None:
        if not session.order_number:
            return
        self._write_json(
            self._order_index_path(session.order_number),
            {
                "order_number": session.order_number,
                "token": session.token,
                "updated_at": session.updated_at.isoformat(),
            },
            compact=True,
        )

    def _save_payment_link_indexes(self, payment_link_session: PaymentLinkSession) -> None:
        if payment_link_session.checkout_token:
            self._write_json(
                self._payment_link_latest_index_path(payment_link_session.checkout_token),
                {
                    "checkout_token": payment_link_session.checkout_token,
                    "session_id": payment_link_session.id,
                    "updated_at": payment_link_session.updated_at.isoformat(),
                },
                compact=True,
            )

        if payment_link_session.request_id:
            self._write_json(
                self._payment_link_request_index_path(payment_link_session.request_id),
                {
                    "request_id": payment_link_session.request_id,
                    "session_id": payment_link_session.id,
                    "updated_at": payment_link_session.updated_at.isoformat(),
                },
                compact=True,
            )

    def save_session(self, session: CheckoutSession) -> None:
        session.touch()
        self._write_json(self._path_for_token(session.token), session.to_dict())
        self._save_session_indexes(session)

    def save_payment_link_session(self, payment_link_session: PaymentLinkSession) -> None:
        payment_link_session.touch()
        self._write_json(
            self._payment_link_session_path(payment_link_session.id),
            payment_link_session.to_dict(),
        )
        self._save_payment_link_indexes(payment_link_session)

    def serialize_session(self, session: CheckoutSession) -> dict[str, Any]:
        payload = session.to_dict()
        latest_payment_link = self._find_latest_payment_link_session(session.token)
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

    def get_session(self, token: str) -> CheckoutSession:
        path = self._path_for_token(token)
        session = self._load_checkout_session_file(path)
        if session is None:
            raise CheckoutNotFoundError(token)

        if session.is_expired():
            raise CheckoutExpiredError(token)

        return session

    def update_address(
        self,
        *,
        token: str,
        house_number: str,
        street: str,
        secondary_address: str | None,
    ) -> CheckoutSession:
        session = self.get_session(token)
        session.house_number = house_number.strip()
        session.street = street.strip()
        session.secondary_address = (secondary_address or "").strip() or None
        session.address_source = "manual"
        session.mark_address_completed()
        self.save_session(session)
        self._sync_voice_session_from_checkout(session)
        return session

    def update_location(
        self,
        *,
        token: str,
        latitude: float | None,
        longitude: float | None,
        permission_granted: bool | None,
    ) -> CheckoutSession:
        session = self.get_session(token)
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

        self.save_session(session)
        self._sync_voice_session_from_checkout(session)
        return session

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

    def _payment_amount_from_summary(self, order_summary: dict[str, Any] | None) -> str:
        summary = order_summary or {}
        for key in ("total", "total_price", "grand_total"):
            raw_value = summary.get(key)
            if raw_value:
                return str(raw_value).replace("$", "").strip()
        return "0.00"

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
        session = self.find_session_by_order_number(order_number) if order_number else None
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

        self.save_session(session)
        return session

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

        latest = self._find_latest_payment_link_session(session.token)
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
                    latest = self._find_latest_payment_link_session(session.token)
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
                    self.save_session(session)
                self._start_payment_poller(session.token)
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
        session = self.get_session(token)

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
        self.save_session(session)
        self.save_payment_link_session(payment_link_session)
        logger.info(
            "Started payment link for order=%s token=%s request_id=%s",
            session.order_number,
            session.token,
            payment_link_session.request_id,
        )
        self._start_payment_poller(session.token)

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

    def complete_checkout(
        self,
        token: str,
        *,
        payment_reference: str | None = None,
    ) -> CheckoutSession:
        session = self.get_session(token)
        session.mark_payment_completed(reference=payment_reference)
        self.save_session(session)
        self._sync_voice_session_from_checkout(session, mark_completed=True)
        return session

    def _scan_latest_payment_link_session(
        self,
        checkout_token: str,
    ) -> PaymentLinkSession | None:
        candidates: list[tuple[float, PaymentLinkSession]] = []
        for path in self.payment_link_session_dir.glob("*.json"):
            payment_link_session = self._load_payment_link_session_file(path)
            if payment_link_session is None:
                continue
            if payment_link_session.checkout_token == checkout_token:
                candidates.append(
                    (payment_link_session.created_at.timestamp(), payment_link_session)
                )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        latest = candidates[0][1]
        self._save_payment_link_indexes(latest)
        return latest

    def _find_latest_payment_link_session(
        self,
        checkout_token: str,
    ) -> PaymentLinkSession | None:
        indexed = self._read_json(self._payment_link_latest_index_path(checkout_token))
        if indexed:
            session_id = indexed.get("session_id")
            if session_id:
                session = self._load_payment_link_session_file(
                    self._payment_link_session_path(session_id)
                )
                if session and session.checkout_token == checkout_token:
                    return session

        return self._scan_latest_payment_link_session(checkout_token)

    def verify_payment_with_provider(self, token: str) -> dict[str, Any]:
        try:
            session = self.get_session(token)
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

        payment_link_session = self._find_latest_payment_link_session(token)
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
            self.save_payment_link_session(payment_link_session)

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
            self.save_session(session)
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
        session = self.find_session_by_order_number(order_number)
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

    def _start_payment_poller(self, token: str) -> None:
        with self._pollers_lock:
            if token in self._active_pollers:
                logger.info("Poller already running for token=%s - skipping.", token)
                return
            self._active_pollers.add(token)

        thread = threading.Thread(
            target=self._run_payment_poller,
            args=(token,),
            name=f"payment-poller-{token[:8]}",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Started Datacap payment poller for token=%s (interval=%.1fs, max=%.0fs)",
            token,
            PAYMENT_POLL_INTERVAL_SECONDS,
            PAYMENT_POLL_MAX_DURATION_SECONDS,
        )

    def _run_payment_poller(self, token: str) -> None:
        import time

        deadline = time.monotonic() + PAYMENT_POLL_MAX_DURATION_SECONDS

        try:
            while time.monotonic() < deadline:
                time.sleep(PAYMENT_POLL_INTERVAL_SECONDS)
                try:
                    result = self.verify_payment_with_provider(token)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "payment-poller: unexpected error for %s: %s", token, exc
                    )
                    continue

                if result.get("payment_completed"):
                    logger.info(
                        "payment-poller: payment confirmed for token=%s via Datacap.",
                        token,
                    )
                    return

                status_lower = str(result.get("status") or "").lower()
                if status_lower in PAYMENT_FAILURE_STATUSES:
                    logger.info(
                        "payment-poller: stopping for token=%s - status=%s",
                        token,
                        status_lower,
                    )
                    return

            logger.info(
                "payment-poller: timeout reached for token=%s after %ds",
                token,
                int(PAYMENT_POLL_MAX_DURATION_SECONDS),
            )
        finally:
            with self._pollers_lock:
                self._active_pollers.discard(token)

    def _scan_session_by_order_number(self, order_number: str) -> CheckoutSession | None:
        for path in self.data_dir.glob("*.json"):
            session = self._load_checkout_session_file(path)
            if session is None:
                continue
            if session.order_number == order_number and not session.is_expired():
                self._save_session_indexes(session)
                return session
        return None

    def find_session_by_order_number(self, order_number: str | None) -> CheckoutSession | None:
        if not order_number:
            return None

        indexed = self._read_json(self._order_index_path(order_number))
        if indexed:
            token = indexed.get("token")
            if token:
                try:
                    session = self.get_session(token)
                except (CheckoutNotFoundError, CheckoutExpiredError):
                    session = None
                if session and session.order_number == order_number:
                    return session

        return self._scan_session_by_order_number(order_number)

    def handle_payment_completed(
        self,
        *,
        order_number: str,
        payment_reference: str | None = None,
    ) -> CheckoutSession | None:
        session = self.find_session_by_order_number(order_number)
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
            self._sync_voice_session_from_checkout(session, mark_completed=True)
            return session

        session.mark_payment_completed(reference=payment_reference)
        self.save_session(session)
        self._sync_voice_session_from_checkout(session, mark_completed=True)
        logger.info("Payment completed for order %s (token=%s)", order_number, session.token)

        if session.customer_phone_number:
            # Prefer the actual payment/checkout link over the static
            # restaurant website so the customer can track their order.
            latest_payment = self._find_latest_payment_link_session(session.token)
            order_link = (
                (latest_payment.public_link_url if latest_payment else None)
                or session.confirmation_link
                or ""
            )
            sms_result = self.sms_service.send(
                SmsSendRequest(
                    template="order_confirmation",
                    phone_number=session.customer_phone_number,
                    order_number=order_number,
                    link=order_link,
                )
            )
            if sms_result.ok:
                logger.info(
                    "Order-confirmation SMS sent to %s (sid=%s)",
                    session.customer_phone_number,
                    sms_result.sid,
                )
            else:
                logger.error(
                    "Failed to send order-confirmation SMS: code=%s msg=%s",
                    sms_result.error_code,
                    sms_result.error_message,
                )
        else:
            logger.info(
                "No customer_phone_number on session %s - skipping SMS",
                session.token,
            )

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

    def _sync_voice_session_from_checkout(
        self,
        checkout_session: CheckoutSession,
        *,
        mark_completed: bool = False,
    ) -> None:
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

        latest_payment_link = self._find_latest_payment_link_session(checkout_session.token)
        if latest_payment_link and latest_payment_link.public_link_url:
            delivery.payment_link = latest_payment_link.public_link_url
            # Keep confirmation_link in sync with the actual checkout
            # URL so voice responses reference the real link, not the
            # static restaurant website.
            delivery.confirmation_link = latest_payment_link.public_link_url

        if checkout_session.address_completed:
            delivery.source = "sms_form"
            delivery.form_completed = True
            delivery.collected = True
            delivery.confirmed = True
            context.delivery_address_confirmed = True

        if mark_completed or checkout_session.payment_completed:
            context.reset()
            voice_session.cart.clear()
            voice_session.conversation_state = ConversationState.COMPLETED
            voice_session.last_response_key = "order_completed"
            voice_session.last_response_payload = {
                "order_number": checkout_session.order_number,
                "payment_reference": checkout_session.payment_reference,
            }

        save_session(voice_session)
