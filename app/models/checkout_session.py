# app/models/checkout_session.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
import secrets


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def default_expiry(hours: int = 2) -> datetime:
    return utc_now() + timedelta(hours=hours)


@dataclass(slots=True)
class CheckoutSession:
    token: str
    restaurant_id: str
    call_sid: str | None = None
    order_number: str | None = None
    customer_phone_number: str | None = None
    address_required: bool = False
    confirmation_link: str | None = None

    order_summary: dict[str, Any] = field(default_factory=dict)

    area: str | None = None
    postal_code: str | None = None
    house_number: str | None = None
    street: str | None = None
    secondary_address: str | None = None
    city: str | None = None
    state: str | None = None
    full_address_raw: str | None = None
    address_source: str | None = None  # "manual" | "device_location"

    latitude: float | None = None
    longitude: float | None = None
    location_permission_granted: bool | None = None

    address_completed: bool = False
    payment_started: bool = False
    payment_completed: bool = False
    payment_reference: str | None = None
    payment_provider: str = "datacap"
    can_retry_payment: bool = False
    last_payment_status: str | None = None

    status: str = "pending_address"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    expires_at: datetime = field(default_factory=default_expiry)

    @classmethod
    def new(
        cls,
        *,
        restaurant_id: str,
        call_sid: str | None,
        order_number: str | None,
        customer_phone_number: str | None,
        address_required: bool,
        area: str | None,
        postal_code: str | None,
        order_summary: dict[str, Any] | None = None,
    ) -> "CheckoutSession":
        return cls(
            token=secrets.token_urlsafe(24),
            restaurant_id=restaurant_id,
            call_sid=call_sid,
            order_number=order_number,
            customer_phone_number=customer_phone_number,
            address_required=address_required,
            area=area,
            postal_code=postal_code,
            order_summary=order_summary or {},
        )

    def is_expired(self) -> bool:
        return utc_now() >= self.expires_at

    def touch(self) -> None:
        self.updated_at = utc_now()

    def mark_address_completed(self) -> None:
        self.address_completed = True
        self.status = "pending_payment"
        self.touch()

    def mark_payment_started(self) -> None:
        self.payment_started = True
        self.can_retry_payment = False
        self.last_payment_status = None
        self.status = "payment_started"
        self.touch()

    def mark_payment_completed(self, reference: str | None = None) -> None:
        self.payment_started = False
        self.payment_completed = True
        self.can_retry_payment = False
        self.last_payment_status = "completed"
        self.payment_reference = reference
        self.status = "completed"
        self.touch()

    def mark_payment_retryable(self, status: str | None = None) -> None:
        self.payment_started = False
        self.payment_completed = False
        self.can_retry_payment = True
        self.last_payment_status = (status or "").strip() or self.last_payment_status
        self.status = "pending_payment_retry"
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "restaurant_id": self.restaurant_id,
            "call_sid": self.call_sid,
            "order_number": self.order_number,
            "customer_phone_number": self.customer_phone_number,
            "address_required": self.address_required,
            "confirmation_link": self.confirmation_link,
            "order_summary": self.order_summary,
            "area": self.area,
            "postal_code": self.postal_code,
            "house_number": self.house_number,
            "street": self.street,
            "secondary_address": self.secondary_address,
            "city": self.city,
            "state": self.state,
            "full_address_raw": self.full_address_raw,
            "address_source": self.address_source,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_permission_granted": self.location_permission_granted,
            "address_completed": self.address_completed,
            "payment_started": self.payment_started,
            "payment_completed": self.payment_completed,
            "payment_reference": self.payment_reference,
            "payment_provider": self.payment_provider,
            "can_retry_payment": self.can_retry_payment,
            "last_payment_status": self.last_payment_status,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckoutSession":
        return cls(
            token=data["token"],
            restaurant_id=data["restaurant_id"],
            call_sid=data.get("call_sid"),
            order_number=data.get("order_number"),
            customer_phone_number=data.get("customer_phone_number"),
            address_required=bool(data.get("address_required", False)),
            confirmation_link=data.get("confirmation_link"),
            order_summary=data.get("order_summary") or {},
            area=data.get("area"),
            postal_code=data.get("postal_code"),
            house_number=data.get("house_number"),
            street=data.get("street"),
            secondary_address=data.get("secondary_address"),
            city=data.get("city"),
            state=data.get("state"),
            full_address_raw=data.get("full_address_raw"),
            address_source=data.get("address_source"),
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            location_permission_granted=data.get("location_permission_granted"),
            address_completed=bool(data.get("address_completed", False)),
            payment_started=bool(data.get("payment_started", False)),
            payment_completed=bool(data.get("payment_completed", False)),
            payment_reference=data.get("payment_reference"),
            payment_provider=data.get("payment_provider", "datacap"),
            can_retry_payment=bool(data.get("can_retry_payment", False)),
            last_payment_status=data.get("last_payment_status"),
            status=data.get("status", "pending_address"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
        )
