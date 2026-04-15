# app/models/payment_link_session.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import secrets


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class PaymentLinkSession:
    id: str = field(default_factory=lambda: secrets.token_urlsafe(18))
    checkout_token: str = ""
    invoice_no: str = "12345"
    amount: str = ""
    request_id: str | None = None
    public_link_id: str | None = None
    public_link_url: str | None = None
    public_link_embedded_url: str | None = None
    public_link_qr_code_url: str | None = None
    status: str = "created"
    payment_type_used: str | None = None
    provider_reference: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "checkout_token": self.checkout_token,
            "invoice_no": self.invoice_no,
            "amount": self.amount,
            "request_id": self.request_id,
            "public_link_id": self.public_link_id,
            "public_link_url": self.public_link_url,
            "public_link_embedded_url": self.public_link_embedded_url,
            "public_link_qr_code_url": self.public_link_qr_code_url,
            "status": self.status,
            "payment_type_used": self.payment_type_used,
            "provider_reference": self.provider_reference,
            "raw_response": self.raw_response,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaymentLinkSession":
        return cls(
            id=data["id"],
            checkout_token=data["checkout_token"],
            invoice_no=data["invoice_no"],
            amount=data["amount"],
            request_id=data.get("request_id"),
            public_link_id=data.get("public_link_id"),
            public_link_url=data.get("public_link_url"),
            public_link_embedded_url=data.get("public_link_embedded_url"),
            public_link_qr_code_url=data.get("public_link_qr_code_url"),
            status=data.get("status", "created"),
            payment_type_used=data.get("payment_type_used"),
            provider_reference=data.get("provider_reference"),
            raw_response=data.get("raw_response") or {},
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )