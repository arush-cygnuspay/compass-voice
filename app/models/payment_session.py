# app/models/payment_session.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import secrets


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class PaymentSession:
    payment_session_id: str = field(default_factory=lambda: secrets.token_urlsafe(18))
    checkout_token: str = ""
    provider: str = "datacap"
    mode: str = "debug"
    status: str = "created"  # created | pending_redirect | pending_provider | completed | failed

    amount: str = ""
    currency: str = "USD"

    provider_base_url: str | None = None
    provider_redirect_url: str | None = None
    provider_reference: str | None = None
    provider_payload: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def touch(self) -> None:
        self.updated_at = utc_now()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "payment_session_id": self.payment_session_id,
            "checkout_token": self.checkout_token,
            "provider": self.provider,
            "mode": self.mode,
            "status": self.status,
            "amount": self.amount,
            "currency": self.currency,
            "provider_base_url": self.provider_base_url,
            "provider_redirect_url": self.provider_redirect_url,
            "provider_reference": self.provider_reference,
            "provider_payload": self.provider_payload,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }