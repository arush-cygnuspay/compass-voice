# app/services/sms_service.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client


DEFAULT_SMS_OVERRIDE_TO = "+923204711572"


SmsTemplate = Literal[
    "payment_link",
    "checkout_link",
    "feedback_link",
    "menu_link",
    "delivery_area_confirmed",
    "order_confirmation",
]


@dataclass(frozen=True, slots=True)
class SmsSendRequest:
    template: SmsTemplate
    phone_number: str
    order_number: str = ""
    link: str = ""
    area: str = ""


@dataclass(frozen=True, slots=True)
class SmsSendResult:
    ok: bool
    sid: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class SmsService:
    def __init__(self) -> None:
        self._account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self._auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        self._from_number = os.getenv("TWILIO_SMS_FROM_NUMBER", "").strip()
        self._override_to_number = os.getenv(
            "COMPASS_SMS_OVERRIDE_TO",
            DEFAULT_SMS_OVERRIDE_TO,
        ).strip()

        self._client: Client | None = None
        if self._account_sid and self._auth_token:
            self._client = Client(self._account_sid, self._auth_token)

    def is_configured(self) -> bool:
        return bool(self._client and self._from_number)

    def send(self, request: SmsSendRequest) -> SmsSendResult:
        if not self.is_configured():
            return SmsSendResult(
                ok=False,
                error_code="sms_not_configured",
                error_message="Twilio SMS is not configured.",
            )

        body = self._build_body(request)
        to_number = self._resolve_to_number(request.phone_number)

        try:
            message = self._client.messages.create(
                body=body,
                from_=self._from_number,
                to=to_number,
            )
            return SmsSendResult(
                ok=True,
                sid=getattr(message, "sid", None),
            )
        except TwilioRestException as exc:
            return SmsSendResult(
                ok=False,
                error_code=str(getattr(exc, "code", "") or "twilio_rest_error"),
                error_message=str(exc),
            )
        except Exception as exc:
            return SmsSendResult(
                ok=False,
                error_code="sms_send_failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    def _resolve_to_number(self, requested_phone_number: str) -> str:
        if self._override_to_number:
            return self._override_to_number
        return requested_phone_number

    def _build_body(self, request: SmsSendRequest) -> str:
        if request.template == "payment_link":
            return (
                f"Compass Order #{request.order_number}\n\n"
                f"Please complete your payment securely using the link below:\n"
                f"{request.link}\n\n"
                f"Reply to this message if you need assistance."
            )

        if request.template == "checkout_link":
            return (
                f"Compass Order #{request.order_number}\n\n"
                f"Please complete your checkout using the secure link below:\n"
                f"{request.link}\n\n"
                f"Reply to this message if you need assistance."
            )

        if request.template == "feedback_link":
            return (
                f"Compass Order #{request.order_number}\n\n"
                f"Thank you for your order. We’d appreciate your feedback:\n"
                f"{request.link}"
            )

        if request.template == "menu_link":
            return f"Compass Menu\n\nView our menu here:\n{request.link}"

        if request.template == "delivery_area_confirmed":
            area_text = request.area or "your area"
            return f"Compass Delivery\n\nWe deliver to {area_text}."

        if request.template == "order_confirmation":
            body = (
                f"✅ Compass Order #{request.order_number}\n\n"
                f"Your order has been placed successfully!\n"
                f"Payment received — we're getting it ready for you.\n\n"
                f"Thank you for ordering with us. 🙏"
            )
            if request.link:
                body += f"\n\nTrack your order here:\n{request.link}"
            return body

        raise ValueError(f"Unsupported SMS template: {request.template}")
