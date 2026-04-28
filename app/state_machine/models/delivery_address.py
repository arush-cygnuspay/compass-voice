# app/state_machine/models/delivery_address.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class DeliveryAddress:
    area: Optional[str] = None
    area_serviceable: Optional[bool] = None
    postal_code: Optional[str] = None

    house_number: Optional[str] = None
    street: Optional[str] = None
    secondary_address: Optional[str] = None  # apartment / suite / unit / building / block
    city: Optional[str] = None
    state: Optional[str] = None
    full_address_raw: Optional[str] = None

    source: Optional[str] = None  # "voice" | "sms_form"
    form_sent: bool = False
    form_completed: bool = False
    collected: bool = False
    confirmed: bool = False

    checkout_link_send_attempts: int = 0
    payment_link_send_attempts: int = 0

    customer_phone_number: Optional[str] = None
    order_number: Optional[str] = None
    payment_link: Optional[str] = None
    address_form_link: Optional[str] = None
    confirmation_link: Optional[str] = None
    feedback_link: Optional[str] = None
    menu_link: Optional[str] = None
    checkout_status: Optional[str] = None
    payment_status: Optional[str] = None
    payment_reference: Optional[str] = None
    payment_wait_mode: Optional[str] = None
    payment_session_state: Optional[str] = None
    last_payment_link_resend_at_epoch: Optional[float] = None
    last_checkout_link_resend_at_epoch: Optional[float] = None
    last_checkout_wait_prompt_at_epoch: Optional[float] = None
    last_checkout_wait_response_key: Optional[str] = None
    last_payment_wait_prompt_at_epoch: Optional[float] = None
    last_payment_wait_response_key: Optional[str] = None
    payment_status_last_prompt_at_epoch: Optional[float] = None
    payment_status_last_response_key: Optional[str] = None

    # "sms" if a phone number is available and SMS delivery is used,
    # "in_session" when there is no phone number and the link must be
    # surfaced inline (chat UI / TTS read-out).
    payment_link_delivery_channel: Optional[str] = None

    @property
    def has_phone_number(self) -> bool:
        value = (self.customer_phone_number or "").strip()
        return bool(value)

    def normalized_phone_number(self) -> Optional[str]:
        """Return the digits-only phone number, or None if missing/invalid."""
        raw = (self.customer_phone_number or "").strip()
        if not raw:
            return None
        digits = "".join(ch for ch in raw if ch.isdigit())
        return digits or None

    def reset_for_new_delivery(self) -> None:
        self.area = None
        self.area_serviceable = None
        self.postal_code = None
        self.house_number = None
        self.street = None
        self.secondary_address = None
        self.city = None
        self.state = None
        self.full_address_raw = None
        self.source = None
        self.form_sent = False
        self.form_completed = False
        self.collected = False
        self.confirmed = False
        self.checkout_link_send_attempts = 0
        self.payment_link_send_attempts = 0
        self.order_number = None
        self.payment_link = None
        self.address_form_link = None
        self.confirmation_link = None
        self.feedback_link = None
        self.menu_link = None
        self.checkout_status = None
        self.payment_status = None
        self.payment_reference = None
        self.payment_wait_mode = None
        self.payment_session_state = None
        self.last_payment_link_resend_at_epoch = None
        self.last_checkout_link_resend_at_epoch = None
        self.last_checkout_wait_prompt_at_epoch = None
        self.last_checkout_wait_response_key = None
        self.last_payment_wait_prompt_at_epoch = None
        self.last_payment_wait_response_key = None
        self.payment_status_last_prompt_at_epoch = None
        self.payment_status_last_response_key = None
        self.payment_link_delivery_channel = None

    def missing_eligibility_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.area:
            missing.append("area")
        if not self.postal_code:
            missing.append("postal_code")
        return missing

    def missing_fulfillment_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.house_number:
            missing.append("house_number")
        if not self.street:
            missing.append("street")
        return missing

    def has_eligibility_seed(self) -> bool:
        return not self.missing_eligibility_fields()

    def has_minimum_fulfillment_address(self) -> bool:
        return not self.missing_fulfillment_fields()

    def to_dict(self) -> dict:
        return {
            "area": self.area,
            "area_serviceable": self.area_serviceable,
            "postal_code": self.postal_code,
            "house_number": self.house_number,
            "street": self.street,
            "secondary_address": self.secondary_address,
            "city": self.city,
            "state": self.state,
            "full_address_raw": self.full_address_raw,
            "source": self.source,
            "form_sent": self.form_sent,
            "form_completed": self.form_completed,
            "collected": self.collected,
            "confirmed": self.confirmed,
            "checkout_link_send_attempts": self.checkout_link_send_attempts,
            "payment_link_send_attempts": self.payment_link_send_attempts,
            "customer_phone_number": self.customer_phone_number,
            "order_number": self.order_number,
            "payment_link": self.payment_link,
            "address_form_link": self.address_form_link,
            "confirmation_link": self.confirmation_link,
            "feedback_link": self.feedback_link,
            "menu_link": self.menu_link,
            "checkout_status": self.checkout_status,
            "payment_status": self.payment_status,
            "payment_reference": self.payment_reference,
            "payment_wait_mode": self.payment_wait_mode,
            "payment_session_state": self.payment_session_state,
            "last_payment_link_resend_at_epoch": self.last_payment_link_resend_at_epoch,
            "last_checkout_link_resend_at_epoch": self.last_checkout_link_resend_at_epoch,
            "last_checkout_wait_prompt_at_epoch": self.last_checkout_wait_prompt_at_epoch,
            "last_checkout_wait_response_key": self.last_checkout_wait_response_key,
            "last_payment_wait_prompt_at_epoch": self.last_payment_wait_prompt_at_epoch,
            "last_payment_wait_response_key": self.last_payment_wait_response_key,
            "payment_status_last_prompt_at_epoch": self.payment_status_last_prompt_at_epoch,
            "payment_status_last_response_key": self.payment_status_last_response_key,
            "payment_link_delivery_channel": self.payment_link_delivery_channel,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "DeliveryAddress":
        data = data or {}
        return cls(
            area=data.get("area"),
            area_serviceable=(
                bool(data["area_serviceable"])
                if data.get("area_serviceable") is not None
                else None
            ),
            postal_code=data.get("postal_code"),
            house_number=data.get("house_number"),
            street=data.get("street"),
            secondary_address=data.get("secondary_address"),
            city=data.get("city"),
            state=data.get("state"),
            full_address_raw=data.get("full_address_raw"),
            source=data.get("source"),
            form_sent=bool(data.get("form_sent", False)),
            form_completed=bool(data.get("form_completed", False)),
            collected=bool(data.get("collected", False)),
            confirmed=bool(data.get("confirmed", False)),
            checkout_link_send_attempts=int(data.get("checkout_link_send_attempts", 0) or 0),
            payment_link_send_attempts=int(data.get("payment_link_send_attempts", 0) or 0),
            customer_phone_number=data.get("customer_phone_number"),
            order_number=data.get("order_number"),
            payment_link=data.get("payment_link"),
            address_form_link=data.get("address_form_link"),
            confirmation_link=data.get("confirmation_link"),
            feedback_link=data.get("feedback_link"),
            menu_link=data.get("menu_link"),
            checkout_status=data.get("checkout_status"),
            payment_status=data.get("payment_status"),
            payment_reference=data.get("payment_reference"),
            payment_wait_mode=data.get("payment_wait_mode"),
            payment_session_state=data.get("payment_session_state"),
            last_payment_link_resend_at_epoch=data.get("last_payment_link_resend_at_epoch"),
            last_checkout_link_resend_at_epoch=data.get("last_checkout_link_resend_at_epoch"),
            last_checkout_wait_prompt_at_epoch=data.get("last_checkout_wait_prompt_at_epoch"),
            last_checkout_wait_response_key=data.get("last_checkout_wait_response_key"),
            last_payment_wait_prompt_at_epoch=data.get("last_payment_wait_prompt_at_epoch"),
            last_payment_wait_response_key=data.get("last_payment_wait_response_key"),
            payment_status_last_prompt_at_epoch=data.get("payment_status_last_prompt_at_epoch"),
            payment_status_last_response_key=data.get("payment_status_last_response_key"),
            payment_link_delivery_channel=data.get("payment_link_delivery_channel"),
        )
