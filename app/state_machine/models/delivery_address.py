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
    feedback_link: Optional[str] = None
    menu_link: Optional[str] = None

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
        self.address_form_link = None

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
            "feedback_link": self.feedback_link,
            "menu_link": self.menu_link,
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
            feedback_link=data.get("feedback_link"),
            menu_link=data.get("menu_link"),
        )