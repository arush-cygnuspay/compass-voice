# app/domain/order_draft.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.state_machine.models.conversation_context import ConversationContext


@dataclass(frozen=True)
class OrderDraft:
    """Immutable snapshot of order-level fields from ConversationContext.

    Used when downstream code needs to read order intent without holding
    a reference to the mutable ConversationContext.
    """

    order_type: Optional[str]
    area: Optional[str]
    postal_code: Optional[str]
    house_number: Optional[str]
    street: Optional[str]
    secondary_address: Optional[str]
    city: Optional[str]
    customer_phone_number: Optional[str]
    order_number: Optional[str]
    delivery_address_confirmed: bool

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_context(cls, ctx: ConversationContext) -> OrderDraft:
        addr = ctx.delivery_address
        return cls(
            order_type=ctx.order_type,
            area=addr.area if addr else None,
            postal_code=addr.postal_code if addr else None,
            house_number=addr.house_number if addr else None,
            street=addr.street if addr else None,
            secondary_address=addr.secondary_address if addr else None,
            city=addr.city if addr else None,
            customer_phone_number=addr.customer_phone_number if addr else None,
            order_number=addr.order_number if addr else None,
            delivery_address_confirmed=getattr(ctx, "delivery_address_confirmed", False),
        )

    # ── derived properties ───────────────────────────────────────────────

    @property
    def is_delivery(self) -> bool:
        return self.order_type == "delivery"

    @property
    def is_pickup(self) -> bool:
        return self.order_type == "pickup"

    @property
    def has_address(self) -> bool:
        return bool(self.house_number or self.street)
