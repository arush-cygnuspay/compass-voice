# app/cart/cart_item.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid


@dataclass
class CartItem:
    """
    Represents a single item entry in the cart.
    Immutable once created (modification = replace).
    """

    cart_item_id: str
    item_id: str
    quantity: int

    # Optional configuration
    variant_id: Optional[str] = None

    # group_id -> list[item_id]
    sides: Dict[str, List[str]] = field(default_factory=dict)

    # group_id -> list[modifier_id]
    modifiers: Dict[str, List[str]] = field(default_factory=dict)

    @staticmethod
    def create(
        item_id: str,
        quantity: int,
        variant_id: Optional[str],
        sides: Dict[str, List[str]],
        modifiers: Dict[str, List[str]],
    ) -> "CartItem":
        return CartItem(
            cart_item_id=str(uuid.uuid4()),
            item_id=item_id,
            quantity=quantity,
            variant_id=variant_id,
            sides=sides.copy(),
            modifiers=modifiers.copy(),
        )

    # ---------------------------
    # Serialization
    # ---------------------------

    def to_dict(self) -> dict:
        return {
            "cart_item_id": self.cart_item_id,
            "item_id": self.item_id,
            "quantity": self.quantity,
            "variant_id": self.variant_id,
            "sides": self.sides,
            "modifiers": self.modifiers,
        }

    @staticmethod
    def from_dict(data: dict) -> "CartItem":
        return CartItem(
            cart_item_id=data["cart_item_id"],
            item_id=data["item_id"],
            quantity=data["quantity"],
            variant_id=data.get("variant_id"),
            sides=data.get("sides", {}),
            modifiers=data.get("modifiers", {}),
        )
