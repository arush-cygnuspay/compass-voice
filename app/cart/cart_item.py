# app/cart/cart_item.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
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

    # side item_id -> variant_id
    side_variants: Dict[str, str] = field(default_factory=dict)

    # group_id -> list[modifier_id] or richer modifier selection entries
    modifiers: Dict[str, List[Any]] = field(default_factory=dict)

    @staticmethod
    def create(
        item_id: str,
        quantity: int,
        variant_id: Optional[str],
        sides: Dict[str, List[str]],
        side_variants: Dict[str, str],
        modifiers: Dict[str, List[Any]],
    ) -> "CartItem":
        return CartItem(
            cart_item_id=str(uuid.uuid4()),
            item_id=item_id,
            quantity=quantity,
            variant_id=variant_id,
            sides={group_id: list(item_ids) for group_id, item_ids in sides.items()},
            side_variants=dict(side_variants),
            modifiers={group_id: list(modifier_ids) for group_id, modifier_ids in modifiers.items()},
        )

    def to_dict(self) -> dict:
        return {
            "cart_item_id": self.cart_item_id,
            "item_id": self.item_id,
            "quantity": self.quantity,
            "variant_id": self.variant_id,
            "sides": self.sides,
            "side_variants": self.side_variants,
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
            side_variants=data.get("side_variants", {}),
            modifiers=data.get("modifiers", {}),
        )
