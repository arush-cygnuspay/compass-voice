# app/cart/cart.py

from typing import List, Optional
from app.cart.cart_item import CartItem


class Cart:
    """
    Aggregate root for the shopping cart.
    Owns all cart items and enforces cart-level rules.
    """

    def __init__(self) -> None:
        self._items: List[CartItem] = []

    # ---------------------------
    # Read operations
    # ---------------------------

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def get_items(self) -> List[CartItem]:
        return list(self._items)

    def get_item(self, cart_item_id: str) -> Optional[CartItem]:
        for item in self._items:
            if item.cart_item_id == cart_item_id:
                return item
        return None

    # ---------------------------
    # Write operations
    # ---------------------------

    def add_item(self, item: CartItem) -> None:
        self._items.append(item)

    def remove_item(self, cart_item_id: str) -> bool:
        for i, item in enumerate(self._items):
            if item.cart_item_id == cart_item_id:
                del self._items[i]
                return True
        return False

    def clear(self) -> None:
        self._items.clear()

    # ---------------------------
    # Serialization
    # ---------------------------

    def to_dict(self) -> dict:
        """
        Serialize Cart aggregate into plain dict.
        """
        return {
            "items": [item.to_dict() for item in self._items]
        }

    @staticmethod
    def from_dict(data: dict) -> "Cart":
        cart = Cart()
        for item_data in data.get("items", []):
            cart.add_item(CartItem.from_dict(item_data))
        return cart
