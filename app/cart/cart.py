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
        """
        Adds a fully specified CartItem to the cart.
        """
        self._items.append(item)

    def remove_item(self, cart_item_id: str) -> bool:
        """
        Removes an item from the cart by cart_item_id.
        Returns True if removed, False if not found.
        """
        for i, item in enumerate(self._items):
            if item.cart_item_id == cart_item_id:
                del self._items[i]
                return True
        return False

    def clear(self) -> None:
        """
        Empties the cart completely.
        """
        self._items.clear()
