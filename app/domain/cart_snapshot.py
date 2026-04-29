# app/domain/cart_snapshot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.cart.cart import Cart
    from app.cart.cart_item import CartItem


@dataclass(frozen=True)
class CartSnapshot:
    """Immutable point-in-time view of a Cart.

    Passed across layer boundaries so downstream code cannot mutate
    the live cart through a stale reference.
    """

    items: tuple[CartItem, ...]

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_cart(cls, cart: Cart) -> CartSnapshot:
        from app.cart.cart import Cart as _Cart  # noqa: F401 — runtime import only
        return cls(items=tuple(cart.get_items()))

    # ── read helpers ─────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def get_items(self) -> tuple[CartItem, ...]:
        return self.items

    @property
    def item_count(self) -> int:
        return sum(i.quantity for i in self.items)
