from typing import Dict
from app.cart.cart import Cart
from app.menu.repository import MenuRepository


class CartSummaryBuilder:
    """
    Builds a read-only cart summary for presentation layers.
    Pure, deterministic, and side-effect free.
    """

    def __init__(self, menu_repo: MenuRepository):
        self.menu_repo = menu_repo

    def build(self, cart: Cart) -> Dict:
        items = []
        total_cents = 0

        for cart_item in cart.get_items():
            menu_item = self.menu_repo.get_item(cart_item.item_id)

            if cart_item.variant_id:
                variant = next(
                    v for v in menu_item.pricing.variants
                    if v.variant_id == cart_item.variant_id
                )
                unit_price = variant.price_cents
            else:
                unit_price = menu_item.pricing.price_cents or 0

            line_total = unit_price * cart_item.quantity
            total_cents += line_total

            items.append({
                "name": menu_item.name,
                "quantity": cart_item.quantity,
                "line_total": f"${line_total / 100:.2f}",
            })

        return {
            "items": items,
            "total": f"${total_cents / 100:.2f}",
        }
