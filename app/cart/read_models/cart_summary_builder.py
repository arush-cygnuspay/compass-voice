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

            base = self._get_item_base_price(cart_item, menu_item)
            sides = self._get_sides_price(cart_item, menu_item)
            modifiers = self._get_modifiers_price(cart_item, menu_item)

            unit_price = base + sides + modifiers
            line_total = unit_price * cart_item.quantity
            total_cents += line_total

            items.append({
                "name": menu_item.name,
                "quantity": cart_item.quantity,
                "unit_price": f"${unit_price / 100:.2f}",
                "line_total": f"${line_total / 100:.2f}",
            })

        return {
            "items": items,
            "total": f"${total_cents / 100:.2f}",
        }


    def _get_item_base_price(self, cart_item, menu_item) -> int:
            if cart_item.variant_id:
                variant = next(
                    v for v in menu_item.pricing.variants
                    if v.variant_id == cart_item.variant_id
                )
                return variant.price_cents
            return menu_item.pricing.price_cents or 0

    def _get_sides_price(self, cart_item, menu_item) -> int:
        total = 0
        for group in menu_item.side_groups:
            chosen_ids = cart_item.sides.get(group.group_id, [])
            for choice in group.choices:
                if choice.item_id in chosen_ids:
                    total += choice.pricing.price_cents or 0
        return total

    def _get_modifiers_price(self, cart_item, menu_item) -> int:
        total = 0
        for group in menu_item.modifier_groups:
            chosen_ids = cart_item.modifiers.get(group.group_id, [])
            for choice in group.choices:
                if choice.modifier_id in chosen_ids:
                    total += choice.price_cents or 0
        return total
