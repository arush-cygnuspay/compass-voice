from typing import Dict, Tuple
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
        # Group items by their characteristics (item_id, variant_id, sides, modifiers)
        grouped_items: Dict[Tuple, Dict] = {}
        total_cents = 0

        for cart_item in cart.get_items():
            menu_item = self.menu_repo.get_item(cart_item.item_id)

            base = self._get_item_base_price(cart_item, menu_item)
            sides = self._get_sides_price(cart_item, menu_item)
            modifiers = self._get_modifiers_price(cart_item, menu_item)

            unit_price = base + sides + modifiers
            
            # Create a grouping key based on item characteristics
            # Items with the same item_id, variant_id, sides, and modifiers should be grouped
            # Convert sides dict to hashable tuple: (group_id, tuple(sorted(item_ids)))
            sides_key = tuple(
                (group_id, tuple(sorted(item_ids)))
                for group_id, item_ids in sorted(cart_item.sides.items())
            )
            # Convert modifiers dict to hashable tuple: (group_id, tuple(sorted(modifier_ids)))
            modifiers_key = tuple(
                (group_id, tuple(sorted(modifier_ids)))
                for group_id, modifier_ids in sorted(cart_item.modifiers.items())
            )
            
            group_key = (
                cart_item.item_id,
                cart_item.variant_id,
                sides_key,
                modifiers_key,
            )

            # Get variant label if variant exists
            variant_label = None
            if cart_item.variant_id:
                variant = next(
                    (v for v in menu_item.pricing.variants if v.variant_id == cart_item.variant_id),
                    None
                )
                if variant:
                    variant_label = variant.label

            if group_key in grouped_items:
                # Add to existing grouped item
                grouped_items[group_key]["quantity"] += cart_item.quantity
                grouped_items[group_key]["line_total_cents"] += unit_price * cart_item.quantity
            else:
                # Create new grouped item
                grouped_items[group_key] = {
                    "name": menu_item.name,
                    "variant_label": variant_label,
                    "quantity": cart_item.quantity,
                    "unit_price_cents": unit_price,
                    "line_total_cents": unit_price * cart_item.quantity,
                }

        # Convert grouped items to final format
        items = []
        for grouped_item in grouped_items.values():
            line_total_cents = grouped_item["line_total_cents"]
            total_cents += line_total_cents
            
            # Build display name with variant label if present
            display_name = grouped_item["name"]
            if grouped_item.get("variant_label"):
                display_name = f"{display_name} ({grouped_item['variant_label']})"
            
            items.append({
                "name": display_name,
                "quantity": grouped_item["quantity"],
                "unit_price": f"${grouped_item['unit_price_cents'] / 100:.2f}",
                "line_total": f"${line_total_cents / 100:.2f}",
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
