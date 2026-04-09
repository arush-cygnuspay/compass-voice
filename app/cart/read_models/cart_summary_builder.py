# app/cart/read_models/cart_summary_builder.py
from __future__ import annotations

from typing import Any, Dict, Tuple

from app.cart.cart import Cart
from app.menu.repository import MenuRepository


class CartSummaryBuilder:
    """
    Builds read-only cart summaries for presentation layers.

    Two presentation modes use the same payload:
    - short cart summary
    - full checkout review summary
    """

    def __init__(self, menu_repo: MenuRepository):
        self.menu_repo = menu_repo

    def build(self, cart: Cart) -> Dict[str, Any]:
        grouped_items: Dict[Tuple, Dict[str, Any]] = {}
        total_cents = 0

        for cart_item in cart.get_items():
            menu_item = self.menu_repo.get_item(cart_item.item_id)

            base_cents = self._get_item_base_price(cart_item, menu_item)
            sides_cents = self._get_sides_price(cart_item, menu_item)
            modifiers_cents = self._get_modifiers_price(cart_item, menu_item)
            unit_price_cents = base_cents + sides_cents + modifiers_cents

            variant_label = self._get_item_variant_label(cart_item, menu_item)
            side_labels = self._get_side_labels(cart_item, menu_item)
            modifier_labels = self._get_modifier_labels(cart_item, menu_item)

            group_key = self._build_group_key(cart_item)

            if group_key in grouped_items:
                grouped_items[group_key]["quantity"] += cart_item.quantity
                grouped_items[group_key]["line_total_cents"] += unit_price_cents * cart_item.quantity
                continue

            grouped_items[group_key] = {
                "name": menu_item.name,
                "variant_label": variant_label,
                "sides": side_labels,
                "modifiers": modifier_labels,
                "quantity": cart_item.quantity,
                "unit_price_cents": unit_price_cents,
                "line_total_cents": unit_price_cents * cart_item.quantity,
            }

        items: list[Dict[str, Any]] = []
        total_item_quantity = 0

        for grouped_item in grouped_items.values():
            total_cents += grouped_item["line_total_cents"]
            total_item_quantity += grouped_item["quantity"]

            display_name = grouped_item["name"]
            if grouped_item["variant_label"]:
                display_name = f"{display_name} ({grouped_item['variant_label']})"

            items.append(
                {
                    "name": display_name,
                    "base_name": grouped_item["name"],
                    "variant_label": grouped_item["variant_label"],
                    "sides": list(grouped_item["sides"]),
                    "modifiers": list(grouped_item["modifiers"]),
                    "quantity": grouped_item["quantity"],
                    "unit_price": f"${grouped_item['unit_price_cents'] / 100:.2f}",
                    "line_total": f"${grouped_item['line_total_cents'] / 100:.2f}",
                }
            )

        return {
            "items": items,
            "item_count": total_item_quantity,
            "total": f"${total_cents / 100:.2f}",
        }

    def _build_group_key(self, cart_item) -> Tuple:
        sides_key = tuple(
            (group_id, tuple(sorted(item_ids)))
            for group_id, item_ids in sorted(cart_item.sides.items())
        )
        side_variants_key = tuple(sorted(cart_item.side_variants.items()))
        modifiers_key = tuple(
            (group_id, tuple(sorted(modifier_ids)))
            for group_id, modifier_ids in sorted(cart_item.modifiers.items())
        )

        return (
            cart_item.item_id,
            cart_item.variant_id,
            sides_key,
            side_variants_key,
            modifiers_key,
        )

    def _get_item_variant_label(self, cart_item, menu_item) -> str | None:
        if not cart_item.variant_id:
            return None

        for variant in getattr(menu_item.pricing, "variants", []) or []:
            if variant.variant_id == cart_item.variant_id:
                return variant.label
        return None

    def _get_side_labels(self, cart_item, menu_item) -> tuple[str, ...]:
        labels: list[str] = []

        for group in getattr(menu_item, "side_groups", []) or []:
            chosen_ids = set(cart_item.sides.get(group.group_id, []))
            if not chosen_ids:
                continue

            for choice in getattr(group, "choices", []) or []:
                if choice.item_id not in chosen_ids:
                    continue

                label = choice.name
                chosen_variant_id = cart_item.side_variants.get(choice.item_id)
                if chosen_variant_id:
                    variant_label = self._get_side_variant_label(choice, chosen_variant_id)
                    if variant_label:
                        label = f"{label} {variant_label}"

                labels.append(label)

        return tuple(labels)

    def _get_side_variant_label(self, side_choice, variant_id: str) -> str | None:
        pricing = getattr(side_choice, "pricing", None)
        variants = getattr(pricing, "variants", None) or []

        for variant in variants:
            if getattr(variant, "variant_id", None) == variant_id:
                return getattr(variant, "label", None)
        return None

    def _get_modifier_labels(self, cart_item, menu_item) -> tuple[str, ...]:
        labels: list[str] = []

        for group in getattr(menu_item, "modifier_groups", []) or []:
            chosen_ids = set(cart_item.modifiers.get(group.group_id, []))
            if not chosen_ids:
                continue

            for choice in getattr(group, "choices", []) or []:
                if choice.modifier_id in chosen_ids:
                    labels.append(choice.name)

        return tuple(labels)

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