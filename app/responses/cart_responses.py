# app/responses/cart_responses.py
from __future__ import annotations

from app.core.quantity_formatter import format_item_quantity, parse_item_quantity
from app.responses.item.format_utils import spoken_quantity_label


def render_cart_summary(payload: dict) -> str:
    items = payload.get("items", [])
    total = payload.get("total")

    if not items:
        return "Your cart is empty."

    item_count = int(payload.get("item_count") or sum(parse_item_quantity(item.get("quantity", 1)) for item in items))

    if item_count == 1 and len(items) == 1:
        item = items[0]
        quantity = format_item_quantity(item.get("quantity", 1))
        name = item.get("name", "item")

        if total:
            return f"You have {quantity} {name}. Total {total}. Would you like to add more or check out?"
        return f"You have {quantity} {name}. Would you like to add more or check out?"

    if total:
        return f"You have {item_count} items. Total {total}. Would you like to add more or check out?"

    return f"You have {item_count} items. What would you like next?"


def render_cart_line_voice_compact(item: dict) -> str:
    """Render a single cart item for voice: quantity + name only, no modifiers or sides."""
    quantity = parse_item_quantity(item.get("quantity", 1))
    name = item.get("name", "item")
    return spoken_quantity_label(quantity, name)


def render_checkout_review_summary(payload: dict, order_type: str | None = None) -> str:
    items = payload.get("items", [])
    total = payload.get("total")

    if not items:
        return "Your cart is empty."

    intro = ""
    if order_type == "pickup":
        intro = "This is a pickup order. "
    elif order_type == "delivery":
        intro = "This is a delivery order. "

    item_parts = [render_cart_line_voice_compact(item) for item in items]
    items_text = ". ".join(item_parts)
    total_text = f" Your total is {total}." if total else ""

    return (
        f"{intro}Please review your order: "
        f"{items_text}."
        f"{total_text} "
        f"Should I place the order?"
    )


def confirm_clear_cart_response() -> str:
    return "Should I clear the cart?"


def cart_cleared_response() -> str:
    return "Okay, your cart is cleared."


def clear_cart_cancelled_response() -> str:
    return "Okay, I kept your cart."
