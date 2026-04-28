# app/responses/cart_responses.py
from __future__ import annotations


def render_cart_summary(payload: dict) -> str:
    items = payload.get("items", [])
    total = payload.get("total")

    if not items:
        return "Your cart is empty."

    item_count = int(payload.get("item_count") or sum(int(item.get("quantity", 1)) for item in items))

    if item_count == 1 and len(items) == 1:
        item = items[0]
        quantity = item.get("quantity", 1)
        name = item.get("name", "item")

        if total:
            return f"You have {quantity} {name}. Total {total}. Would you like to add more or check out?"
        return f"You have {quantity} {name}. Would you like to add more or check out?"

    if total:
        return f"You have {item_count} items. Total {total}. Would you like to add more or check out?"

    return f"You have {item_count} items. What would you like next?"


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

    item_parts = []
    for item in items:
        quantity = int(item.get("quantity", 1))
        name = item.get("name", "item")
        sides = [str(x).strip() for x in item.get("sides", []) if str(x).strip()]
        modifiers = [str(x).strip() for x in item.get("modifiers", []) if str(x).strip()]

        config_parts = []
        if sides:
            config_parts.append("with " + ", ".join(sides))
        if modifiers:
            config_parts.append("add " + ", ".join(modifiers))

        line = f"{quantity} {name}"
        if config_parts:
            line = f"{line}, " + ", ".join(config_parts)

        item_parts.append(line)

    items_text = ". ".join(item_parts)
    total_text = f" Your total is {total}." if total else ""

    return (
        f"{intro}Please review your order. "
        f"{items_text}."
        f"{total_text} "
        f"Should I place the order and continue to checkout? "
        f"If you want to change something, just tell me what to update."
    )


def confirm_clear_cart_response() -> str:
    return "Should I clear the cart?"


def cart_cleared_response() -> str:
    return "Okay, your cart is cleared."


def clear_cart_cancelled_response() -> str:
    return "Okay, I kept your cart."
