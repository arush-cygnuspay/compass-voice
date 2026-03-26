# app/responses/cart_responses.py


def render_cart_summary(payload: dict) -> str:
    items = payload.get("items", [])

    if not items:
        return "Your cart is empty."

    count = len(items)
    total = payload.get("total")

    if count == 1:
        item = items[0]
        quantity = item.get("quantity", 1)
        name = item.get("name", "item")

        if total:
            return f"You have {quantity} {name}. Total {total}. Add more or check out?"

        return f"You have {quantity} {name}. Add more or check out?"

    if total:
        return f"You have {count} items. Total {total}. Add more or check out?"

    return f"You have {count} items. What would you like next?"


def confirm_clear_cart_response() -> str:
    return "Do you want to clear your cart? Please say yes or no."


def cart_cleared_response() -> str:
    return "Okay, your cart is cleared."


def clear_cart_cancelled_response() -> str:
    return "Okay, I kept your cart."