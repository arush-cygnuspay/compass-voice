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
            return f"{quantity} {name}. Total {total}. Add more or checkout?"

        return f"{quantity} {name}. Add more or checkout?"

    if total:
        return f"{count} items. Total {total}. Add more or checkout?"

    return f"{count} items. Add more or checkout?"


def confirm_clear_cart_response() -> str:
    return "Clear the cart? Yes or no."


def cart_cleared_response() -> str:
    return "Cart cleared."


def clear_cart_cancelled_response() -> str:
    return "Okay, keeping your cart."