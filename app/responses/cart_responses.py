# app/responses/cart_responses.py

def render_cart_summary(payload: dict) -> str:
    items = payload.get("items", [])

    if not items:
        return "Your cart is empty."

    lines = ["Here’s your order:\n"]

    for item in items:
        quantity = item.get("quantity", 1)
        name = item.get("name", "Unknown item")
        line_total = item.get("line_total", "$0.00")

        lines.append(
            f"- {quantity} {name} — {line_total}"
        )

    total = payload.get("total", "$0.00")
    lines.append(f"\nTotal: {total}")

    return "\n".join(lines)

