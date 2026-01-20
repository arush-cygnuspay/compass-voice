# app/responses/cart_responses.py

def render_cart_summary(payload):
    lines = ["Here’s your order:"]
    for item in payload["items"]:
        lines.append(
            f"- {item['quantity']} {item['name']} — {item['line_total']}"
        )
    lines.append(f"\nTotal: {payload['total']}")
    lines.append("Would you like to proceed?")

    return "\n".join(lines)
