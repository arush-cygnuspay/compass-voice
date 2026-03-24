# app/responses/menu_responses.py

from app.responses.utils import numbered_list


DEFAULT_LIST_LIMIT = 5



def show_category_response(payload: dict) -> str:
    category_name = payload.get("category_name", "this category")
    items = payload.get("items", [])

    if not items:
        return f"There are no items available under {category_name} right now."

    lines = [f"We have the following {category_name}:"]

    lines.extend(
        numbered_list(
            items,
            max_items=DEFAULT_LIST_LIMIT,
        )
    )

    lines.append("Which one would you like?")

    return "\n".join(lines)


def show_item_info_response(payload: dict) -> str:
    item_name = payload.get("item_name", "This item")
    description = payload.get("description", "").strip()

    if description:
        return f"{item_name}: {description}"

    return f"{item_name} is available on our menu."


def menu_ambiguity_response(payload: dict) -> str:
    options = payload.get("options", [])

    if not options:
        return "I found multiple matches. Could you please be more specific?"

    lines = ["I found multiple matches:"]

    lines.extend(
        numbered_list(
            options,
            max_items=DEFAULT_LIST_LIMIT,
        )
    )

    lines.append("Which one did you mean?")

    return "\n".join(lines)


def menu_not_found_response() -> str:
    return "Sorry, I couldn’t find that on the menu."


def show_item_price_response(payload: dict) -> str:
    name = payload["item_name"]
    pricing = payload["pricing"]

    variant_label = payload.get("variant_label")
    variant_price_cents = payload.get("variant_price_cents")

    if variant_label and variant_price_cents is not None:
        return f"{variant_label.title()} {name} costs ${variant_price_cents / 100:.2f}."

    mode = pricing.get("mode")
    price_cents = pricing.get("price_cents")
    variants = pricing.get("variants") or []

    if mode == "fixed" and price_cents is not None:
        return f"{name} costs ${price_cents / 100:.2f}."

    if mode == "unit" and price_cents is not None:
        return f"{name} costs ${price_cents / 100:.2f} per unit."

    if mode == "variant":
        lines = [f"{name} is available in the following options:"]
        for v in variants:
            label = v.get("label", "Option")
            cents = v.get("price_cents")
            if cents is None:
                continue
            lines.append(f"- {label}: ${cents / 100:.2f}")
        return "\n".join(lines)

    return "Price information is unavailable."


def show_menu_categories_response(payload: dict) -> str:
    categories = [str(x).strip() for x in (payload.get("categories") or []) if str(x).strip()]

    if not categories:
        return "We have several categories on the menu. What would you like to see?"

    lines = ["You can browse these menu categories:"]
    lines.extend(numbered_list(categories, max_items=6))
    lines.append("Which category would you like?")
    return "\n".join(lines)


def show_item_availability_response(payload: dict) -> str:
    item_name = payload.get("item_name", "That item")
    description = str(payload.get("description", "")).strip()
    pricing = payload.get("pricing") or {}
    mode = pricing.get("mode")
    variants = pricing.get("variants") or []

    if mode == "variant" and variants:
        labels = [
            str(v.get("label", "")).strip()
            for v in variants
            if str(v.get("label", "")).strip()
        ]
        if labels:
            joined = ", ".join(labels[:5])
            if description:
                return f"Yes, {item_name} is available. {description} It comes in: {joined}."
            return f"Yes, {item_name} is available. It comes in: {joined}."

    if description:
        return f"Yes, {item_name} is available. {description}"

    return f"Yes, {item_name} is available."


def show_modifier_availability_response(payload: dict) -> str:
    match_type = payload.get("match_type")

    if match_type == "modifier":
        modifier_name = payload.get("modifier_name", "That add-on")
        group_name = payload.get("group_name", "this item")
        price_cents = payload.get("price_cents")

        if price_cents is None or int(price_cents) <= 0:
            return f"Yes, {modifier_name} is available for this item under {group_name}."

        return f"Yes, {modifier_name} is available for this item under {group_name} for ${int(price_cents) / 100:.2f}."

    if match_type == "side":
        item_name = payload.get("item_name", "That option")
        group_name = payload.get("group_name", "this item")
        return f"Yes, {item_name} is available for this item under {group_name}."

    return "Yes, that option is available for this item."


def modifier_available_with_item_context_response(payload: dict) -> str:
    modifier_name = str(payload.get("modifier_name", "that add-on")).strip() or "that add-on"
    return (
        f"{modifier_name.title()} is usually an add-on or modifier, not a standalone menu item. "
        "Tell me the item name and I’ll check whether it’s available for that item."
    )
