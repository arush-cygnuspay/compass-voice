# app/responses/menu_responses.py

from __future__ import annotations

from app.responses.utils import numbered_list

DEFAULT_LIST_LIMIT = 5
DEFAULT_CATEGORY_LIMIT = 6


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _format_price(cents: object) -> str | None:
    try:
        cents_int = int(cents)
    except (TypeError, ValueError):
        return None
    return f"${cents_int / 100:.2f}"


def _join_options(options: list[str], max_items: int = DEFAULT_LIST_LIMIT) -> str:
    cleaned = [_clean_text(option) for option in options if _clean_text(option)]
    if not cleaned:
        return ""

    limited = cleaned[:max_items]
    if len(limited) == 1:
        return limited[0]
    if len(limited) == 2:
        return f"{limited[0]} or {limited[1]}"
    return f"{', '.join(limited[:-1])}, or {limited[-1]}"


def show_category_response(payload: dict) -> str:
    category_name = _clean_text(payload.get("category_name")) or "that category"
    items = [_clean_text(item) for item in (payload.get("items") or []) if _clean_text(item)]

    if not items:
        return f"There is nothing available in {category_name} right now."

    joined_items = _join_options(items)
    if joined_items:
        return f"In {category_name}, we have {joined_items}. What would you like?"

    lines = [f"In {category_name}, we have:"]
    lines.extend(numbered_list(items, max_items=DEFAULT_LIST_LIMIT))
    lines.append("What would you like?")
    return "\n".join(lines)


def show_item_info_response(payload: dict) -> str:
    item_name = _clean_text(payload.get("item_name")) or "This item"
    description = _clean_text(payload.get("description"))

    if description:
        return f"{item_name}: {description}"

    return f"{item_name} is on the menu."


def menu_ambiguity_response(payload: dict) -> str:
    options = [_clean_text(option) for option in (payload.get("options") or []) if _clean_text(option)]

    if not options:
        return "I found multiple matches. Please be more specific."

    joined_options = _join_options(options)
    if joined_options:
        return f"I found a few matches: {joined_options}. Which one did you mean?"

    lines = ["I found a few matches:"]
    lines.extend(numbered_list(options, max_items=DEFAULT_LIST_LIMIT))
    lines.append("Which one did you mean?")
    return "\n".join(lines)


def menu_not_found_response() -> str:
    return "Sorry, I could not find that on the menu."


def show_item_price_response(payload: dict) -> str:
    name = _clean_text(payload.get("item_name")) or "That item"
    pricing = payload.get("pricing") or {}

    variant_label = _clean_text(payload.get("variant_label"))
    variant_price = _format_price(payload.get("variant_price_cents"))

    if variant_label and variant_price:
        return f"{variant_label.title()} {name} is {variant_price}."

    mode = pricing.get("mode")
    price = _format_price(pricing.get("price_cents"))
    variants = pricing.get("variants") or []

    if mode == "fixed" and price:
        return f"{name} is {price}."

    if mode == "unit" and price:
        return f"{name} is {price} per unit."

    if mode == "variant":
        variant_parts: list[str] = []
        for variant in variants:
            label = _clean_text(variant.get("label")) or "Option"
            variant_price = _format_price(variant.get("price_cents"))
            if not variant_price:
                continue
            variant_parts.append(f"{label} {variant_price}")

        joined_variants = _join_options(variant_parts)
        if joined_variants:
            return f"{name} comes in {joined_variants}."

    return "Price information is unavailable."


def show_menu_categories_response(payload: dict) -> str:
    categories = [
        _clean_text(category)
        for category in (payload.get("categories") or [])
        if _clean_text(category)
    ]

    if not categories:
        return "We have several menu categories. Which one would you like?"

    joined_categories = _join_options(categories, max_items=DEFAULT_CATEGORY_LIMIT)
    if joined_categories:
        return f"Our categories are {joined_categories}. Which one would you like?"

    lines = ["Menu categories:"]
    lines.extend(numbered_list(categories, max_items=DEFAULT_CATEGORY_LIMIT))
    lines.append("Which category would you like?")
    return "\n".join(lines)


def show_item_availability_response(payload: dict) -> str:
    item_name = _clean_text(payload.get("item_name")) or "That item"
    description = _clean_text(payload.get("description"))
    pricing = payload.get("pricing") or {}
    mode = pricing.get("mode")
    variants = pricing.get("variants") or []

    if mode == "variant" and variants:
        labels = [
            _clean_text(variant.get("label"))
            for variant in variants
            if _clean_text(variant.get("label"))
        ]
        joined_labels = _join_options(labels)
        if joined_labels and description:
            return f"Yes, {item_name} is available. {description} It comes in {joined_labels}."
        if joined_labels:
            return f"Yes, {item_name} is available. It comes in {joined_labels}."

    if description:
        return f"Yes, {item_name} is available. {description}"

    return f"Yes, {item_name} is available."


def show_modifier_availability_response(payload: dict) -> str:
    match_type = _clean_text(payload.get("match_type"))

    if match_type == "modifier":
        modifier_name = _clean_text(payload.get("modifier_name")) or "That add-on"
        group_name = _clean_text(payload.get("group_name")) or "this item"
        price = _format_price(payload.get("price_cents"))

        if price:
            return f"Yes, {modifier_name} is available under {group_name} for {price}."
        return f"Yes, {modifier_name} is available under {group_name}."

    if match_type == "side":
        item_name = _clean_text(payload.get("item_name")) or "That option"
        group_name = _clean_text(payload.get("group_name")) or "this item"
        return f"Yes, {item_name} is available under {group_name}."

    return "Yes, that option is available."


def modifier_available_with_item_context_response(payload: dict) -> str:
    modifier_name = _clean_text(payload.get("modifier_name")) or "that add-on"
    return (
        f"{modifier_name.title()} is usually an add-on, not a main item. "
        "Tell me the item, and I will check it."
    )