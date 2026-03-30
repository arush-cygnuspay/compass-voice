# app/responses/menu_responses.py
from __future__ import annotations

DEFAULT_LIST_LIMIT = 4
DEFAULT_CATEGORY_LIMIT = 5


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
    limited = cleaned[:max_items]

    if not limited:
        return ""
    if len(limited) == 1:
        return limited[0]
    if len(limited) == 2:
        return f"{limited[0]} or {limited[1]}"
    return f"{limited[0]}, {limited[1]}, or {limited[2]}"


def show_category_response(payload: dict) -> str:
    category_name = _clean_text(payload.get("category_name")) or "that category"
    items = [
        _clean_text(item)
        for item in (payload.get("items") or [])
        if _clean_text(item)
    ]

    if not items:
        return f"There is nothing available in {category_name} right now."

    joined = _join_options(items)
    if joined:
        return f"In {category_name}, we have {joined}. What would you like?"
    return f"What would you like from {category_name}?"


def show_menu_categories_response(payload: dict) -> str:
    categories = [
        _clean_text(category)
        for category in (payload.get("categories") or [])
        if _clean_text(category)
    ]

    if not categories:
        return "Which category would you like?"

    joined = _join_options(categories, max_items=DEFAULT_CATEGORY_LIMIT)
    if joined:
        return f"Our categories are {joined}. Which one would you like?"
    return "Which category would you like?"


def show_item_info_response(payload: dict) -> str:
    item_name = _clean_text(payload.get("item_name")) or "That item"
    description = _clean_text(payload.get("description"))

    if description:
        return f"{item_name}. {description}"
    return f"{item_name} is on the menu."


def show_item_price_response(payload: dict) -> str:
    name = _clean_text(payload.get("item_name")) or "That item"
    pricing = payload.get("pricing") or {}

    variant_label = _clean_text(payload.get("variant_label"))
    variant_price = _format_price(payload.get("variant_price_cents"))

    if variant_label and variant_price:
        return f"{variant_label} {name} is {variant_price}."

    mode = pricing.get("mode")
    price = _format_price(pricing.get("price_cents"))
    variants = pricing.get("variants") or []

    if mode == "fixed" and price:
        return f"{name} is {price}."

    if mode == "unit" and price:
        return f"{name} is {price} each."

    if mode == "variant":
        parts: list[str] = []
        for variant in variants:
            label = _clean_text(variant.get("label"))
            variant_price_value = _format_price(variant.get("price_cents"))
            if label and variant_price_value:
                parts.append(f"{label} {variant_price_value}")

        joined = _join_options(parts)
        if joined:
            return f"{name} comes in {joined}."

    return "Price information is not available right now."


def show_item_availability_response(payload: dict) -> str:
    item_name = _clean_text(payload.get("item_name")) or "That item"
    description = _clean_text(payload.get("description"))
    pricing = payload.get("pricing") or {}
    variants = pricing.get("variants") or []

    if variants:
        labels = [
            _clean_text(variant.get("label"))
            for variant in variants
            if _clean_text(variant.get("label"))
        ]
        joined = _join_options(labels)
        if joined and description:
            return f"Yes, {item_name} is available. {description} It comes in {joined}."
        if joined:
            return f"Yes, {item_name} is available. It comes in {joined}."

    if description:
        return f"Yes, {item_name} is available. {description}"

    return f"Yes, {item_name} is available."


def show_modifier_availability_response(payload: dict) -> str:
    match_type = _clean_text(payload.get("match_type"))

    if match_type == "modifier":
        name = _clean_text(payload.get("modifier_name")) or "That add-on"
        price = _format_price(payload.get("price_cents"))

        if price:
            return f"Yes, {name} is available for {price}."
        return f"Yes, {name} is available."

    if match_type == "side":
        name = _clean_text(payload.get("item_name")) or "That side"
        return f"Yes, {name} is available."

    return "Yes, that option is available."


def menu_ambiguity_response(payload: dict) -> str:
    options = [
        _clean_text(option)
        for option in (payload.get("options") or [])
        if _clean_text(option)
    ]

    if not options:
        return "I found a few matches. Which one did you mean?"

    joined = _join_options(options)
    if joined:
        return f"I found {joined}. Which one did you mean?"
    return "I found a few matches. Which one did you mean?"


def menu_not_found_response() -> str:
    return "I could not find that on the menu."


def modifier_available_with_item_context_response(payload: dict) -> str:
    name = _clean_text(payload.get("modifier_name")) or "That add-on"
    return f"{name} is an add-on. Tell me the item, and I’ll check it."