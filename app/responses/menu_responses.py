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
