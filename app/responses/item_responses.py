# app/responses/item_responses.py
from app.menu.repository import MenuRepository
from app.state_machine.context import ConversationContext


def ask_for_side(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)

    idx = context.current_side_group_index
    group = item.side_groups[idx]

    verb = "choose"
    group_name = normalize_group_name(group.name, verb)

    prefix = (
        "Please choose"
        if idx == 0
        else "Now please choose"
    )

    lines = [
        f"{prefix} {group_name} for your {item.name}.",
        f"You can choose {group.min_selector} option(s):"
    ]

    for i, choice in enumerate(group.choices, 1):
        price = (
            f" (+${choice.pricing.price_cents / 100:.2f})"
            if choice.pricing.price_cents
            else ""
        )
        lines.append(f"{i}. {choice.name}{price}")

    return "\n".join(lines)

def ask_for_modifier(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)

    idx = context.current_modifier_group_index
    group = item.modifier_groups[idx]

    verb = "add"
    group_name = normalize_group_name(group.name, verb)

    prefix = (
        "Would you like to add"
        if idx == 0
        else "You can also add"
    )

    lines = [
        f"{prefix} {group_name} to your {item.name}?",
        "You can choose one option or say no:"
    ]

    for i, choice in enumerate(group.choices, 1):
        price = (
            f" (+${choice.price_cents / 100:.2f})"
            if choice.price_cents
            else "(+$0.0)"
        )
        lines.append(f"{i}. {choice.name}{price}")

    return "\n".join(lines)

def ask_for_size(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    lines = ["Which size would you like?"]

    for v in item.pricing.variants:
        lines.append(f"- {v.label} (${v.price_cents / 100:.2f})")

    return "\n".join(lines)

def required_side_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.side_groups[context.current_side_group_index]

    lines = [
        f"You must choose {group.min_selector} option to continue.",
        f"{group.name} is required for your {item.name}.",
        "Please select one of the following options:",
    ]

    for i, choice in enumerate(group.choices, 1):
        price = (
            f" (+${choice.pricing.price_cents / 100:.2f})"
            if choice.pricing.price_cents
            else ""
        )
        lines.append(f"{i}. {choice.name}{price}")

    return "\n".join(lines)

def required_modifier_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)
    group = item.modifier_groups[context.current_modifier_group_index]

    lines = [
        f"You must select an option to proceed.",
        f"{group.name} is required for your {item.name}.",
        "Please choose one of the following:",
    ]

    for i, choice in enumerate(group.choices, 1):
        price = (
            f" (+${choice.price_cents / 100:.2f})"
            if choice.price_cents
            else "(+$0.00)"
        )
        lines.append(f"{i}. {choice.name}{price}")

    return "\n".join(lines)

def required_size_cannot_skip(
    context: ConversationContext,
    menu_repo: MenuRepository,
) -> str:
    item = menu_repo.store.get_item(context.current_item_id)

    lines = [
        "You must choose a valid size option to continue.",
        f"Please select one of the following sizes for your {item.name}:",
    ]

    for v in item.pricing.variants:
        lines.append(
            f"- {v.label} (${v.price_cents / 100:.2f})"
        )

    return "\n".join(lines)

def item_added_successfully(
    payload: dict,
) -> str:
    item_name = payload["item_name"]
    quantity = payload["quantity"]

    if quantity > 1:
        return (
            f"{quantity} {item_name}s have been added to your cart.\n"
            "Would you like to add anything else?"
        )

    return (
        f"Your {item_name} has been added to the cart.\n"
        "Would you like to add anything else?"
    )


def normalize_group_name(group_name: str, verb: str) -> str:
    """
    Ensures we don't duplicate verbs like:
    'choose choose cheese'
    """
    normalized = group_name.strip()

    lower = normalized.lower()
    verb_lower = verb.lower()

    if lower.startswith(verb_lower):
        return normalized[len(verb):].lstrip()

    return normalized
