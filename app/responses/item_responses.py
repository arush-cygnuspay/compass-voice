# app/responses/item_responses.py
from app.menu.repository import MenuRepository
from app.state_machine.context import ConversationContext


def ask_for_side(context: ConversationContext, menu_repo: MenuRepository) -> str:
    item = menu_repo.store.get_item(context.current_item_id)

    idx = context.current_side_group_index
    group = item.side_groups[idx]

    prefix = (
        "Please choose"
        if idx == 0
        else "Now please choose"
    )

    lines = [
        f"{prefix} {group.name} for your {item.name}.",
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

    prefix = (
        "Would you like to add"
        if idx == 0
        else "You can also add"
    )

    lines = [
        f"{prefix} {group.name} to your {item.name}?",
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

def item_added_successfully(
    payload: dict,
) -> str:
    item_name = payload["item_name"]
    quantity = payload["quantity"]

    if quantity > 1:
        return (
            f"✅ {quantity} {item_name}s have been added to your cart.\n"
            "Would you like to add anything else?"
        )

    return (
        f"✅ Your {item_name} has been added to the cart.\n"
        "Would you like to add anything else?"
    )
