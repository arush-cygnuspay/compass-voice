# app/responses/side_responses.py
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

