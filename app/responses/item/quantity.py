# app/responses/item/quantity.py
"""Voice responses for the quantity step of the add-item flow."""
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.state_machine.models.conversation_context import ConversationContext


def ask_item_quantity(payload: dict) -> str:
    item_name = payload.get("item_name")
    if item_name:
        return f"How many {item_name} would you like?"
    return "How many would you like?"


def invalid_quantity_option(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    item_name = (payload or {}).get("item_name") or context.current_item_name
    if (payload or {}).get("reprompt_escalation"):
        if item_name:
            return f"Please say a number for {item_name}, like 1 or 2."
        return "Please say a number, like 1 or 2."
    if item_name:
        return f"Please give a valid quantity for {item_name}."
    return "Please give a valid quantity."
