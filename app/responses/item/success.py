# app/responses/item/success.py
"""Voice responses for successful item operations and terminal item states."""
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.responses.item.format_utils import _added_text, _format_selected_names
from app.state_machine.models.conversation_context import ConversationContext


def item_added_successfully(payload: dict) -> str:
    quantity = int(payload.get("quantity", 1))
    item_name = str(payload.get("item_name") or "").strip()

    unmatched = payload.get("unmatched_names") or []
    unmatched_note = ""
    if unmatched:
        unmatched_note = f" I couldn't find {_format_selected_names(unmatched)}."

    if payload.get("queue_transition"):
        prev_name = str(payload.get("prev_item_name") or "").strip()
        prev_qty = int(payload.get("prev_quantity", 1) or 1)
        next_name = str(payload.get("next_item_name") or "").strip()
        remaining = int(payload.get("remaining_queue_count", 0) or 0)

        added = _added_text(prev_name, prev_qty)
        this_added = _added_text(item_name, quantity)

        if remaining > 0:
            return f"{added}. {this_added}.{unmatched_note} {remaining} more to go. Would you like anything else?"
        return f"{added}. {this_added}.{unmatched_note} That's everything. Would you like anything else?"

    if quantity > 1 and item_name:
        return f"Added {quantity} {item_name}.{unmatched_note} Would you like anything else?"
    if quantity > 1:
        return f"Added {quantity}.{unmatched_note} Would you like anything else?"
    if item_name:
        return f"{item_name} added.{unmatched_note} Would you like anything else?"
    return f"Added.{unmatched_note} Would you like anything else?"


def item_cancelled_successfully(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    return "Okay, cancelled. What would you like next?"


def item_context_missing(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    return "Something went wrong with that item. Let's start again."


def size_not_applicable(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    item_name = (payload or {}).get("item_name") or context.current_item_name or "That item"
    return f"{item_name} does not need a size. Let's continue."
