# app/responses/item/success.py
"""Voice responses for successful item operations and terminal item states."""
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.responses.item.format_utils import _added_text, _format_selected_names
from app.state_machine.models.conversation_context import ConversationContext


def _filter_item_labels(
    unmatched: list,
    item_aliases: list | tuple,
    item_voice_labels: list | tuple,
    item_name: str,
) -> list:
    """Remove entries from *unmatched* that are alias/voice-label forms of the
    resolved item.

    Prevents contradictory responses like "I couldn't find cheeseburger. Cheese
    Burger added." when "cheeseburger" is just the compact ASR form of the item.
    """
    if not unmatched:
        return list(unmatched)
    item_label_norms: set[str] = set()
    for label in (item_name,) + tuple(item_aliases) + tuple(item_voice_labels):
        norm = normalize_text(str(label))
        if norm:
            item_label_norms.add(norm)
            item_label_norms.add(norm.replace(" ", ""))
    filtered: list = []
    for entry in unmatched:
        norm = normalize_text(str(entry))
        if norm in item_label_norms:
            continue
        compact = norm.replace(" ", "")
        if compact and compact in item_label_norms:
            continue
        filtered.append(entry)
    return filtered


def item_added_successfully(payload: dict) -> str:
    quantity = int(payload.get("quantity", 1))
    item_name = str(payload.get("item_name") or "").strip()

    # Filter unmatched_names: suppress any alias/voice-label form of the item
    # itself so we never say "I couldn't find cheeseburger. Cheese Burger added."
    item_aliases = list(payload.get("item_aliases") or [])
    item_voice_labels = list(payload.get("item_voice_labels") or [])
    unmatched = list(payload.get("unmatched_names") or [])
    unmatched = _filter_item_labels(unmatched, item_aliases, item_voice_labels, item_name)

    unmatched_note = ""
    if unmatched:
        unmatched_note = f" I couldn't find {_format_selected_names(unmatched)}."

    if payload.get("queue_transition"):
        prev_name = str(payload.get("prev_item_name") or "").strip()
        prev_qty = int(payload.get("prev_quantity", 1) or 1)
        remaining = int(payload.get("remaining_queue_count", 0) or 0)

        # Deduplicate stacked acknowledgements: when the queued item and the
        # current item are the same (e.g. user said "2 Cokes" split into a
        # queue), merge into a single combined-quantity acknowledgement instead
        # of "Coke added. Coke added."
        if prev_name and item_name and prev_name.lower() == item_name.lower():
            combined_qty = prev_qty + quantity
            combined_added = _added_text(item_name, combined_qty)
            if remaining > 0:
                return f"{combined_added}.{unmatched_note} {remaining} more to go. Would you like anything else?"
            return f"{combined_added}.{unmatched_note} That's everything. Would you like anything else?"

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
