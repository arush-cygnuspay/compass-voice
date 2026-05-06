# app/responses/item/not_found.py
"""Voice responses for item not-found and escalation scenarios."""
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.responses.item.format_utils import _format_suggestions, _looks_customizable
from app.state_machine.models.conversation_context import ConversationContext


def repeat_item_request(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    return "Which item would you like?"


def item_not_found(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    payload = payload or {}
    query = str(payload.get("query") or "").strip()

    suggestions = list(payload.get("suggestions") or [])
    if not suggestions:
        suggestions = list(payload.get("suggested_item_names") or [])
    if not suggestions:
        suggestions = [s for s in (payload.get("suggested_category_names") or []) if s]

    unavail = f"I don't see {query} on the menu." if query else "I don't see that on the menu."

    if suggestions:
        capped = [str(s).strip() for s in suggestions[:4] if str(s).strip()]
        customizable = [s for s in capped if _looks_customizable(s)]
        regular = [s for s in capped if not _looks_customizable(s)]

        if customizable and not regular:
            cust = customizable[0]
            return f"{unavail} {cust} can be customized to your liking. Want that?"

        options = _format_suggestions(capped)
        prefix = f"I don't see {query} on the menu" if query else "I don't see that on the menu"
        return f"{prefix}, but we have {options}. Which one would you like?"

    return f"{unavail} What else can I get you?"


def item_not_found_near_miss(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    item_name = str((payload or {}).get("item_name") or "").strip()
    if item_name:
        return f"Did you mean {item_name}?"
    return "Did you mean that item?"


def item_not_found_escalation(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    return "I don't have that item. You can choose another item, or I can connect you to the restaurant."


def item_clarification_limit_reached(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    return "I'm having trouble finding that item. What else can I get for you?"
