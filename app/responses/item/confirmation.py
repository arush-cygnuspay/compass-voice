# app/responses/item/confirmation.py
"""Voice responses for item confirmation and cancellation flows."""
from __future__ import annotations

from app.menu.repository import MenuRepository
from app.responses.item.format_utils import _format_options, _format_options
from app.state_machine.models.conversation_context import ConversationContext


def confirm_item_ambiguous(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    payload = payload or {}
    item_names = payload.get("candidate_item_names") or []
    category_names = payload.get("candidate_category_names") or []

    if item_names:
        return f"Did you mean {_format_options(item_names)}?"
    if category_names:
        return f"Did you mean {_format_options(category_names)}?"
    return "I found a few matches. Which one did you mean?"


def confirm_item_from_category(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    payload = payload or {}
    category_name = payload.get("category_name")
    item_names = payload.get("candidate_item_names") or []

    if item_names and category_name:
        return f"In {category_name}, I found {_format_options(item_names)}. Which one would you like?"
    if item_names:
        return f"I found {_format_options(item_names)}. Which one would you like?"
    return "Which item would you like?"


def confirm_cancel_current_item(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    item_name = (payload or {}).get("item_name") or context.current_item_name or "this item"
    return f"Cancel {item_name}?"


def confirm_cancel_current_item_for_new_request(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    item_name = (payload or {}).get("item_name") or context.current_item_name or "this item"
    return f"Still adding {item_name}. Cancel it and move on?"


def continue_current_item_after_cancel_denied(
    context: ConversationContext,
    menu_repo: MenuRepository,
    payload: dict | None,
) -> str:
    options = (payload or {}).get("available_choices") or list(context.available_choices_values)
    formatted = _format_options(options)
    if formatted:
        return f"Okay, continuing. {formatted}."
    return "Okay, continuing."
