# app/responses/side_size_responses.py
"""Voice responses for the side-variant (size-of-a-side) selection step.

These renderers are registered in ResponseBuilder but live here — in the
responses layer — where all other item-step renderers live.
"""
from __future__ import annotations

from app.responses.item.format_utils import _build_entity_feedback


def ask_for_side_size(payload: dict) -> str:
    side_item_name = payload.get("side_item_name") or "that side"
    available_sizes = [str(x).strip() for x in (payload.get("available_sizes") or []) if str(x).strip()]
    feedback = _build_entity_feedback(payload)

    if not available_sizes:
        prompt = f"What size for {side_item_name}?"
    elif len(available_sizes) == 1:
        prompt = f"Size for {side_item_name}? {available_sizes[0]}."
    elif len(available_sizes) == 2:
        prompt = f"Size for {side_item_name}? {available_sizes[0]} or {available_sizes[1]}."
    else:
        prompt = f"Size for {side_item_name}? {available_sizes[0]}, {available_sizes[1]}, or {available_sizes[2]}."

    return f"{feedback}{prompt}" if feedback else prompt


def repeat_side_size_options(payload: dict) -> str:
    side_item_name = payload.get("side_item_name") or "that side"
    available_sizes = [str(x).strip() for x in (payload.get("available_sizes") or []) if str(x).strip()]
    feedback = _build_entity_feedback(payload)

    if not available_sizes:
        prompt = f"What size for {side_item_name}?"
    elif len(available_sizes) == 1:
        prompt = f"Choose {available_sizes[0]}."
    elif len(available_sizes) == 2:
        prompt = f"Choose {available_sizes[0]} or {available_sizes[1]}."
    else:
        prompt = f"Choose {available_sizes[0]}, {available_sizes[1]}, or {available_sizes[2]}."

    return f"{feedback}{prompt}" if feedback else prompt
