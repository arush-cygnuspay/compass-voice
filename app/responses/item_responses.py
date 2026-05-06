# app/responses/item_responses.py
"""Backward-compatible re-export shim.

All item-centric response functions have moved to app.responses.item.* sub-modules.
This file exists so that existing imports (``from app.responses.item_responses import …``)
continue to work without modification.
"""
# ruff: noqa: F401  (re-exported names are intentionally unused here)
from app.responses.item.format_utils import (
    _has_echable_content,
    _progress_prompt,
)
from app.responses.item import (
    _added_text,
    _build_entity_feedback,
    _format_item_summary_list,
    ask_for_modifier,
    ask_for_side,
    ask_for_size,
    ask_item_quantity,
    clarify_modifier_choice,
    clarify_side_choice,
    confirm_cancel_current_item,
    confirm_cancel_current_item_for_new_request,
    confirm_item_ambiguous,
    confirm_item_from_category,
    continue_current_item_after_cancel_denied,
    invalid_quantity_option,
    invalid_size_option,
    item_added_successfully,
    item_cancelled_successfully,
    item_clarification_limit_reached,
    item_context_missing,
    item_not_found,
    item_not_found_escalation,
    item_not_found_near_miss,
    list_modifier_options,
    list_side_options,
    repeat_item_request,
    repeat_modifier_options,
    repeat_side_options,
    repeat_size_options,
    required_modifier_cannot_skip,
    required_side_cannot_skip,
    required_size_cannot_skip,
    size_not_applicable,
    too_many_modifier_choices,
    too_many_side_choices,
)
