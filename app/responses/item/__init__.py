# app/responses/item/__init__.py
"""Public re-exports for the item responses package.

Consumers should import from here (or from the top-level item_responses shim)
rather than from individual sub-modules, so that internal reorganisations
remain transparent.
"""
from app.responses.item.confirmation import (
    confirm_cancel_current_item,
    confirm_cancel_current_item_for_new_request,
    confirm_item_ambiguous,
    confirm_item_from_category,
    continue_current_item_after_cancel_denied,
)
from app.responses.item.format_utils import (
    _added_text,
    _build_entity_feedback,
    _format_item_summary_list,
)
from app.responses.item.modifiers import (
    ask_for_modifier,
    clarify_modifier_choice,
    list_modifier_options,
    repeat_modifier_options,
    required_modifier_cannot_skip,
    too_many_modifier_choices,
)
from app.responses.item.not_found import (
    item_clarification_limit_reached,
    item_not_found,
    item_not_found_escalation,
    item_not_found_near_miss,
    repeat_item_request,
)
from app.responses.item.quantity import ask_item_quantity, invalid_quantity_option
from app.responses.item.sides import (
    ask_for_side,
    clarify_side_choice,
    list_side_options,
    repeat_side_options,
    required_side_cannot_skip,
    too_many_side_choices,
)
from app.responses.item.sizes import (
    ask_for_size,
    invalid_size_option,
    repeat_size_options,
    required_size_cannot_skip,
)
from app.responses.item.success import (
    item_added_successfully,
    item_cancelled_successfully,
    item_context_missing,
    size_not_applicable,
)

__all__ = [
    # confirmation
    "confirm_cancel_current_item",
    "confirm_cancel_current_item_for_new_request",
    "confirm_item_ambiguous",
    "confirm_item_from_category",
    "continue_current_item_after_cancel_denied",
    # format utils (imported by response_builder)
    "_added_text",
    "_build_entity_feedback",
    "_format_item_summary_list",
    # modifiers
    "ask_for_modifier",
    "clarify_modifier_choice",
    "list_modifier_options",
    "repeat_modifier_options",
    "required_modifier_cannot_skip",
    "too_many_modifier_choices",
    # not found
    "item_clarification_limit_reached",
    "item_not_found",
    "item_not_found_escalation",
    "item_not_found_near_miss",
    "repeat_item_request",
    # quantity
    "ask_item_quantity",
    "invalid_quantity_option",
    # sides
    "ask_for_side",
    "clarify_side_choice",
    "list_side_options",
    "repeat_side_options",
    "required_side_cannot_skip",
    "too_many_side_choices",
    # sizes
    "ask_for_size",
    "invalid_size_option",
    "repeat_size_options",
    "required_size_cannot_skip",
    # success / terminal states
    "item_added_successfully",
    "item_cancelled_successfully",
    "item_context_missing",
    "size_not_applicable",
]
