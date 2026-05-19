# app/nlu/turn_resolver/allowed_response_key_provider.py
"""State-specific allowed response key provider for GPT context building.

GPT may later suggest a response_key_hint, but this system remains the
final authority on which keys are valid for each state.
Unknown states return only safe generic keys.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.state_machine.models.conversation_context import ConversationContext

_GENERIC_KEYS: tuple[str, ...] = (
    "generic_clarification",
    "intent_not_allowed",
)

_STATE_RESPONSE_KEYS: dict[str, tuple[str, ...]] = {
    "idle": (
        "item_added_successfully",
        "ask_for_item",
        "confirm_order_summary",
        "item_not_found",
        "item_ambiguous",
        "multi_item_acknowledged",
        "generic_clarification",
        "intent_not_allowed",
    ),
    "waiting_for_order_type": (
        "ask_order_type",
        "order_type_set_pickup",
        "order_type_set_delivery",
        "generic_clarification",
    ),
    "waiting_for_modifier": (
        "ask_for_modifier",
        "list_modifier_options",
        "repeat_modifier_options",
        "required_modifier_cannot_skip",
        "modifier_selected",
        "modifier_not_found",
        "generic_clarification",
    ),
    "waiting_for_side": (
        "ask_for_side",
        "list_side_options",
        "repeat_side_options",
        "required_side_cannot_skip",
        "side_selected",
        "side_not_found",
        "generic_clarification",
    ),
    "waiting_for_size": (
        "ask_for_size",
        "repeat_size_options",
        "size_selected",
        "size_not_found",
        "generic_clarification",
    ),
    "waiting_for_side_size": (
        "ask_for_side_size",
        "repeat_side_size_options",
        "side_size_selected",
        "generic_clarification",
    ),
    "confirming_item": (
        "confirm_item_summary",
        "item_confirmed",
        "item_correction_accepted",
        "item_cancelled",
        "generic_clarification",
    ),
    "confirming_order": (
        "confirm_order_summary",
        "pickup_ask_sms_permission",
        "order_cancelled",
        "order_updated",
        "generic_clarification",
    ),
    "waiting_for_pickup_sms_permission": (
        "pickup_sms_sent_end_call",
        "pickup_no_sms_end_call",
        "pickup_repeat_sms_permission",
    ),
    "completed": (
        "order_completed",
    ),
    "transferring_to_human_agent": (
        "transferring_to_human_agent",
    ),
    "cancellation_confirmation": (
        "flow_guard_confirm_cancel",
        "cancellation_confirmed",
        "cancellation_denied",
        "generic_clarification",
    ),
}


class AllowedResponseKeyProvider:
    """Returns state-specific allowed response keys.

    Thread-safe: all state is class-level read-only dicts.
    """

    def get_allowed_response_keys_for_state(
        self,
        state: str,
        context: "ConversationContext | None" = None,
    ) -> tuple[str, ...]:
        """Return allowed response keys for *state*.

        Unknown states return only generic safe keys.
        Context is accepted for future context-aware filtering but not used now.
        """
        key = (state or "").strip().lower()
        return _STATE_RESPONSE_KEYS.get(key, _GENERIC_KEYS)
