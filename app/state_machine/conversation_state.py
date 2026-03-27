# app/state_machine/conversation_state.py

from __future__ import annotations

from enum import Enum


class ConversationState(str, Enum):
    """
    Single authoritative dialog state.

    Rules:
    - States represent task phases (slot-filling / confirmation / payment).
    - States are NOT intents and NOT overlays.
    """

    # System
    GREETING = "greeting"
    IDLE = "idle"
    ERROR_RECOVERY = "error_recovery"

    # Add-item flow
    CONFIRMING_ITEM = "confirming_item"
    WAITING_FOR_SIZE = "waiting_for_size"
    WAITING_FOR_SIDE = "waiting_for_side"
    WAITING_FOR_SIDE_SIZE = "waiting_for_side_size"
    WAITING_FOR_MODIFIER = "waiting_for_modifier"
    WAITING_FOR_QUANTITY = "waiting_for_quantity"
    FINALIZING_ADD_ITEM = "finalizing_add_item"

    # Modify / remove flows
    MODIFYING_ITEM = "modifying_item"
    REMOVING_ITEM = "removing_item"

    # Order / payment flow
    CONFIRMING_ORDER = "confirming_order"
    WAITING_FOR_PAYMENT = "waiting_for_payment"

    # Mid-flow cancellation confirmation
    CANCELLATION_CONFIRMATION = "cancellation_confirmation"