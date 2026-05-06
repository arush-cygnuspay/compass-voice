# app/state_machine/models/conversation_state.py

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
    COMPLETED = "completed"

    # Add-item flow
    CONFIRMING_ITEM = "confirming_item"
    WAITING_FOR_SIZE = "waiting_for_size"
    WAITING_FOR_SIDE = "waiting_for_side"
    WAITING_FOR_SIDE_SIZE = "waiting_for_side_size"
    WAITING_FOR_MODIFIER = "waiting_for_modifier"
    WAITING_FOR_QUANTITY = "waiting_for_quantity"

    # Modify / remove flows
    MODIFYING_ITEM = "modifying_item"
    REMOVING_ITEM = "removing_item"

    # Order / payment flow
    CONFIRMING_ORDER = "confirming_order"
    WAITING_FOR_PICKUP_SMS_PERMISSION = "waiting_for_pickup_sms_permission"
    WAITING_FOR_PAYMENT = "waiting_for_payment"
    WAITING_FOR_CHECKOUT_COMPLETION = "waiting_for_checkout_completion"

    # Mid-flow cancellation confirmation
    CANCELLATION_CONFIRMATION = "cancellation_confirmation"

    WAITING_FOR_ORDER_TYPE = "waiting_for_order_type"
    WAITING_FOR_DELIVERY_ELIGIBILITY = "waiting_for_delivery_eligibility"
    WAITING_FOR_DELIVERY_ADDRESS_COLLECTION = "waiting_for_delivery_address_collection"

    # Caller device gating — reserved for explicit landline detection flows.
    # Not part of the initial call path; all new calls start at WAITING_FOR_ORDER_TYPE.
    WAITING_FOR_CALLER_DEVICE_TYPE = "waiting_for_caller_device_type"
    WAITING_FOR_LANDLINE_PICKUP_CONFIRMATION = "waiting_for_landline_pickup_confirmation"
    TRANSFERRING_TO_HUMAN_AGENT = "transferring_to_human_agent"
