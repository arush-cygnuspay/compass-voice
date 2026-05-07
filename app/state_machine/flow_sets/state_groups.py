# app/state_machine/flow_sets/state_groups.py
"""State cohorts used across the FSM.

Import from here — or from the parent package ``app.state_machine.flow_sets`` —
to get state-grouping sets.  This module owns the single source of truth for
which ConversationStates belong to each logical phase.
"""
from __future__ import annotations

from app.state_machine.models.conversation_state import ConversationState


ADD_ITEM_FLOW_STATES: set[ConversationState] = {
    ConversationState.CONFIRMING_ITEM,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_QUANTITY,
}

ORDER_FLOW_STATES: set[ConversationState] = {
    ConversationState.CONFIRMING_ORDER,
    ConversationState.WAITING_FOR_PAYMENT,
}

DELIVERY_GATING_STATES: set[ConversationState] = {
    ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
    ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
}

MID_ITEM_BLOCKING_STATES: set[ConversationState] = {
    ConversationState.CONFIRMING_ITEM,
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
    ConversationState.REMOVING_ITEM,
    ConversationState.MODIFYING_ITEM,
}

ACTIVE_TASK_STATES: set[ConversationState] = {
    ConversationState.WAITING_FOR_ORDER_TYPE,
    ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
    ConversationState.WAITING_FOR_DELIVERY_ADDRESS_COLLECTION,
    ConversationState.CONFIRMING_ITEM,
    ConversationState.WAITING_FOR_SIDE,
    ConversationState.WAITING_FOR_SIDE_SIZE,
    ConversationState.WAITING_FOR_MODIFIER,
    ConversationState.WAITING_FOR_SIZE,
    ConversationState.WAITING_FOR_QUANTITY,
    ConversationState.REMOVING_ITEM,
    ConversationState.MODIFYING_ITEM,
    ConversationState.CONFIRMING_ORDER,
    ConversationState.WAITING_FOR_PAYMENT,
    ConversationState.WAITING_FOR_CHECKOUT_COMPLETION,
}

__all__ = [
    "ADD_ITEM_FLOW_STATES",
    "ORDER_FLOW_STATES",
    "DELIVERY_GATING_STATES",
    "MID_ITEM_BLOCKING_STATES",
    "ACTIVE_TASK_STATES",
]
