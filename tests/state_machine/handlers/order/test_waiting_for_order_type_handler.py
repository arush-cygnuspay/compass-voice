"""Tests for WaitingForOrderTypeHandler.

Verifies:
- pickup and delivery resolved via OrderTypeResolver (no local word lists)
- unknown answer returns repeat_order_type
- ordering intent before order type is set returns ordering_blocked_need_order_type
- item-like text (ITEM slot) also triggers the redirect
- context.order_type / delivery_address_required mutated correctly
- no regression on next_state transitions
"""
from __future__ import annotations

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.state_machine.handlers.order.waiting_for_order_type_handler import (
    WaitingForOrderTypeHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


def _ctx() -> ConversationContext:
    return ConversationContext()


def _handle(user_text: str, intent: Intent = Intent.UNKNOWN, context: ConversationContext | None = None):
    ctx = context or _ctx()
    result = WaitingForOrderTypeHandler().handle(
        intent=intent,
        context=ctx,
        user_text=user_text,
        session=None,
    )
    return result, ctx


# ---------------------------------------------------------------------------
# Pickup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "pickup",
    "pick up",
    "for pickup",
    "pickup please",
    "takeout",
    "take out",
    "carryout",
    "carry out",
    "I'll pick it up",
    "I'll grab it",
    "come get it",
])
def test_pickup_phrases_resolve_to_idle(text):
    result, ctx = _handle(text)
    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "order_type_captured_pickup"
    assert ctx.order_type == "pickup"
    assert ctx.delivery_address_required is False
    assert ctx.onboarding_complete is True


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "delivery",
    "for delivery",
    "deliver it",
    "drop it off",
    "drop off",
    "send it",
    "bring it",
    "delivery please",
])
def test_delivery_phrases_resolve_to_delivery_eligibility(text):
    result, ctx = _handle(text)
    assert result.next_state == ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY
    assert result.response_key == "ask_for_delivery_area"
    assert ctx.order_type == "delivery"
    assert ctx.delivery_address_required is True
    assert ctx.onboarding_complete is False


# ---------------------------------------------------------------------------
# Unknown answer → repeat prompt, context unchanged
# ---------------------------------------------------------------------------

def test_unknown_text_returns_repeat_order_type():
    # "not sure" has no ordering prefix and no order-type keyword → repeat prompt
    result, ctx = _handle("not sure yet")
    assert result.next_state == ConversationState.WAITING_FOR_ORDER_TYPE
    assert result.response_key == "repeat_order_type"
    assert ctx.order_type is None


def test_empty_text_returns_repeat_order_type():
    result, ctx = _handle("")
    assert result.next_state == ConversationState.WAITING_FOR_ORDER_TYPE
    assert result.response_key == "repeat_order_type"


# ---------------------------------------------------------------------------
# Ordering attempt before order type selected → redirect
# ---------------------------------------------------------------------------

def test_add_item_intent_blocks_and_asks_for_order_type():
    result, ctx = _handle("I want a burger", intent=Intent.ADD_ITEM)
    assert result.next_state == ConversationState.WAITING_FOR_ORDER_TYPE
    assert result.response_key == "ordering_blocked_need_order_type"
    assert ctx.order_type is None


def test_item_slot_in_context_triggers_ordering_redirect():
    ctx = _ctx()
    ctx.last_slots = (SlotValue(name="ITEM", value="Chicken Burger"),)
    result, _ = _handle("chicken burger", context=ctx)
    assert result.next_state == ConversationState.WAITING_FOR_ORDER_TYPE
    assert result.response_key == "ordering_blocked_need_order_type"


def test_ordering_prefix_triggers_redirect():
    result, ctx = _handle("can I get a zinger")
    assert result.next_state == ConversationState.WAITING_FOR_ORDER_TYPE
    assert result.response_key == "ordering_blocked_need_order_type"


# ---------------------------------------------------------------------------
# Context mutations — pickup branch
# ---------------------------------------------------------------------------

def test_pickup_sets_onboarding_complete():
    _, ctx = _handle("pickup")
    assert ctx.onboarding_complete is True


def test_pickup_clears_prompt_field():
    ctx = _ctx()
    ctx.current_prompt_field = "some_field"
    _handle("pickup", context=ctx)
    assert ctx.current_prompt_field is None


# ---------------------------------------------------------------------------
# Context mutations — delivery branch
# ---------------------------------------------------------------------------

def test_delivery_leaves_onboarding_incomplete():
    _, ctx = _handle("delivery")
    assert ctx.onboarding_complete is False


def test_delivery_sets_delivery_address_required():
    _, ctx = _handle("delivery")
    assert ctx.delivery_address_required is True


def test_delivery_confirmed_reset_to_false():
    ctx = _ctx()
    ctx.delivery_address_confirmed = True
    _handle("delivery", context=ctx)
    assert ctx.delivery_address_confirmed is False


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

def test_uppercase_pickup_resolves():
    result, ctx = _handle("PICKUP")
    assert result.next_state == ConversationState.IDLE
    assert ctx.order_type == "pickup"


def test_uppercase_delivery_resolves():
    result, ctx = _handle("DELIVERY")
    assert result.next_state == ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY
    assert ctx.order_type == "delivery"
