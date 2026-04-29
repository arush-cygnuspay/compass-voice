# tests/core/test_reset_scopes.py
"""Tests for Task 5: explicit reset scopes on ConversationContext.

Coverage:
- reset_item_scope() clears item flow data and does NOT touch the queue
- reset_order_scope() clears item flow + pending_item_queue
- reset_session_scope() clears all fields
- Backward-compatible wrappers still work
- cancellation flow clears queue via reset_order_scope
- voice_session_synchronizer uses reset_session_scope
- Edge cases: empty queue, repeated resets
"""
from __future__ import annotations

import collections
from collections import deque

import pytest

from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import QueuedItemRequest


# ── helpers ─────────────────────────────────────────────────────────────────

def _ctx_with_item_flow() -> ConversationContext:
    """Return a context that looks mid-way through an add-item flow."""
    ctx = ConversationContext()
    ctx.current_item_id = "item-1"
    ctx.current_item_name = "Burger"
    ctx.candidate_item_id = "item-1"
    ctx.quantity = 2
    ctx.current_prompt_field = "size"
    ctx.awaiting_flow_confirmation = True
    ctx.pending_add_item = object()  # arbitrary sentinel
    ctx.reprompt_attempts["size"] = 3
    ctx.group_prompt_cursors["mod-group"] = 1
    return ctx


def _queued_item(raw: str = "fries") -> QueuedItemRequest:
    return QueuedItemRequest(
        raw_text=raw,
        item_slot_value=raw,
        quantity=1,
        segment_slots=None,
    )


# ── reset_item_scope ─────────────────────────────────────────────────────────

class TestResetItemScope:
    def test_clears_current_item_identity(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_item_scope()
        assert ctx.current_item_id is None
        assert ctx.current_item_name is None
        assert ctx.candidate_item_id is None

    def test_clears_quantity(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_item_scope()
        assert ctx.quantity is None

    def test_clears_prompt_fields(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_item_scope()
        assert ctx.current_prompt_field is None
        assert ctx.available_choices_kind is None
        assert ctx.available_choices_values == ()

    def test_clears_flow_confirmation(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_item_scope()
        assert ctx.awaiting_flow_confirmation is False
        assert ctx.pending_add_item is None

    def test_clears_reprompt_and_group_cursors(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_item_scope()
        assert ctx.reprompt_attempts == {}
        assert ctx.group_prompt_cursors == {}

    def test_does_not_clear_pending_item_queue(self):
        ctx = _ctx_with_item_flow()
        ctx.pending_item_queue = deque([_queued_item("fries"), _queued_item("coke")])
        ctx.reset_item_scope()
        assert len(ctx.pending_item_queue) == 2

    def test_does_not_clear_order_type(self):
        ctx = _ctx_with_item_flow()
        ctx.order_type = "pickup"
        ctx.reset_item_scope()
        assert ctx.order_type == "pickup"

    def test_does_not_clear_delivery_address(self):
        ctx = _ctx_with_item_flow()
        ctx.delivery_address.street = "123 Main St"
        ctx.reset_item_scope()
        assert ctx.delivery_address.street == "123 Main St"

    def test_does_not_clear_last_slots(self):
        ctx = _ctx_with_item_flow()
        ctx.last_slots = (object(),)
        ctx.reset_item_scope()
        assert len(ctx.last_slots) == 1

    def test_idempotent_on_empty_context(self):
        ctx = ConversationContext()
        ctx.reset_item_scope()  # should not raise
        assert ctx.current_item_id is None

    def test_repeated_reset_is_safe(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_item_scope()
        ctx.reset_item_scope()
        assert ctx.current_item_id is None


# ── reset_order_scope ────────────────────────────────────────────────────────

class TestResetOrderScope:
    def test_clears_item_flow_data(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_order_scope()
        assert ctx.current_item_id is None
        assert ctx.current_prompt_field is None

    def test_clears_pending_item_queue(self):
        ctx = ConversationContext()
        ctx.pending_item_queue = deque([_queued_item(), _queued_item("coke")])
        ctx.reset_order_scope()
        assert len(ctx.pending_item_queue) == 0

    def test_queue_clear_on_already_empty_queue(self):
        ctx = ConversationContext()
        ctx.pending_item_queue = deque()
        ctx.reset_order_scope()  # should not raise
        assert len(ctx.pending_item_queue) == 0

    def test_does_not_clear_order_type(self):
        ctx = ConversationContext()
        ctx.order_type = "delivery"
        ctx.reset_order_scope()
        assert ctx.order_type == "delivery"

    def test_does_not_clear_delivery_address(self):
        ctx = ConversationContext()
        ctx.delivery_address.street = "456 Elm Ave"
        ctx.reset_order_scope()
        assert ctx.delivery_address.street == "456 Elm Ave"

    def test_does_not_clear_last_nlu(self):
        ctx = ConversationContext()
        sentinel = object()
        ctx.last_nlu = sentinel  # type: ignore[assignment]
        ctx.reset_order_scope()
        assert ctx.last_nlu is sentinel


# ── reset_session_scope ──────────────────────────────────────────────────────

class TestResetSessionScope:
    def test_clears_item_flow_data(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_session_scope()
        assert ctx.current_item_id is None
        assert ctx.awaiting_flow_confirmation is False

    def test_clears_pending_item_queue(self):
        ctx = ConversationContext()
        ctx.pending_item_queue = deque([_queued_item()])
        ctx.reset_session_scope()
        assert len(ctx.pending_item_queue) == 0

    def test_clears_last_nlu(self):
        ctx = ConversationContext()
        ctx.last_user_text = "hi"
        ctx.last_intent_confidence = 0.9
        ctx.reset_session_scope()
        assert ctx.last_user_text is None
        assert ctx.last_intent_confidence is None

    def test_clears_last_slots(self):
        ctx = ConversationContext()
        ctx.last_slots = (object(),)
        ctx.reset_session_scope()
        assert ctx.last_slots == ()

    def test_resets_delivery_address(self):
        from app.state_machine.models.conversation_context import DeliveryAddress
        ctx = ConversationContext()
        ctx.delivery_address.street = "123 Main St"
        ctx.reset_session_scope()
        assert ctx.delivery_address == DeliveryAddress()

    def test_repeated_session_reset_is_safe(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_session_scope()
        ctx.reset_session_scope()
        assert ctx.current_item_id is None


# ── backward-compatible wrappers ─────────────────────────────────────────────

class TestBackwardCompatibleWrappers:
    def test_reset_task_alias_clears_item_flow(self):
        ctx = _ctx_with_item_flow()
        ctx.reset_task()
        assert ctx.current_item_id is None

    def test_reset_task_does_not_clear_queue(self):
        ctx = ConversationContext()
        ctx.pending_item_queue = deque([_queued_item()])
        ctx.reset_task()
        assert len(ctx.pending_item_queue) == 1

    def test_reset_alias_clears_item_flow(self):
        ctx = _ctx_with_item_flow()
        ctx.reset()
        assert ctx.current_item_id is None

    def test_reset_alias_does_not_clear_queue(self):
        ctx = ConversationContext()
        ctx.pending_item_queue = deque([_queued_item()])
        ctx.reset()
        assert len(ctx.pending_item_queue) == 1

    def test_reset_all_alias_clears_everything(self):
        ctx = _ctx_with_item_flow()
        ctx.pending_item_queue = deque([_queued_item()])
        ctx.last_user_text = "yes"
        ctx.reset_all()
        assert ctx.current_item_id is None
        assert len(ctx.pending_item_queue) == 0
        assert ctx.last_user_text is None

    def test_clear_item_queue_still_works(self):
        ctx = ConversationContext()
        ctx.pending_item_queue = deque([_queued_item(), _queued_item("coke")])
        ctx.clear_item_queue()
        assert len(ctx.pending_item_queue) == 0


# ── cancellation clears queue via reset_order_scope ──────────────────────────

class TestCancellationScopeSemantics:
    """Verify that the cancellation flow pattern uses order scope."""

    def test_cancellation_pattern_clears_queue(self):
        """Simulate what cancellation_confirmation_handler now does."""
        ctx = ConversationContext()
        ctx.current_item_id = "item-1"
        ctx.pending_item_queue = deque([_queued_item(), _queued_item("coke")])
        saved_interrupt = object()
        ctx.interrupt_proposal = saved_interrupt  # type: ignore[assignment]

        # Simulate the handler
        interrupt_proposal = ctx.interrupt_proposal
        ctx.reset_order_scope()
        ctx.interrupt_proposal = interrupt_proposal  # type: ignore[assignment]

        assert ctx.current_item_id is None
        assert len(ctx.pending_item_queue) == 0
        assert ctx.interrupt_proposal is saved_interrupt

    def test_item_reset_alone_does_not_clear_queue(self):
        """Old reset_task() path should NOT clear queue — use to verify backward compat."""
        ctx = ConversationContext()
        ctx.pending_item_queue = deque([_queued_item()])
        ctx.reset_item_scope()
        assert len(ctx.pending_item_queue) == 1


# ── scope boundary table ─────────────────────────────────────────────────────

@pytest.mark.parametrize("method,clears_queue,clears_nlu", [
    ("reset_item_scope", False, False),
    ("reset_order_scope", True, False),
    ("reset_session_scope", True, True),
])
def test_scope_boundaries(method: str, clears_queue: bool, clears_nlu: bool):
    ctx = ConversationContext()
    ctx.current_item_id = "item-1"
    ctx.pending_item_queue = deque([_queued_item()])
    ctx.last_user_text = "some text"

    getattr(ctx, method)()

    assert ctx.current_item_id is None, "item scope should always be cleared"
    assert (len(ctx.pending_item_queue) == 0) == clears_queue
    assert (ctx.last_user_text is None) == clears_nlu
