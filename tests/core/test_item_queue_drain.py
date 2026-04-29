# tests/core/test_item_queue_drain.py
"""
Tests for ItemQueueService.try_drain iterative implementation.

Coverage:
- Empty queue / guard conditions → returns None
- Single-item queue: needs user input
- Single-item queue: instantly added, queue exhausted
- Normal multi-item queue: FIFO order, correct payloads
- Chain of instant adds: chain_prev_items matches former recursive behaviour
- Queue exactly at MAX_QUEUE_DEPTH succeeds (no overflow log)
- Queue over MAX_QUEUE_DEPTH: excess rejected and logged
- Overflow truncates tail items, head order preserved
- No recursion in drain path (low-limit recursion stress test)
- Edge cases: None response_payload, quantity injection, state not mutated
"""
from __future__ import annotations

import sys
import types
from collections import deque
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.command_executor import CommandExecutor
from app.core.item_queue_service import MAX_QUEUE_DEPTH, ItemQueueService
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import QueuedItemRequest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_session(queue: list[QueuedItemRequest] | None = None) -> Session:
    s = Session(session_id="sess_test", restaurant_id="r1")
    s.conversation_state = ConversationState.IDLE
    if queue:
        s.conversation_context.pending_item_queue = deque(queue)
    return s


def _queued(raw: str, item_slot: str = "", qty: int = 1) -> QueuedItemRequest:
    return QueuedItemRequest(
        raw_text=raw,
        item_slot_value=item_slot or raw,
        quantity=qty,
        segment_slots=None,
    )


def _instant_result(item_name: str = "Burger", qty: int = 1) -> HandlerResult:
    return HandlerResult(
        next_state=ConversationState.IDLE,
        response_key="item_added_successfully",
        response_payload={"item_name": item_name, "quantity": qty},
    )


def _waiting_result(
    state: ConversationState = ConversationState.WAITING_FOR_SIZE,
) -> HandlerResult:
    return HandlerResult(
        next_state=state,
        response_key="ask_for_size",
        response_payload={"item_name": "BigMac", "quantity": 1},
    )


def _make_service(handler_results: list[HandlerResult]) -> ItemQueueService:
    mock_handler = MagicMock()
    mock_handler.handle.side_effect = handler_results
    return ItemQueueService(
        handlers={"add_item_handler": mock_handler},
        command_executor=MagicMock(spec=CommandExecutor),
    )


def _seed_result(item_name: str = "Burger") -> HandlerResult:
    """Simulates current_result — the item just added before drain is called."""
    return HandlerResult(
        next_state=ConversationState.IDLE,
        response_key="item_added_successfully",
        response_payload={"item_name": item_name, "quantity": 1},
    )


# ── entry guards ──────────────────────────────────────────────────────────────

def test_returns_none_when_response_key_is_not_item_added():
    session = _make_session([_queued("fries")])
    result = _make_service([]).try_drain(
        session=session,
        current_result=HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="ask_for_size",
        ),
    )
    assert result is None


def test_returns_none_when_next_state_is_not_idle():
    session = _make_session([_queued("fries")])
    result = _make_service([]).try_drain(
        session=session,
        current_result=HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIZE,
            response_key="item_added_successfully",
        ),
    )
    assert result is None


def test_returns_none_when_queue_is_empty():
    session = _make_session(queue=None)
    result = _make_service([]).try_drain(
        session=session,
        current_result=_seed_result(),
    )
    assert result is None


def test_returns_none_when_add_handler_missing():
    service = ItemQueueService(
        handlers={},
        command_executor=MagicMock(spec=CommandExecutor),
    )
    session = _make_session([_queued("fries")])
    result = service.try_drain(session=session, current_result=_seed_result())
    assert result is None


# ── single-item queue ─────────────────────────────────────────────────────────

def test_single_item_needs_user_input():
    service = _make_service([_waiting_result(ConversationState.WAITING_FOR_SIZE)])
    session = _make_session([_queued("bigmac")])

    result = service.try_drain(session=session, current_result=_seed_result("Burger"))

    assert result is not None
    assert result.next_state == ConversationState.WAITING_FOR_SIZE
    assert result.response_key == "ask_for_size"
    payload = result.response_payload
    assert payload["queue_transition"] is True
    assert payload["prev_item_name"] == "Burger"
    assert payload["next_item_name"] == "bigmac"
    assert payload["remaining_queue_count"] == 0
    assert "chain_prev_items" not in payload


def test_single_item_instantly_added():
    service = _make_service([_instant_result("Fries")])
    session = _make_session([_queued("fries")])

    result = service.try_drain(session=session, current_result=_seed_result("Burger"))

    assert result is not None
    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_added_successfully"
    payload = result.response_payload
    assert payload["prev_item_name"] == "Burger"
    assert payload["next_item_name"] == "fries"
    assert payload["remaining_queue_count"] == 0
    assert "chain_prev_items" not in payload


# ── multi-item queue ──────────────────────────────────────────────────────────

def test_two_hop_second_needs_user_input():
    """A(seed) → B(instant) → C(needs input): result is for C."""
    service = _make_service([_instant_result("Fries"), _waiting_result()])
    session = _make_session([_queued("fries"), _queued("bigmac")])

    result = service.try_drain(session=session, current_result=_seed_result("Burger"))

    assert result is not None
    assert result.next_state == ConversationState.WAITING_FOR_SIZE
    payload = result.response_payload
    assert payload["prev_item_name"] == "Fries"   # B was last added
    assert payload["next_item_name"] == "bigmac"  # C is next
    assert payload["remaining_queue_count"] == 0
    assert payload["chain_prev_items"] == [{"name": "Burger", "quantity": 1}]


def test_two_hop_all_instantly_added():
    """A(seed) → B(instant) → C(instant), queue empty: response is for C."""
    service = _make_service([_instant_result("Fries"), _instant_result("Coke")])
    session = _make_session([_queued("fries"), _queued("coke")])

    result = service.try_drain(session=session, current_result=_seed_result("Burger"))

    assert result is not None
    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_added_successfully"
    payload = result.response_payload
    assert payload["prev_item_name"] == "Fries"
    assert payload["next_item_name"] == "coke"
    assert payload["chain_prev_items"] == [{"name": "Burger", "quantity": 1}]


def test_three_hop_chain_prev_items_is_first_prev_only():
    """A→B(instant)→C(instant)→D(needs input): chain_prev_items=[A] only.

    The former recursive implementation always overwrote chain_prev_items with
    the outermost caller's prev_item.  The iterative version preserves this
    by using chain_prev_items[0].
    """
    service = _make_service([
        _instant_result("Fries"),  # B
        _instant_result("Coke"),   # C
        _waiting_result(),         # D
    ])
    session = _make_session([_queued("fries"), _queued("coke"), _queued("bigmac")])

    result = service.try_drain(session=session, current_result=_seed_result("Burger"))

    assert result is not None
    payload = result.response_payload
    assert payload["chain_prev_items"] == [{"name": "Burger", "quantity": 1}]
    assert payload["prev_item_name"] == "Coke"
    assert payload["next_item_name"] == "bigmac"


def test_queue_consumed_in_fifo_order():
    """Items must be processed left-to-right (deque popleft)."""
    processed_texts: list[str] = []

    def _handler(*, intent, context, user_text, session):
        processed_texts.append(user_text)
        return _waiting_result()  # stop after first

    mock = MagicMock()
    mock.handle.side_effect = _handler
    service = ItemQueueService(
        handlers={"add_item_handler": mock},
        command_executor=MagicMock(spec=CommandExecutor),
    )

    session = _make_session([_queued("first"), _queued("second"), _queued("third")])
    service.try_drain(session=session, current_result=_seed_result())

    assert processed_texts == ["first"]


def test_remaining_queue_count_is_queue_size_after_pop():
    service = _make_service([_waiting_result()])
    session = _make_session([_queued("a"), _queued("b"), _queued("c")])

    result = service.try_drain(session=session, current_result=_seed_result())

    assert result is not None
    assert result.response_payload["remaining_queue_count"] == 2  # b, c remain


# ── commands and context ──────────────────────────────────────────────────────

def test_commands_applied_inside_loop():
    result_with_cmd = HandlerResult(
        next_state=ConversationState.WAITING_FOR_SIZE,
        response_key="ask_for_size",
        response_payload={"item_name": "BigMac", "quantity": 1},
        command={"type": "SOME_COMMAND"},
    )
    service = _make_service([result_with_cmd])
    session = _make_session([_queued("bigmac")])

    service.try_drain(session=session, current_result=_seed_result())

    service.command_executor.execute.assert_called_once()


def test_returned_result_command_is_none():
    """Returned HandlerResult.command must be None — applied inside the loop."""
    service = _make_service([_waiting_result()])
    session = _make_session([_queued("bigmac")])

    result = service.try_drain(session=session, current_result=_seed_result())

    assert result is not None
    assert result.command is None


def test_returned_result_reset_context_is_false():
    service = _make_service([_waiting_result()])
    session = _make_session([_queued("bigmac")])

    result = service.try_drain(session=session, current_result=_seed_result())

    assert result is not None
    assert result.reset_context is False


def test_queued_item_quantity_injected_into_context():
    captured: list[int] = []

    def _handler(*, intent, context, user_text, session):
        captured.append(context.quantity)
        return _waiting_result()

    mock = MagicMock()
    mock.handle.side_effect = _handler
    service = ItemQueueService(
        handlers={"add_item_handler": mock},
        command_executor=MagicMock(spec=CommandExecutor),
    )

    session = _make_session([_queued("fries", qty=3)])
    service.try_drain(session=session, current_result=_seed_result())

    assert captured == [3]


def test_session_state_not_mutated_by_drain():
    """try_drain must NOT write session.conversation_state."""
    service = _make_service([_waiting_result()])
    session = _make_session([_queued("bigmac")])
    session.conversation_state = ConversationState.IDLE

    service.try_drain(session=session, current_result=_seed_result())

    assert session.conversation_state == ConversationState.IDLE


# ── overflow ──────────────────────────────────────────────────────────────────

def test_queue_exactly_at_max_depth_no_overflow_log(capsys):
    queue = [_queued(f"item_{i}") for i in range(MAX_QUEUE_DEPTH)]
    service = _make_service([_waiting_result()])  # stop after first item
    session = _make_session(queue)

    service.try_drain(session=session, current_result=_seed_result())

    assert "ITEM_QUEUE_OVERFLOW" not in capsys.readouterr().out


def test_queue_over_max_depth_logs_overflow(capsys):
    queue = [_queued(f"item_{i}") for i in range(MAX_QUEUE_DEPTH + 5)]
    service = _make_service([_waiting_result()])
    session = _make_session(queue)

    service.try_drain(session=session, current_result=_seed_result())

    assert "ITEM_QUEUE_OVERFLOW" in capsys.readouterr().out


def test_overflow_log_contains_session_id(capsys):
    queue = [_queued(f"item_{i}") for i in range(MAX_QUEUE_DEPTH + 2)]
    service = _make_service([_waiting_result()])
    session = _make_session(queue)
    session.session_id = "overflow_session_99"

    service.try_drain(session=session, current_result=_seed_result())

    assert "overflow_session_99" in capsys.readouterr().out


def test_overflow_truncates_tail_preserves_head_order():
    """Overflow items are removed from the tail; head order is preserved."""
    n = MAX_QUEUE_DEPTH + 5
    queue = [_queued(f"item_{i}") for i in range(n)]
    service = _make_service([_waiting_result()])
    session = _make_session(queue)

    service.try_drain(session=session, current_result=_seed_result())

    # Handler was called with item_0 (head of queue)
    call_text = service.handlers["add_item_handler"].handle.call_args[1]["user_text"]
    assert call_text == "item_0"


def test_overflow_queue_shrinks_to_max_depth_minus_one():
    """After drain, queue should have MAX_QUEUE_DEPTH - 1 items remaining."""
    n = MAX_QUEUE_DEPTH + 10
    queue = [_queued(f"item_{i}") for i in range(n)]
    service = _make_service([_waiting_result()])  # stops after first
    session = _make_session(queue)

    service.try_drain(session=session, current_result=_seed_result())

    remaining = len(session.conversation_context.pending_item_queue)
    assert remaining == MAX_QUEUE_DEPTH - 1


def test_drain_still_returns_result_when_overflow_detected():
    queue = [_queued(f"item_{i}") for i in range(MAX_QUEUE_DEPTH + 3)]
    service = _make_service([_waiting_result()])
    session = _make_session(queue)

    result = service.try_drain(session=session, current_result=_seed_result())

    assert result is not None


# ── no recursion ──────────────────────────────────────────────────────────────

def test_iterative_drain_does_not_use_recursion():
    """With Python recursion limit at 50, a 5-item instant chain must not crash."""
    original = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(50)

        handler_results = [_instant_result(f"Item{i}") for i in range(4)]
        handler_results.append(_waiting_result())
        service = _make_service(handler_results)

        session = _make_session([_queued(f"item_{i}") for i in range(5)])
        result = service.try_drain(session=session, current_result=_seed_result("Seed"))

        assert result is not None
    finally:
        sys.setrecursionlimit(original)


# ── edge cases ────────────────────────────────────────────────────────────────

def test_none_response_payload_handled_gracefully():
    result_no_payload = HandlerResult(
        next_state=ConversationState.WAITING_FOR_SIZE,
        response_key="ask_for_size",
        response_payload=None,
    )
    service = _make_service([result_no_payload])
    session = _make_session([_queued("bigmac")])

    result = service.try_drain(session=session, current_result=_seed_result())

    assert result is not None
    assert result.response_payload["queue_transition"] is True


def test_current_result_none_payload_defaults_gracefully():
    seed = HandlerResult(
        next_state=ConversationState.IDLE,
        response_key="item_added_successfully",
        response_payload=None,  # no item_name — should default to "item"
    )
    service = _make_service([_waiting_result()])
    session = _make_session([_queued("fries")])

    result = service.try_drain(session=session, current_result=seed)

    assert result is not None
    assert result.response_payload["prev_item_name"] == "item"


def test_max_queue_depth_is_positive_integer():
    assert isinstance(MAX_QUEUE_DEPTH, int)
    assert MAX_QUEUE_DEPTH > 0


def test_max_queue_depth_configurable_via_env(monkeypatch):
    monkeypatch.setenv("COMPASS_MAX_ITEM_QUEUE_DEPTH", "7")
    import importlib
    import app.core.item_queue_service as module
    from app.config.nlu import get_nlu_config
    get_nlu_config.cache_clear()
    importlib.reload(module)
    try:
        assert module.MAX_QUEUE_DEPTH == 7
    finally:
        get_nlu_config.cache_clear()
        importlib.reload(module)
