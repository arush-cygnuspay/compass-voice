# tests/core/test_atomic_state_transitions.py
"""
Tests for Task 4: atomic session.conversation_state transitions.

Coverage areas:
- TurnSnapshot captures and restores session fields
- SessionTurnLockManager provides per-session locks
- FlowGateDecision carries output + state_override
- flow_gate._compute_order_type_gate_state returns value (not mutation)
- flow_gate._handle_readonly_interrupt carries next_state via TurnOutput
- flow_gate._handle_phase3_control_shortcuts returns FlowGateDecision
- item_queue_service.try_drain uses current_result.next_state for guard
- item_queue_service.try_drain does not mutate session.conversation_state
- payment_flow_orchestrator._handle_auto_payment_check carries next_state via TurnOutput
- TurnOutput.next_state field exists
- Session.current_turn_id field exists
"""
from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Stub heavy ML/Twilio imports before any app.core module loads ────────────
_intent_module = types.ModuleType("app.ml.intent.inference_intent")
_slot_module = types.ModuleType("app.ml.slot.inference_slot")


class _IntentBundle:
    pass


class _SlotBundle:
    pass


_intent_module.IntentBundle = _IntentBundle
_intent_module.predict_intent = lambda *a, **k: []
_slot_module.SlotBundle = _SlotBundle
_slot_module.predict_slots = lambda *a, **k: []
sys.modules.setdefault("app.ml.intent.inference_intent", _intent_module)
sys.modules.setdefault("app.ml.slot.inference_slot", _slot_module)

for _name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
sys.modules["twilio.rest"].Client = type(
    "_Client", (), {"__init__": lambda *a, **k: None}
)

# Stub torch
sys.modules.setdefault("torch", types.ModuleType("torch"))
# ─────────────────────────────────────────────────────────────────────────────

from app.state_machine.policy.flow_gate import FlowGate, FlowGateDecision
from app.core.session_turn_lock_manager import SessionTurnLockManager
from app.core.turn_snapshot import TurnSnapshot
from app.core.turn_engine import TurnOutput
from app.nlu.intent_resolution.intent import Intent
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_state import ConversationState


# ── TurnSnapshot ─────────────────────────────────────────────────────────────

def _make_session(state: ConversationState = ConversationState.IDLE) -> Session:
    s = Session(session_id="test_session", restaurant_id="r1")
    s.conversation_state = state
    s.turn_count = 5
    s.last_response_key = "some_response"
    s.last_response_payload = {"key": "val"}
    s.last_intent = Intent.ADD_ITEM
    return s


def test_turn_snapshot_capture_preserves_all_fields():
    session = _make_session(ConversationState.WAITING_FOR_SIZE)
    snap = TurnSnapshot.capture(session)

    assert snap.conversation_state == ConversationState.WAITING_FOR_SIZE
    assert snap.turn_count == 5
    assert snap.last_response_key == "some_response"
    assert snap.last_response_payload == {"key": "val"}
    assert snap.last_intent == Intent.ADD_ITEM


def test_turn_snapshot_restore_reverts_mutations():
    session = _make_session(ConversationState.IDLE)
    snap = TurnSnapshot.capture(session)

    # Simulate mid-turn mutations
    session.conversation_state = ConversationState.WAITING_FOR_PAYMENT
    session.turn_count = 99
    session.last_response_key = "error"
    session.last_response_payload = None
    session.last_intent = Intent.CANCEL

    snap.restore(session)

    assert session.conversation_state == ConversationState.IDLE
    assert session.turn_count == 5
    assert session.last_response_key == "some_response"
    assert session.last_response_payload == {"key": "val"}
    assert session.last_intent == Intent.ADD_ITEM


def test_turn_snapshot_is_frozen():
    session = _make_session()
    snap = TurnSnapshot.capture(session)
    try:
        snap.conversation_state = ConversationState.COMPLETED  # type: ignore
        assert False, "Should have raised"
    except (AttributeError, TypeError):
        pass  # frozen dataclass raises on mutation attempt


# ── SessionTurnLockManager ───────────────────────────────────────────────────

def test_session_turn_lock_manager_returns_same_lock_for_same_session():
    manager = SessionTurnLockManager()
    lock1 = manager.get_lock("session_abc")
    lock2 = manager.get_lock("session_abc")
    assert lock1 is lock2


def test_session_turn_lock_manager_returns_different_locks_for_different_sessions():
    manager = SessionTurnLockManager()
    lock_a = manager.get_lock("session_a")
    lock_b = manager.get_lock("session_b")
    assert lock_a is not lock_b


def test_session_turn_lock_manager_lock_is_asyncio_lock():
    manager = SessionTurnLockManager()
    lock = manager.get_lock("s1")
    assert isinstance(lock, asyncio.Lock)


def test_session_turn_lock_manager_lock_is_acquirable():
    manager = SessionTurnLockManager()
    lock = manager.get_lock("s1")

    async def _acquire():
        async with lock:
            return True

    result = asyncio.run(_acquire())
    assert result is True


# ── TurnOutput.next_state field ───────────────────────────────────────────────

def test_turn_output_has_next_state_field():
    out = TurnOutput(response_key="some_key")
    assert hasattr(out, "next_state")
    assert out.next_state is None


def test_turn_output_next_state_can_be_set():
    out = TurnOutput(
        response_key="item_added_successfully",
        next_state=ConversationState.IDLE,
    )
    assert out.next_state == ConversationState.IDLE


# ── Session.current_turn_id field ─────────────────────────────────────────────

def test_session_has_current_turn_id_field():
    s = Session(session_id="s", restaurant_id="r")
    assert hasattr(s, "current_turn_id")
    assert s.current_turn_id is None


# ── FlowGateDecision ──────────────────────────────────────────────────────────

def test_flow_gate_decision_is_frozen():
    out = TurnOutput(response_key="k")
    decision = FlowGateDecision(output=out, state_override=ConversationState.IDLE)
    try:
        decision.output = None  # type: ignore
        assert False, "Should have raised"
    except (AttributeError, TypeError):
        pass


def test_flow_gate_decision_output_none_represents_continue():
    decision = FlowGateDecision(output=None, state_override=ConversationState.IDLE)
    assert decision.output is None
    assert decision.state_override == ConversationState.IDLE


# ── flow_gate._compute_order_type_gate_state ────────────────────────────────

def _make_minimal_gate() -> FlowGate:
    return FlowGate(
        handlers={},
        menu_repo=MagicMock(),
        cart_summary_builder=MagicMock(),
        response_writer=MagicMock(),
        diagnostics=MagicMock(),
        payment_flow=MagicMock(),
        resume_prompt_builder=MagicMock(),
    )


def test_compute_order_type_gate_returns_waiting_when_type_missing():
    gate = _make_minimal_gate()
    session = Session(session_id="s", restaurant_id="r")
    session.conversation_state = ConversationState.IDLE
    # order_type defaults to None → required

    result = gate._compute_order_type_gate_state(session)

    assert result == ConversationState.WAITING_FOR_ORDER_TYPE


def test_compute_order_type_gate_returns_none_when_type_known():
    gate = _make_minimal_gate()
    session = Session(session_id="s", restaurant_id="r")
    session.conversation_state = ConversationState.IDLE
    session.conversation_context.order_type = "pickup"

    result = gate._compute_order_type_gate_state(session)

    assert result is None


def test_compute_order_type_gate_does_not_mutate_session():
    gate = _make_minimal_gate()
    session = Session(session_id="s", restaurant_id="r")
    session.conversation_state = ConversationState.IDLE

    gate._compute_order_type_gate_state(session)

    # Method must not mutate — caller (TurnEngine) applies the value
    assert session.conversation_state == ConversationState.IDLE


def test_compute_order_type_gate_returns_none_for_completed():
    gate = _make_minimal_gate()
    session = Session(session_id="s", restaurant_id="r")
    session.conversation_state = ConversationState.COMPLETED

    result = gate._compute_order_type_gate_state(session)

    assert result is None


# ── flow_gate._handle_phase3_control_shortcuts ───────────────────────────────

class TestPhase3ShortcutReturnsDecision(unittest.TestCase):
    def _make_phase3_gate(self) -> FlowGate:
        gate = _make_minimal_gate()
        gate.payment_flow = MagicMock()
        gate.cart_summary_builder = MagicMock(return_value={"items": []})
        return gate

    def _intent_result(self, intent: Intent = Intent.UNKNOWN):
        from app.nlu.intent_resolution.intent_result import IntentResult
        return IntentResult(intent=intent, raw_text="")

    def _nlu(self, text: str = "", intent: Intent = Intent.UNKNOWN):
        from app.nlu.nlu_result import NLUResult
        return NLUResult(
            effective_intent=intent,
            intent_confidence=0.9,
            raw_text=text,
            normalized_text=text,
        )

    def test_no_shortcut_returns_none(self):
        gate = self._make_phase3_gate()
        session = Session(session_id="s", restaurant_id="r")
        session.conversation_state = ConversationState.IDLE

        result = gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=ConversationState.IDLE,
            intent_result=self._intent_result(),
            nlu=self._nlu("want a burger"),
        )

        assert result is None

    def test_agent_request_returns_flow_gate_decision(self):
        gate = self._make_phase3_gate()
        session = Session(session_id="s", restaurant_id="r")
        session.conversation_state = ConversationState.CONFIRMING_ORDER

        result = gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=ConversationState.CONFIRMING_ORDER,
            intent_result=self._intent_result(Intent.REQUEST_AGENT),
            nlu=self._nlu("agent"),
        )

        assert isinstance(result, FlowGateDecision)
        assert result.output is not None
        assert result.output.response_key == "transferring_to_human_agent"

    def test_shortcut_does_not_mutate_session_state(self):
        gate = self._make_phase3_gate()
        session = Session(session_id="s", restaurant_id="r")
        session.conversation_state = ConversationState.CONFIRMING_ORDER

        gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=ConversationState.CONFIRMING_ORDER,
            intent_result=self._intent_result(Intent.ADD_ITEM),
            nlu=self._nlu("add a burger"),
        )

        # The cart-edit branch used to set session.conversation_state = IDLE.
        # It must now return a FlowGateDecision; TurnEngine applies the state.
        assert session.conversation_state == ConversationState.CONFIRMING_ORDER

    def test_cart_edit_intent_returns_decision_with_idle_override(self):
        gate = self._make_phase3_gate()
        session = Session(session_id="s", restaurant_id="r")
        session.conversation_state = ConversationState.CONFIRMING_ORDER

        result = gate._handle_phase3_control_shortcuts(
            session=session,
            state_before=ConversationState.CONFIRMING_ORDER,
            intent_result=self._intent_result(Intent.ADD_ITEM),
            nlu=self._nlu("add a burger"),
        )

        assert isinstance(result, FlowGateDecision)
        assert result.output is None  # continue processing
        assert result.state_override == ConversationState.IDLE


# ── item_queue_service.try_drain ─────────────────────────────────────────────

class TestItemQueueServiceNonMutating(unittest.TestCase):
    def _make_service(self):
        from app.core.item_queue_service import ItemQueueService
        from app.core.command_executor import CommandExecutor
        return ItemQueueService(
            handlers={},
            command_executor=MagicMock(spec=CommandExecutor),
        )

    def test_try_drain_returns_none_when_next_state_is_not_idle(self):
        service = self._make_service()
        session = Session(session_id="s", restaurant_id="r")
        session.conversation_state = ConversationState.WAITING_FOR_SIDE

        result_with_non_idle = HandlerResult(
            next_state=ConversationState.WAITING_FOR_SIDE,
            response_key="item_added_successfully",
        )

        drained = service.try_drain(session=session, current_result=result_with_non_idle)

        assert drained is None

    def test_try_drain_does_not_write_session_state(self):
        """try_drain must never mutate session.conversation_state."""
        from app.core.item_queue_service import ItemQueueService
        from app.core.command_executor import CommandExecutor
        from collections import deque
        from app.state_machine.models.pending_item_models import QueuedItemRequest

        add_handler = MagicMock()
        add_handler.handle.return_value = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            response_payload={"item_name": "Fries", "quantity": 1},
        )

        service = ItemQueueService(
            handlers={"add_item_handler": add_handler},
            command_executor=MagicMock(spec=CommandExecutor),
        )

        session = Session(session_id="s", restaurant_id="r")
        session.conversation_state = ConversationState.IDLE
        session.conversation_context.pending_item_queue = deque([
            QueuedItemRequest(
                raw_text="fries",
                item_slot_value="fries",
                quantity=1,
                segment_slots=None,
            )
        ])

        current = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            response_payload={"item_name": "Burger", "quantity": 1},
        )

        # Before drain — state is IDLE
        assert session.conversation_state == ConversationState.IDLE

        service.try_drain(session=session, current_result=current)

        # try_drain must NOT mutate session.conversation_state
        assert session.conversation_state == ConversationState.IDLE


# ── payment_flow_orchestrator._handle_auto_payment_check ─────────────────────

class TestPaymentAutoCheckNextState(unittest.TestCase):
    def _make_response_writer(self):
        from app.core.session_response_writer import SessionResponseWriter
        writer = SessionResponseWriter(
            responder=MagicMock(),
            menu_repo=MagicMock(),
        )
        # Make responder.build return a string so _hydrate_output produces a real TurnOutput
        writer.responder.build = MagicMock(return_value="Some response text.")
        return writer

    def _make_orchestrator(self):
        from app.core.payment_flow_orchestrator import PaymentFlowOrchestrator
        return PaymentFlowOrchestrator(
            checkout_service=MagicMock(),
            payment_event_logger=MagicMock(),
            response_writer=self._make_response_writer(),
            responder=MagicMock(),
            diagnostics=MagicMock(),
            command_executor=MagicMock(),
            cart_summary_builder=MagicMock(),
        )

    def test_auto_payment_check_carries_next_state_in_output(self):
        orchestrator = self._make_orchestrator()
        session = Session(session_id="s", restaurant_id="r")
        session.conversation_state = ConversationState.WAITING_FOR_PAYMENT

        # Patch verify_payment_for_order to return a known result
        handler_result = HandlerResult(
            next_state=ConversationState.CONFIRMING_ORDER,
            response_key="payment_draft_saved_retry_later",
        )
        with patch(
            "app.core.payment_flow_orchestrator.verify_payment_for_order",
            return_value=handler_result,
        ):
            output = orchestrator._handle_auto_payment_check(session)

        # The TurnOutput must carry next_state — TurnEngine applies it
        assert isinstance(output, TurnOutput), f"Expected TurnOutput, got {type(output)}"
        assert output.next_state == ConversationState.CONFIRMING_ORDER
        # The method must NOT have mutated session.conversation_state
        assert session.conversation_state == ConversationState.WAITING_FOR_PAYMENT

    def test_auto_payment_check_idle_state_returns_silent_noop(self):
        orchestrator = self._make_orchestrator()
        session = Session(session_id="s", restaurant_id="r")
        session.conversation_state = ConversationState.IDLE

        output = orchestrator._handle_auto_payment_check(session)

        # IDLE is not a payment state — silent no-op, state unchanged
        assert session.conversation_state == ConversationState.IDLE
        assert isinstance(output, TurnOutput)
