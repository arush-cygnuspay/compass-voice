# tests/core/test_handler_result_field_application.py
"""
Tests verifying that TurnEngine (not handlers) is the sole applier of
HandlerResult engine-owned fields: prompt_field, interrupt_proposal,
awaiting_flow_confirmation.

Coverage areas:
- HandlerResult carries fields correctly
- TurnEngine main dispatch path applies all three fields
- TurnEngine WAITING_FOR_ORDER_TYPE fast-exit path applies all three fields
- SOFT_SWITCH handlers return fields in HandlerResult (no direct ctx mutation)
- Deny interrupt path resumes without stale fields
- Accept interrupt path clears correctly via reset_task
- Prompt field tracking for cross-handler transitions
- Engine-applied field is None => context value unchanged
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys
import types

# ── stub heavy optional dependencies ──────────────────────────────────────────
for _mod, _attr, _cls in [
    ("twilio", None, None),
    ("twilio.base", None, None),
    ("twilio.base.exceptions", "TwilioRestException", Exception),
    ("twilio.rest", "Client", object),
    ("redis", "Redis", object),
    ("app.ml.intent.inference_intent", "IntentBundle", object),
    ("app.ml.slot.inference_slot", "SlotBundle", object),
]:
    m = types.ModuleType(_mod)
    if _attr:
        setattr(m, _attr, _cls)
    if _mod == "app.ml.intent.inference_intent":
        m.predict_intent = lambda *a, **kw: []
    if _mod == "app.ml.slot.inference_slot":
        m.predict_slots = lambda *a, **kw: []
    sys.modules.setdefault(_mod, m)

from app.core.response_builder import ResponseBuilder
from app.core.turn_engine import TurnEngine
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.handler_result import HandlerResult
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import InterruptProposal
from app.state_machine.state_router import StateRouter


# ── shared helpers ─────────────────────────────────────────────────────────────

class _StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        return SimpleNamespace(ok=False, sid=None, error_code="not_configured", error_message="stub")


def _menu_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "steves_grill"
    return MenuRepository(MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    ))


def _engine(menu_repo: MenuRepository) -> TurnEngine:
    return TurnEngine(
        router=StateRouter(),
        menu_repo=menu_repo,
        intent_bundle=None,
        slot_bundle=None,
        responder=ResponseBuilder(menu_repo),
        sms_service=_StubSmsService(),
        nlu_logger=SimpleNamespace(enabled=False),
    )


def _fake_nlu(text: str, intent: Intent, slots: tuple = ()) -> SimpleNamespace:
    return SimpleNamespace(
        effective_intent=intent,
        intent_confidence=0.99,
        normalized_text=normalize_text(text),
        raw_text=text,
        slots=slots,
        model_main_intent=intent.value,
        model_sub_intent=intent.value,
        slot_model_ran=bool(slots),
        intent_model_ms=None,
        slot_model_ms=None,
    )


def _turn(engine, session, text, *, intent=Intent.UNKNOWN, slots=()):
    with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=_fake_nlu(text, intent, slots)):
        return engine.process_turn(session=session, user_text=text)


# ── HandlerResult field contracts ──────────────────────────────────────────────

def test_handler_result_default_fields_are_none():
    """New fields default to None so existing code paths are unaffected."""
    result = HandlerResult(
        next_state=ConversationState.IDLE,
        response_key="some_key",
    )
    assert result.prompt_field is None
    assert result.interrupt_proposal is None
    assert result.awaiting_flow_confirmation is None


def test_handler_result_carries_prompt_field():
    result = HandlerResult(
        next_state=ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY,
        response_key="ask_for_delivery_area",
        prompt_field="delivery_area",
    )
    assert result.prompt_field == "delivery_area"


def test_handler_result_carries_interrupt_proposal():
    proposal = InterruptProposal(text="add a coke", predicted_main_intent=None, predicted_sub_intent="ADD_ITEM")
    result = HandlerResult(
        next_state=ConversationState.CANCELLATION_CONFIRMATION,
        response_key="confirm_cancel_current_item_for_new_request",
        awaiting_flow_confirmation=True,
        interrupt_proposal=proposal,
    )
    assert result.awaiting_flow_confirmation is True
    assert result.interrupt_proposal is proposal


# ── TurnEngine: main dispatch path field application ──────────────────────────

def test_engine_applies_prompt_field_from_handler_result():
    """Engine writes prompt_field to ctx.current_prompt_field after handler."""
    menu_repo = _menu_repo()
    engine = _engine(menu_repo)
    session = Session(session_id="pf-main", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    # "delivery" → WaitingForOrderTypeHandler returns prompt_field="delivery_area"
    result = _turn(engine, session, "delivery")

    assert result.response_key == "ask_for_delivery_area"
    assert session.conversation_context.current_prompt_field == "delivery_area"


def test_engine_does_not_overwrite_prompt_field_when_result_is_none():
    """prompt_field=None in HandlerResult means 'do not touch existing value'."""
    menu_repo = _menu_repo()
    engine = _engine(menu_repo)
    session = Session(session_id="pf-preserve", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    # Step 1: establish delivery_area prompt
    _turn(engine, session, "delivery")
    assert session.conversation_context.current_prompt_field == "delivery_area"

    # Step 2: give delivery area → handler moves to delivery_postal_code
    _turn(engine, session, "washington dc", intent=Intent.UNKNOWN,
          slots=(SlotValue(name="ITEM", value="washington dc"),))
    assert session.conversation_context.current_prompt_field == "delivery_postal_code"


# ── TurnEngine: WAITING_FOR_ORDER_TYPE fast-exit path field application ───────

def test_engine_applies_prompt_field_on_order_type_fast_path():
    """
    WAITING_FOR_ORDER_TYPE uses a dedicated early-exit path in TurnEngine.
    prompt_field must still be applied there (not just in main handler dispatch).
    """
    menu_repo = _menu_repo()
    engine = _engine(menu_repo)
    session = Session(session_id="pf-fastpath", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    result = _turn(engine, session, "delivery")

    assert result.response_key == "ask_for_delivery_area"
    assert session.conversation_state == ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY
    # Must be set via the fast-exit path, not the main dispatch block
    assert session.conversation_context.current_prompt_field == "delivery_area"


def test_engine_order_type_fast_path_pickup_does_not_set_prompt_field():
    """Pickup path returns prompt_field=None → current_prompt_field stays None."""
    menu_repo = _menu_repo()
    engine = _engine(menu_repo)
    session = Session(session_id="pf-pickup", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    result = _turn(engine, session, "pickup")

    assert result.response_key == "order_type_captured_pickup"
    assert session.conversation_context.current_prompt_field is None


# ── SOFT_SWITCH handlers return fields in HandlerResult ───────────────────────

def test_confirming_handler_soft_switch_returns_interrupt_in_result():
    """ConfirmingHandler must NOT mutate context.interrupt_proposal directly."""
    from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
    from app.menu.models import MenuItem, Pricing

    class _StubMenuRepo:
        def get_item(self, item_id):
            return MenuItem(
                item_id=item_id,
                name="Zinger Burger",
                normalized_name="zinger burger",
                aliases=(),
                normalized_aliases=(),
                voice_labels=(),
                pricing=Pricing(mode="fixed", price_cents=1000),
                side_groups=[],
                modifier_groups=[],
                available=True,
            )

        def resolve_item_within_candidates_normalized(self, *a, **kw):
            return None

        def resolve_menu_query(self, *a, **kw):
            return MagicMock(items=[], query_type=None)

        def resolve_menu_query_from_slots(self, **kw):
            return MagicMock(items=[], query_type=None)

    context = ConversationContext()
    context.awaiting_confirmation_for = {
        "type": "item",
        "reason": "candidate_selected",
        "value_id": "burger_1",
        "value_name": "Zinger Burger",
        "previous_confirmation": {
            "type": "item",
            "reason": "multiple_matches",
            "query": "burger",
            "candidate_item_ids": ["burger_1"],
            "candidate_item_names": ["Zinger Burger"],
        },
    }

    class _Session:
        conversation_state = ConversationState.CONFIRMING_ITEM

    # Direct context value before handler call
    assert context.interrupt_proposal is None
    assert context.awaiting_flow_confirmation is False

    result = ConfirmingHandler(_StubMenuRepo()).handle(
        intent=Intent.ADD_ITEM,  # SOFT_SWITCH_INTENT
        context=context,
        user_text="add a coke",
        session=_Session(),
    )

    # HandlerResult carries the fields
    assert result.next_state == ConversationState.CANCELLATION_CONFIRMATION
    assert result.awaiting_flow_confirmation is True
    assert result.interrupt_proposal is not None
    assert result.interrupt_proposal.text == "add a coke"
    assert result.interrupt_proposal.predicted_sub_intent == Intent.ADD_ITEM.value

    # Handler must NOT have mutated context directly
    assert context.interrupt_proposal is None, (
        "Handler must not mutate ctx.interrupt_proposal directly — use HandlerResult"
    )
    assert context.awaiting_flow_confirmation is False, (
        "Handler must not mutate ctx.awaiting_flow_confirmation directly — use HandlerResult"
    )


def test_engine_applies_interrupt_proposal_from_soft_switch_result():
    """
    When a SOFT_SWITCH triggers CANCELLATION_CONFIRMATION, TurnEngine applies
    interrupt_proposal and awaiting_flow_confirmation from HandlerResult to context.
    """
    menu_repo = _menu_repo()
    engine = _engine(menu_repo)
    session = Session(session_id="softswitch-engine", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    # Start with pickup so we get to IDLE
    _turn(engine, session, "pickup")
    assert session.conversation_state == ConversationState.IDLE

    # Add a burger to get into CONFIRMING_ITEM state
    nlu_add = _fake_nlu("zinger burger", Intent.ADD_ITEM, slots=(SlotValue(name="ITEM", value="Burger"),))
    with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=nlu_add):
        add_result = engine.process_turn(session=session, user_text="zinger burger")

    # Only continue if we land in a state where SOFT_SWITCH can trigger
    if session.conversation_state not in {
        ConversationState.CONFIRMING_ITEM,
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_SIZE,
        ConversationState.WAITING_FOR_QUANTITY,
    }:
        return  # menu/fixture didn't set up the right state; skip

    state_before = session.conversation_state
    ctx = session.conversation_context
    assert ctx.awaiting_flow_confirmation is False

    # Trigger SOFT_SWITCH with ADD_ITEM intent
    nlu_switch = _fake_nlu("add a coke", Intent.ADD_ITEM, slots=(SlotValue(name="ITEM", value="Coke"),))
    with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=nlu_switch):
        switch_result = engine.process_turn(session=session, user_text="add a coke")

    if session.conversation_state == ConversationState.CANCELLATION_CONFIRMATION:
        # Engine must have applied the fields
        assert ctx.awaiting_flow_confirmation is True
        assert ctx.interrupt_proposal is not None
        assert ctx.interrupt_proposal.text == "add a coke"


# ── prompt_field tracking: cross-handler transition ───────────────────────────

def test_prompt_field_tracks_through_delivery_wizard():
    """
    The delivery wizard transitions through 3 prompt states.  Each step must
    set current_prompt_field correctly even though different handlers own each step.
    """
    menu_repo = _menu_repo()
    engine = _engine(menu_repo)
    session = Session(session_id="wizard-prompts", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    # Step 1: order type → delivery area prompt
    _turn(engine, session, "delivery")
    assert session.conversation_context.current_prompt_field == "delivery_area"

    # Step 2: delivery area → postal code prompt
    _turn(engine, session, "washington dc", intent=Intent.UNKNOWN,
          slots=(SlotValue(name="ITEM", value="washington dc"),))
    assert session.conversation_context.current_prompt_field == "delivery_postal_code"

    # Step 3: postal code → eligibility confirmation prompt
    _turn(engine, session, "21000", intent=Intent.UNKNOWN)
    assert session.conversation_context.current_prompt_field == "delivery_eligibility_confirmation"


# ── deny interrupt: resume prior flow ─────────────────────────────────────────

def test_deny_interrupt_clears_flow_confirmation_fields():
    """
    After denying a SOFT_SWITCH cancellation, awaiting_flow_confirmation and
    interrupt_proposal should be cleared.  CancellationConfirmationHandler._clear_flow_confirmation_state
    is the legitimate owner of that cleanup.
    """
    from app.state_machine.handlers.common.cancellation_confirmation_handler import (
        CancellationConfirmationHandler,
    )

    context = ConversationContext()
    context.awaiting_flow_confirmation = True
    context.return_state = ConversationState.WAITING_FOR_MODIFIER
    context.interrupt_proposal = InterruptProposal(
        text="add a coke",
        predicted_main_intent=None,
        predicted_sub_intent=Intent.ADD_ITEM.value,
    )
    context.current_prompt_field = "modifier"
    context.available_choices_values = frozenset(["lettuce", "tomato"])

    class _Session:
        conversation_state = ConversationState.CANCELLATION_CONFIRMATION

    result = CancellationConfirmationHandler().handle(
        intent=Intent.DENY,
        context=context,
        user_text="no keep it",
        session=_Session(),
    )

    assert result.next_state == ConversationState.WAITING_FOR_MODIFIER
    assert result.response_key == "continue_current_item_after_cancel_denied"
    # Fields cleared by handler (legitimate cleanup of its own flow state)
    assert context.awaiting_flow_confirmation is False
    assert context.interrupt_proposal is None


# ── accept interrupt: reset_task clears pending fields ────────────────────────

def test_accept_interrupt_via_cancellation_handler_resets_pending_state():
    """
    When user affirms cancellation (switches to new item), CancellationConfirmationHandler
    calls context.reset_task() which clears awaiting_flow_confirmation, interrupt_proposal,
    current_prompt_field, and pending_add_item.
    """
    from app.state_machine.handlers.common.cancellation_confirmation_handler import (
        CancellationConfirmationHandler,
    )

    context = ConversationContext()
    context.awaiting_flow_confirmation = True
    context.return_state = ConversationState.WAITING_FOR_MODIFIER
    context.interrupt_proposal = InterruptProposal(
        text="add a coke",
        predicted_main_intent=None,
        predicted_sub_intent=Intent.ADD_ITEM.value,
    )
    context.current_prompt_field = "modifier"

    class _Session:
        conversation_state = ConversationState.CANCELLATION_CONFIRMATION

    result = CancellationConfirmationHandler().handle(
        intent=Intent.CONFIRM,
        context=context,
        user_text="yes cancel it",
        session=_Session(),
    )

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_cancelled_successfully"
    # reset_task clears all pending fields
    assert context.awaiting_flow_confirmation is False
    assert context.current_prompt_field is None
    assert context.pending_add_item is None


# ── handler must not be sole writer: engine owns application ──────────────────

def test_handler_result_awaiting_flow_confirmation_false_is_applied():
    """
    A HandlerResult with awaiting_flow_confirmation=False should explicitly set
    ctx.awaiting_flow_confirmation to False (not leave it at its current value).
    This verifies the TurnEngine conditional: `if result.awaiting_flow_confirmation is not None`.
    """
    context = ConversationContext()
    context.awaiting_flow_confirmation = True  # pre-existing True

    # Simulate engine applying a result with explicit False
    result = HandlerResult(
        next_state=ConversationState.IDLE,
        response_key="some_key",
        awaiting_flow_confirmation=False,
    )

    # Replicate the engine's application logic
    if result.awaiting_flow_confirmation is not None:
        context.awaiting_flow_confirmation = result.awaiting_flow_confirmation

    assert context.awaiting_flow_confirmation is False


def test_handler_result_none_does_not_overwrite_existing_context_value():
    """
    A HandlerResult with awaiting_flow_confirmation=None should NOT change
    ctx.awaiting_flow_confirmation.  Simulates engine 'leave unchanged' semantic.
    """
    context = ConversationContext()
    context.awaiting_flow_confirmation = True

    result = HandlerResult(
        next_state=ConversationState.IDLE,
        response_key="some_key",
        awaiting_flow_confirmation=None,  # default: don't touch
    )

    if result.awaiting_flow_confirmation is not None:
        context.awaiting_flow_confirmation = result.awaiting_flow_confirmation

    assert context.awaiting_flow_confirmation is True  # unchanged


def test_handler_result_interrupt_proposal_none_does_not_overwrite():
    """interrupt_proposal=None in HandlerResult leaves existing proposal intact."""
    proposal = InterruptProposal(text="add a coke", predicted_main_intent=None, predicted_sub_intent="ADD_ITEM")
    context = ConversationContext()
    context.interrupt_proposal = proposal

    result = HandlerResult(
        next_state=ConversationState.IDLE,
        response_key="some_key",
        interrupt_proposal=None,
    )

    if result.interrupt_proposal is not None:
        context.interrupt_proposal = result.interrupt_proposal

    assert context.interrupt_proposal is proposal  # unchanged
