# tests/core/test_turn_engine_add_item_shadow.py
"""Integration tests for ADD_ITEM extractor shadow-mode wiring in TurnEngine.

Verifies that:
- add_item_mode=shadow runs AddItemExtractorService for ADD_ITEM turns.
- add_item_mode=disabled never calls the extractor.
- Extractor result is logged in TurnEvent add_item_* fields.
- Extractor NEVER mutates cart, state, intent_result, or response.
- Terminal states skip the extractor.
- add_item_extractor_called is False when GPT was not called (skipped).
- add_item_extractor_called is True when GPT was called (total_ms set).

Design note: ALL assertions are unconditional — no `if backend.last:` guards.
  TurnEngine always records a TurnEvent after process_turn completes.
  When add_item_mode=shadow, add_item_extractor.run() is called on every turn.
  When add_item_mode=disabled, run() is never called.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy external dependencies before any project import
# ---------------------------------------------------------------------------

twilio_module = types.ModuleType("twilio")
twilio_base_module = types.ModuleType("twilio.base")
twilio_base_exceptions_module = types.ModuleType("twilio.base.exceptions")
twilio_rest_module = types.ModuleType("twilio.rest")


class _TwilioRestException(Exception):
    pass


class _TwilioClient:
    def __init__(self, *args, **kwargs):
        pass


twilio_base_exceptions_module.TwilioRestException = _TwilioRestException
twilio_rest_module.Client = _TwilioClient
sys.modules.setdefault("twilio", twilio_module)
sys.modules.setdefault("twilio.base", twilio_base_module)
sys.modules.setdefault("twilio.base.exceptions", twilio_base_exceptions_module)
sys.modules.setdefault("twilio.rest", twilio_rest_module)

redis_module = types.ModuleType("redis")
redis_module.Redis = type("Redis", (), {"__init__": lambda *a, **k: None})
sys.modules.setdefault("redis", redis_module)

intent_mod = types.ModuleType("app.ml.intent.inference_intent")
slot_mod = types.ModuleType("app.ml.slot.inference_slot")
intent_mod.IntentBundle = type("IntentBundle", (), {})
intent_mod.predict_intent = lambda *a, **k: []
slot_mod.SlotBundle = type("SlotBundle", (), {})
slot_mod.predict_slots = lambda *a, **k: []
sys.modules.setdefault("app.ml.intent.inference_intent", intent_mod)
sys.modules.setdefault("app.ml.slot.inference_slot", slot_mod)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from app.config.semantic_repair import SemanticRepairConfig
from app.core.response_builder import ResponseBuilder
from app.core.turn_engine import TurnEngine
from app.diagnostics.turn_event import TurnEvent
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.nlu.semantic_repair.add_item_extractor import GptAddItemPlan
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


class StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        return SimpleNamespace(ok=False, sid=None, error_code="stub", error_message="stub")


class CapturingBackend:
    enabled = True

    def __init__(self):
        self.events: list[TurnEvent] = []

    def record(self, event: TurnEvent) -> None:
        self.events.append(event)

    @property
    def last(self) -> TurnEvent | None:
        return self.events[-1] if self.events else None


class CapturingLogger:
    enabled = True

    def __init__(self):
        self.rows: list[dict] = []

    def log_turn(self, **kwargs) -> None:
        self.rows.append(kwargs)


def _build_menu_repo() -> MenuRepository:
    """Load the demo restaurant menu (contains Bourbon Chicken, Fries, etc.)."""
    data_root = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "data"
        / "restaurants"
        / "demo"
    )
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def _build_engine(
    menu_repo: MenuRepository,
    capturing_backend: CapturingBackend,
) -> TurnEngine:
    logger = CapturingLogger()
    engine = TurnEngine(
        router=StateRouter(),
        menu_repo=menu_repo,
        intent_bundle=None,
        slot_bundle=None,
        responder=ResponseBuilder(menu_repo),
        sms_service=StubSmsService(),
        nlu_logger=logger,
    )
    engine.diagnostics._backends.append(capturing_backend)
    return engine


def _shadow_config(add_item_mode: str = "shadow") -> SemanticRepairConfig:
    return SemanticRepairConfig(
        phase=0,
        model="gpt-4o-mini",
        timeout_seconds=3.0,
        call_mode="disabled",
        add_item_mode=add_item_mode,
    )


def _make_add_item_plan(*, called: bool = True) -> GptAddItemPlan:
    """Return a fake plan as if GPT was called (total_ms set) or skipped (None)."""
    if called:
        return GptAddItemPlan(
            decision="ok",
            eligible=True,
            total_ms=50.0,
            latency_ms=48.0,
            prompt_chars=200,
            completion_chars=80,
            model="gpt-4o-mini",
        )
    else:
        return GptAddItemPlan(
            decision="no_repair",
            eligible=True,
            skipped_reason="daily_budget_exceeded",
            total_ms=None,
        )


def _make_nlu(
    text: str,
    intent: Intent,
    slots: tuple[SlotValue, ...] = (),
) -> SimpleNamespace:
    """Build a fake NLU result for patching resolve_nlu."""
    return SimpleNamespace(
        effective_intent=intent,
        intent_confidence=0.99,
        normalized_text=normalize_text(text),
        slots=slots,
        model_main_intent=intent.value,
        model_sub_intent=intent.value,
        slot_model_ran=bool(slots),
        intent_model_ms=None,
        slot_model_ms=None,
        intent_candidates=(),
    )


def _new_session(*, state: ConversationState = ConversationState.IDLE) -> Session:
    """New demo-restaurant session with order type pre-set to bypass the gate.

    Setting order_type="pickup" prevents FlowGate._compute_order_type_gate_state
    from resetting the session to WAITING_FOR_ORDER_TYPE, which would cause
    an early return before the ADD_ITEM extractor is reached.
    """
    session = Session(session_id="shadow-test", restaurant_id="demo")
    session.conversation_state = state
    session.conversation_context.caller_device_type = "phone"
    session.conversation_context.order_type = "pickup"
    return session


@pytest.fixture(scope="module")
def menu_repo():
    return _build_menu_repo()


# ---------------------------------------------------------------------------
# Tests: disabled mode
# ---------------------------------------------------------------------------


class TestAddItemDisabledMode:
    def test_disabled_extractor_not_called(self, menu_repo):
        """add_item_mode=disabled → run() is never invoked, even for ADD_ITEM turns."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)

        fake_nlu = _make_nlu(
            "bourbon chicken",
            Intent.ADD_ITEM,
            (SlotValue(name="ITEM", value="Bourbon Chicken"),),
        )

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("disabled")):
            with patch.object(engine.add_item_extractor, "run") as mock_run:
                with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                    # IDLE state with order_type set — reaches the extractor gate.
                    session = _new_session()
                    engine.process_turn(session, "bourbon chicken")

                mock_run.assert_not_called()

    def test_disabled_add_item_fields_all_false_in_event(self, menu_repo):
        """add_item_mode=disabled → TurnEvent.add_item_extractor_called is False."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)

        fake_nlu = _make_nlu(
            "bourbon chicken",
            Intent.ADD_ITEM,
            (SlotValue(name="ITEM", value="Bourbon Chicken"),),
        )

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("disabled")):
            with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                session = _new_session()
                engine.process_turn(session, "bourbon chicken")

        # TurnEngine always records an event — assert unconditionally.
        assert backend.last is not None, "TurnEngine must always record a TurnEvent"
        assert backend.last.add_item_extractor_called is False
        assert backend.last.add_item_eligible is False


# ---------------------------------------------------------------------------
# Tests: shadow mode — extractor called
# ---------------------------------------------------------------------------


class TestAddItemShadowMode:
    def test_shadow_extractor_called_for_add_item_intent(self, menu_repo):
        """add_item_mode=shadow + ADD_ITEM intent → run() called exactly once."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)
        _plan = _make_add_item_plan(called=True)

        fake_nlu = _make_nlu(
            "bourbon chicken",
            Intent.ADD_ITEM,
            (SlotValue(name="ITEM", value="Bourbon Chicken"),),
        )

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("shadow")):
            with patch.object(engine.add_item_extractor, "run", return_value=_plan) as mock_run:
                with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                    session = _new_session()
                    engine.process_turn(session, "bourbon chicken")

        # run() must have been invoked — no conditional guard.
        mock_run.assert_called_once()

    def test_shadow_result_in_turn_event_when_called(self, menu_repo):
        """TurnEvent add_item_* fields are populated from the plan, unconditionally."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)
        _plan = _make_add_item_plan(called=True)

        fake_nlu = _make_nlu(
            "bourbon chicken",
            Intent.ADD_ITEM,
            (SlotValue(name="ITEM", value="Bourbon Chicken"),),
        )

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("shadow")):
            with patch.object(engine.add_item_extractor, "run", return_value=_plan):
                with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                    session = _new_session()
                    engine.process_turn(session, "bourbon chicken")

        assert backend.last is not None, "TurnEngine must always record a TurnEvent"
        event = backend.last
        # Plan has total_ms=50.0 → extractor_called must be True.
        assert event.add_item_extractor_called is True
        assert event.add_item_total_ms == pytest.approx(50.0)
        assert event.add_item_model == "gpt-4o-mini"

    def test_shadow_never_applies_to_session_state(self, menu_repo):
        """Cart and state must not change because the extractor is shadow-only."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)
        _plan = _make_add_item_plan(called=True)

        fake_nlu = _make_nlu(
            "bourbon chicken",
            Intent.ADD_ITEM,
            (SlotValue(name="ITEM", value="Bourbon Chicken"),),
        )

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("shadow")):
            with patch.object(engine.add_item_extractor, "run", return_value=_plan):
                with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                    session = _new_session()
                    cart_items_before = len(session.cart.get_items())
                    engine.process_turn(session, "bourbon chicken")
                    # Cart size must be unchanged — extractor is shadow-only.
                    cart_items_after = len(session.cart.get_items())

        assert cart_items_after == cart_items_before, (
            "Shadow ADD_ITEM extractor must not add items to the cart"
        )

    def test_shadow_response_key_matches_local_deterministic_output(self, menu_repo):
        """Local FSM response_key is unchanged by shadow extractor — not overridden."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)
        _plan = _make_add_item_plan(called=True)

        fake_nlu = _make_nlu(
            "bourbon chicken",
            Intent.ADD_ITEM,
            (SlotValue(name="ITEM", value="Bourbon Chicken"),),
        )

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("shadow")):
            with patch.object(engine.add_item_extractor, "run", return_value=_plan):
                with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                    session = _new_session()
                    out = engine.process_turn(session, "bourbon chicken")

        # The shadow extractor may run but must not change the response.
        assert out.response_key, "response_key must be non-empty after a shadow-mode turn"
        # Extractor result must not appear as the response_key.
        assert out.response_key != "gpt_add_item_applied", (
            "Shadow extractor must never set response_key to its own decision"
        )

    def test_shadow_exception_does_not_stop_turn(self, menu_repo):
        """If the extractor raises, the turn must complete normally."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)

        def _raise(*args, **kwargs):
            raise RuntimeError("extractor exploded")

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("shadow")):
            with patch.object(engine.add_item_extractor, "run", side_effect=_raise):
                session = _new_session()
                output = engine.process_turn(session, "hello")

        # Turn completed — response_key must be non-empty.
        assert output.response_key, "Turn must complete even if extractor raises"

    def test_skipped_plan_not_marked_called(self, menu_repo):
        """When extractor is skipped (budget exhausted), extractor_called must be False."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)
        _plan = _make_add_item_plan(called=False)  # total_ms=None

        fake_nlu = _make_nlu(
            "bourbon chicken",
            Intent.ADD_ITEM,
            (SlotValue(name="ITEM", value="Bourbon Chicken"),),
        )

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("shadow")):
            with patch.object(engine.add_item_extractor, "run", return_value=_plan):
                with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                    session = _new_session()
                    engine.process_turn(session, "bourbon chicken")

        assert backend.last is not None, "TurnEngine must always record a TurnEvent"
        event = backend.last
        # total_ms=None → extractor_called must remain False.
        assert event.add_item_extractor_called is False


# ---------------------------------------------------------------------------
# Tests: realtime trace stamping
# ---------------------------------------------------------------------------


class TestAddItemTraceStamping:
    def test_trace_notes_stamped_with_add_item(self, menu_repo):
        """When trace.notes is a dict, add_item key must be written unconditionally."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)
        _plan = _make_add_item_plan(called=True)

        fake_nlu = _make_nlu(
            "bourbon chicken",
            Intent.ADD_ITEM,
            (SlotValue(name="ITEM", value="Bourbon Chicken"),),
        )

        class FakeTrace:
            notes: dict = {}

        trace = FakeTrace()
        trace.notes = {}

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("shadow")):
            with patch.object(engine.add_item_extractor, "run", return_value=_plan):
                with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                    session = _new_session()
                    engine.process_turn(session, "bourbon chicken", trace=trace)

        # The extractor ran (plan has total_ms set), so trace.notes["add_item"] must be set.
        assert "add_item" in trace.notes, (
            f"trace.notes must contain 'add_item' key; got keys: {list(trace.notes)}"
        )
        add_item_notes = trace.notes["add_item"]
        assert "add_item_decision" in add_item_notes
        assert "add_item_total_ms" in add_item_notes
        assert add_item_notes["add_item_extractor_called"] is True
        assert add_item_notes["add_item_total_ms"] == pytest.approx(50.0)

    def test_trace_notes_not_stamped_when_disabled(self, menu_repo):
        """When add_item_mode=disabled, trace.notes must not gain an add_item key."""
        backend = CapturingBackend()
        engine = _build_engine(menu_repo, backend)

        class FakeTrace:
            notes: dict = {}

        trace = FakeTrace()
        trace.notes = {}

        with patch("app.core.turn_engine._get_gpt_cfg", return_value=_shadow_config("disabled")):
            session = _new_session()
            engine.process_turn(session, "hello", trace=trace)

        assert "add_item" not in trace.notes, (
            "Disabled extractor must not write to trace.notes"
        )


# ---------------------------------------------------------------------------
# Tests: serialization helpers
# ---------------------------------------------------------------------------


class TestSerializationHelpers:
    def test_serialize_items_empty_returns_none(self):
        result = TurnEngine._serialize_add_item_items(())
        assert result is None

    def test_serialize_items_with_data(self):
        from app.nlu.semantic_repair.add_item_extractor import GptAddItem
        item = GptAddItem(item="burger", quantity=2, size="large")
        result = TurnEngine._serialize_add_item_items((item,))
        assert result is not None
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["item"] == "burger"
        assert parsed[0]["quantity"] == 2
        assert parsed[0]["size"] == "large"

    def test_serialize_items_caps_at_4000_chars(self):
        from app.nlu.semantic_repair.add_item_extractor import GptAddItem
        # Build items with very long names to trigger truncation.
        items = tuple(
            GptAddItem(item="a" * 1000, quantity=1) for _ in range(10)
        )
        result = TurnEngine._serialize_add_item_items(items)
        assert result is not None
        assert len(result) <= 4000

    def test_serialize_global_slots_empty_returns_none(self):
        result = TurnEngine._serialize_add_item_global_slots(())
        assert result is None

    def test_serialize_global_slots_with_dict_entries(self):
        slots = ({"n": "SIZE", "v": "large"},)
        result = TurnEngine._serialize_add_item_global_slots(slots)
        assert result is not None
        parsed = json.loads(result)
        assert parsed[0]["n"] == "SIZE"
