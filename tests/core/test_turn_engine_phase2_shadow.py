# tests/core/test_turn_engine_phase2_shadow.py
"""Integration tests for Phase 2 GPT shadow-mode repair inside TurnEngine.

Verifies that:
- Phase 0 does not call GPT.
- Phase 2 calls GPT for eligible UNKNOWN turns.
- Phase 2 never applies the GPT repair (intent unchanged).
- Timeout is logged correctly and does not apply repair.
- final_intent_after_gpt equals local_intent_before_gpt in phase 2.
- The OpenAI API key is never logged in any TurnEvent field.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy external dependencies before any project imports
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


class _RedisClient:
    def __init__(self, *args, **kwargs):
        pass


redis_module.Redis = _RedisClient
sys.modules.setdefault("redis", redis_module)

intent_inference_module = types.ModuleType("app.ml.intent.inference_intent")
slot_inference_module = types.ModuleType("app.ml.slot.inference_slot")


class _IntentBundle:
    pass


class _SlotBundle:
    pass


intent_inference_module.IntentBundle = _IntentBundle
intent_inference_module.predict_intent = lambda *args, **kwargs: []
slot_inference_module.SlotBundle = _SlotBundle
slot_inference_module.predict_slots = lambda *args, **kwargs: []

sys.modules.setdefault("app.ml.intent.inference_intent", intent_inference_module)
sys.modules.setdefault("app.ml.slot.inference_slot", slot_inference_module)

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
from app.nlu.nlu_result import NLUResult, SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.nlu.semantic_repair.gpt_repair_result import GPT_NOT_CALLED
from app.nlu.semantic_repair.repair_service import GptRepairService
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        return SimpleNamespace(
            ok=False, sid=None, error_code="not_configured", error_message="not configured"
        )


class CapturingBackend:
    """Captures full TurnEvent objects so tests can assert GPT shadow fields."""

    enabled = True

    def __init__(self) -> None:
        self.events: list[TurnEvent] = []

    def record(self, event: TurnEvent) -> None:
        self.events.append(event)

    @property
    def last(self) -> TurnEvent | None:
        return self.events[-1] if self.events else None


class CapturingLogger:
    """Minimal NluCsvLogger stand-in used by CsvDiagnosticsBackend."""

    enabled = True

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log_turn(self, **kwargs) -> None:
        self.rows.append(kwargs)


def _build_menu_repo() -> MenuRepository:
    data_root = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "data"
        / "restaurants"
        / "steves_grill"
    )
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def _build_engine(
    menu_repo: MenuRepository,
    capturing_backend: CapturingBackend,
    *,
    gpt_phase: int = 0,
) -> TurnEngine:
    """Build a TurnEngine with the given GPT phase and a CapturingBackend."""
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
    # Inject the capturing backend so we see full TurnEvent objects.
    engine.diagnostics._backends.append(capturing_backend)

    # Override the repair service with the requested phase.
    cfg = SemanticRepairConfig(
        phase=gpt_phase,
        model="gpt-4o-mini",
        timeout_seconds=3.0,
    )
    engine.gpt_repair = GptRepairService(config=cfg)
    return engine


def _make_nlu(text: str, intent: Intent, slots: tuple = ()) -> SimpleNamespace:
    return SimpleNamespace(
        effective_intent=intent,
        intent_confidence=0.99 if intent != Intent.UNKNOWN else 0.1,
        normalized_text=normalize_text(text),
        raw_text=text,
        slots=tuple(slots),
        model_main_intent=intent.value,
        model_sub_intent=intent.value,
        slot_model_ran=bool(slots),
        intent_model_ms=None,
        slot_model_ms=None,
    )


def _turn(engine: TurnEngine, session: Session, text: str, *, intent: Intent = Intent.UNKNOWN):
    fake_nlu = _make_nlu(text, intent)
    with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
        return engine.process_turn(session=session, user_text=text)


def _idle_session() -> Session:
    session = Session(session_id="shadow-test", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.IDLE
    # Pre-set order type so the order-type gate doesn't redirect to
    # WAITING_FOR_ORDER_TYPE, which has its own NLU path and returns before
    # the main NLU + GPT shadow call.
    session.conversation_context.order_type = "pickup"
    return session


def _make_openai_response(content: str) -> SimpleNamespace:
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _valid_repair_response(intent: str = "add_item") -> SimpleNamespace:
    payload = json.dumps({
        "decision": "repair",
        "repaired_intent": intent,
        "repaired_control_intent": None,
        "slot_corrections": {},
        "confidence": 0.92,
        "reason": "user is clearly ordering",
        "requires_handler_validation": True,
    })
    return _make_openai_response(payload)


# ---------------------------------------------------------------------------
# Phase 0 — GPT must not be called
# ---------------------------------------------------------------------------


def test_phase0_gpt_not_called() -> None:
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=0)

    with patch.object(engine.gpt_repair, "_call_gpt") as mock_call:
        _turn(engine, _idle_session(), "I want something to eat")
        mock_call.assert_not_called()


def test_phase0_event_has_gpt_called_false_or_none() -> None:
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=0)

    _turn(engine, _idle_session(), "I want something to eat")

    event = cb.last
    assert event is not None
    # In phase 0 the GPT was not called; gpt_called should be False or None.
    assert event.gpt_called in (False, None)


# ---------------------------------------------------------------------------
# Phase 2 — GPT called for eligible UNKNOWN intent
# ---------------------------------------------------------------------------


def test_phase2_calls_gpt_for_unknown_intent() -> None:
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=2)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _valid_repair_response("add_item")
    engine.gpt_repair._client = mock_client

    _turn(engine, _idle_session(), "I want something good to eat")

    mock_client.chat.completions.create.assert_called_once()


def test_phase2_does_not_call_gpt_for_known_intent() -> None:
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=2)

    with patch.object(engine.gpt_repair, "_call_gpt") as mock_call:
        _turn(engine, _idle_session(), "add item", intent=Intent.ADD_ITEM)
        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 2 — repair is never applied
# ---------------------------------------------------------------------------


def test_phase2_does_not_apply_repair() -> None:
    """Even when GPT suggests a repair, intent_result and session state are unchanged."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=2)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _valid_repair_response("checkout")
    engine.gpt_repair._client = mock_client

    session = _idle_session()
    # The NLU returns UNKNOWN; GPT suggests "checkout" but it must not be applied.
    _turn(engine, session, "I want something good to eat")

    event = cb.last
    assert event is not None
    # The logged gpt_applied must be False.
    assert event.gpt_applied is False
    # The GPT repair logged the repaired_intent as "checkout"...
    assert event.gpt_selected_intent == "checkout"
    # ...but the effective pred_intent that actually routed the turn stayed as "unknown".
    assert event.pred_intent == Intent.UNKNOWN.value


def test_phase2_gpt_applied_always_false_even_on_repair_decision() -> None:
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=2)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _valid_repair_response("add_item")
    engine.gpt_repair._client = mock_client

    _turn(engine, _idle_session(), "give me a burger")

    event = cb.last
    assert event is not None
    assert event.gpt_applied is False


# ---------------------------------------------------------------------------
# Timeout — gpt_timeout=True, no repair applied
# ---------------------------------------------------------------------------


def test_timeout_logs_gpt_timeout_true() -> None:
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=2)

    class APITimeoutError(Exception):
        pass

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError("timed out")
    engine.gpt_repair._client = mock_client

    _turn(engine, _idle_session(), "I want a sandwich")

    event = cb.last
    assert event is not None
    assert event.gpt_timeout is True
    assert event.gpt_decision == "no_repair"
    assert event.gpt_applied is False


def test_timeout_does_not_apply_repair() -> None:
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=2)

    class RequestTimeoutError(Exception):
        pass

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RequestTimeoutError()
    engine.gpt_repair._client = mock_client

    session = _idle_session()
    _turn(engine, session, "I want a sandwich")

    event = cb.last
    assert event is not None
    assert event.gpt_applied is False
    assert event.pred_intent == Intent.UNKNOWN.value


# ---------------------------------------------------------------------------
# Invariant: final_intent_after_gpt == local_intent_before_gpt
# ---------------------------------------------------------------------------


def test_final_intent_after_gpt_equals_local_intent_before_gpt_phase2() -> None:
    """Even when GPT returns a repair, the final intent logged must match the local intent."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=2)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _valid_repair_response("checkout")
    engine.gpt_repair._client = mock_client

    _turn(engine, _idle_session(), "I would like something please")

    event = cb.last
    assert event is not None
    assert event.local_intent_before_gpt is not None
    assert event.final_intent_after_gpt is not None
    assert event.final_intent_after_gpt == event.local_intent_before_gpt


def test_final_intent_invariant_on_timeout() -> None:
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=2)

    class APITimeoutError(Exception):
        pass

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError()
    engine.gpt_repair._client = mock_client

    _turn(engine, _idle_session(), "something delicious please")

    event = cb.last
    assert event is not None
    assert event.final_intent_after_gpt == event.local_intent_before_gpt


def test_final_intent_invariant_phase0() -> None:
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, gpt_phase=0)

    _turn(engine, _idle_session(), "something random")

    event = cb.last
    assert event is not None
    # In phase 0 we still log the analysis, so these fields should be populated.
    if event.local_intent_before_gpt is not None:
        assert event.final_intent_after_gpt == event.local_intent_before_gpt


# ---------------------------------------------------------------------------
# API key never logged
# ---------------------------------------------------------------------------

_FAKE_KEY = "sk-test-secretkeyvalue12345678901234567890"


def test_api_key_never_in_turn_event() -> None:
    """Run a phase-2 turn and verify the API key does not appear in any TurnEvent field."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()

    with patch.dict("os.environ", {"OPENAI_API_KEY": _FAKE_KEY}):
        engine = _build_engine(menu_repo, cb, gpt_phase=2)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _valid_repair_response("add_item")
        engine.gpt_repair._client = mock_client

        _turn(engine, _idle_session(), "I want a meal")

    event = cb.last
    assert event is not None

    import dataclasses
    for field in dataclasses.fields(event):
        val = getattr(event, field.name)
        if val is None:
            continue
        assert _FAKE_KEY not in str(val), (
            f"API key found in TurnEvent.{field.name}={val!r}"
        )


def test_api_key_not_in_gpt_reason_or_parse_error_fields() -> None:
    """Even if GPT call fails with the key in the error message, it must not be logged."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()

    with patch.dict("os.environ", {"OPENAI_API_KEY": _FAKE_KEY}):
        engine = _build_engine(menu_repo, cb, gpt_phase=2)
        # Simulate error that accidentally mentions the API key in its message
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            f"Authentication failed for {_FAKE_KEY}"
        )
        engine.gpt_repair._client = mock_client

        _turn(engine, _idle_session(), "I want a meal")

    event = cb.last
    assert event is not None
    # The parse_error or reason field should not contain the key.
    for field_name in ("gpt_reason", "gpt_parse_error"):
        val = getattr(event, field_name, None)
        if val:
            assert _FAKE_KEY not in str(val), f"API key in {field_name}: {val!r}"
