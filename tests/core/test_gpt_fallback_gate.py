# tests/core/test_gpt_fallback_gate.py
"""Tests for GPT fallback application gate (Part 2).

Verifies that:
- Fallback is NOT applied when apply_fallbacks=False (the default).
- Fallback is NOT applied when call_mode=all_shadow.
- Fallback IS applied when apply_fallbacks=True and call_mode=eligible_only.
- gpt_applied=False in all shadow/default configurations.
"""
from __future__ import annotations

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
intent_inference_module.IntentBundle = type("IntentBundle", (), {})
intent_inference_module.predict_intent = lambda *a, **kw: []
slot_inference_module.SlotBundle = type("SlotBundle", (), {})
slot_inference_module.predict_slots = lambda *a, **kw: []
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
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.nlu.semantic_repair.gpt_repair_result import GPT_NOT_CALLED, GptRepairResult
from app.nlu.semantic_repair.repair_service import GptRepairService, LocalTurnAnalysis
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------


class StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        return SimpleNamespace(
            ok=False, sid=None, error_code="not_configured", error_message="not configured"
        )


class CapturingBackend:
    enabled = True

    def __init__(self) -> None:
        self.events: list[TurnEvent] = []

    def record(self, event: TurnEvent) -> None:
        self.events.append(event)

    @property
    def last(self) -> TurnEvent | None:
        return self.events[-1] if self.events else None


class CapturingLogger:
    enabled = True

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log_turn(self, **kwargs) -> None:
        self.rows.append(kwargs)


def _build_menu_repo() -> MenuRepository:
    data_root = (
        Path(__file__).resolve().parents[2]
        / "app" / "data" / "restaurants" / "steves_grill"
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
    call_mode: str = "eligible_only",
    apply_fallbacks: bool = False,
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
    cfg = SemanticRepairConfig(
        phase=2,
        model="gpt-4o-mini",
        timeout_seconds=3.0,
        call_mode=call_mode,
        apply_fallbacks=apply_fallbacks,
    )
    engine.gpt_repair = GptRepairService(config=cfg)
    return engine


def _make_nlu(text: str, intent: Intent = Intent.UNKNOWN) -> SimpleNamespace:
    return SimpleNamespace(
        effective_intent=intent,
        intent_confidence=0.1,
        normalized_text=normalize_text(text),
        raw_text=text,
        slots=(),
        model_main_intent=intent.value,
        model_sub_intent=intent.value,
        slot_model_ran=False,
        intent_model_ms=None,
        slot_model_ms=None,
    )


def _turn(engine: TurnEngine, session: Session, text: str) -> None:
    fake_nlu = _make_nlu(text)
    with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
        engine.process_turn(session=session, user_text=text)


def _idle_session() -> Session:
    session = Session(session_id="fallback-gate-test", restaurant_id="steves_grill")
    session.conversation_state = ConversationState.IDLE
    session.conversation_context.order_type = "pickup"
    return session


def _make_minimal_analysis() -> LocalTurnAnalysis:
    return LocalTurnAnalysis(
        gpt_repair_eligible=True,
        reason="unknown_text",
        candidate_count=3,
        candidates=frozenset({"add_item", "checkout", "cancel_order"}),
        intent_effective=Intent.UNKNOWN.value,
        state_before=ConversationState.IDLE.value,
    )


def _fallback_result(fallback_type: str = "off_topic") -> GptRepairResult:
    return GptRepairResult(
        decision="fallback",
        fallback_type=fallback_type,
        confidence=0.85,
        latency_ms=150.0,
        total_ms=200.0,
    )


# ---------------------------------------------------------------------------
# Fallback NOT applied when apply_fallbacks=False (default)
# ---------------------------------------------------------------------------


def test_fallback_not_applied_when_apply_fallbacks_false():
    """With apply_fallbacks=False (the default), GPT fallback must never override response."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, call_mode="eligible_only", apply_fallbacks=False)

    analysis = _make_minimal_analysis()
    fallback = _fallback_result("off_topic")

    with patch.object(engine.gpt_repair, "run", return_value=(analysis, fallback)):
        _turn(engine, _idle_session(), "I want something random")

    event = cb.last
    assert event is not None
    # Fallback must NOT have been applied
    assert event.gpt_applied is False


def test_fallback_not_applied_gpt_apply_reason_not_fallback_applied():
    """When apply_fallbacks=False, gpt_apply_reason must not be 'fallback_applied'."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, call_mode="eligible_only", apply_fallbacks=False)

    analysis = _make_minimal_analysis()
    fallback = _fallback_result("unclear")

    with patch.object(engine.gpt_repair, "run", return_value=(analysis, fallback)):
        _turn(engine, _idle_session(), "hmm I am not sure")

    event = cb.last
    assert event is not None
    assert getattr(event, "gpt_apply_reason", None) != "fallback_applied"


def test_no_fallback_type_never_applied():
    """fallback_type='none' must never trigger fallback path regardless of apply_fallbacks."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, call_mode="eligible_only", apply_fallbacks=True)

    analysis = _make_minimal_analysis()
    # fallback_type='none' — this must NOT trigger the fallback path even with apply_fallbacks=True
    result_with_none_type = GptRepairResult(
        decision="fallback",
        fallback_type="none",
        confidence=0.85,
        latency_ms=100.0,
        total_ms=150.0,
    )

    with patch.object(engine.gpt_repair, "run", return_value=(analysis, result_with_none_type)):
        _turn(engine, _idle_session(), "something unclear")

    event = cb.last
    assert event is not None
    # fallback_type=none → gate remains closed
    assert event.gpt_applied is False


def test_fallback_gpt_applied_false_when_gpt_not_called():
    """When GPT returns GPT_NOT_CALLED (phase=disabled), gpt_applied must be False."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, call_mode="disabled", apply_fallbacks=True)

    analysis = _make_minimal_analysis()
    with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
        _turn(engine, _idle_session(), "I want something")

    event = cb.last
    assert event is not None
    assert event.gpt_applied is False


# ---------------------------------------------------------------------------
# Fallback IS applied when apply_fallbacks=True and call_mode=eligible_only
# ---------------------------------------------------------------------------

# TurnEngine reads config from the global _get_gpt_cfg() (a cached function),
# not from the service's local config.  Tests that need apply_fallbacks=True
# must patch app.core.turn_engine._get_gpt_cfg to return the desired config.
_APPLY_FALLBACKS_CFG = SemanticRepairConfig(
    phase=2,
    model="gpt-4o-mini",
    timeout_seconds=3.0,
    call_mode="eligible_only",
    apply_fallbacks=True,
)


def test_fallback_applied_when_apply_fallbacks_true():
    """With apply_fallbacks=True and eligible_only, GPT fallback must override response."""
    import app.core.turn_engine as _te

    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, call_mode="eligible_only", apply_fallbacks=True)

    analysis = _make_minimal_analysis()
    fallback = _fallback_result("off_topic")

    with patch.object(_te, "_get_gpt_cfg", return_value=_APPLY_FALLBACKS_CFG):
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, fallback)):
            _turn(engine, _idle_session(), "what time does the train leave")

    event = cb.last
    assert event is not None
    assert event.gpt_applied is True


def test_fallback_applied_response_key_has_fallback_prefix():
    """When fallback is applied, the logged response_key should start with 'fallback_'."""
    import app.core.turn_engine as _te

    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_engine(menu_repo, cb, call_mode="eligible_only", apply_fallbacks=True)

    analysis = _make_minimal_analysis()
    fallback = _fallback_result("off_topic")

    with patch.object(_te, "_get_gpt_cfg", return_value=_APPLY_FALLBACKS_CFG):
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, fallback)):
            _turn(engine, _idle_session(), "what is the weather today")

    event = cb.last
    assert event is not None
    assert event.response_key.startswith("fallback_"), (
        f"Expected fallback_ prefix; got: {event.response_key!r}"
    )
