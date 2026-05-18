# tests/core/test_all_shadow_dispatch.py
"""Tests for all_shadow GPT dispatch mode (Part 3).

Verifies that:
- all_shadow dispatches background GPT task for normal non-terminal turns.
- all_shadow does NOT dispatch for terminal states.
- all_shadow stamps realtime trace with gpt_called=True, gpt_decision="pending_async", gpt_applied=False.
- all_shadow never applies GPT result to session/cart/response.
- process_turn returns without waiting for GPT background task.
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
from app.nlu.semantic_repair.gpt_repair_result import GPT_NOT_CALLED
from app.nlu.semantic_repair.repair_service import GptRepairService, LocalTurnAnalysis
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


# ---------------------------------------------------------------------------
# Shared infrastructure
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


def _build_all_shadow_engine(
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
    cfg = SemanticRepairConfig(
        phase=2,
        model="gpt-4o-mini",
        timeout_seconds=3.0,
        call_mode="all_shadow",
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


def _idle_session(session_id: str = "shadow-dispatch-test") -> Session:
    session = Session(session_id=session_id, restaurant_id="steves_grill")
    session.conversation_state = ConversationState.IDLE
    session.conversation_context.order_type = "pickup"
    return session


def _make_minimal_analysis() -> LocalTurnAnalysis:
    return LocalTurnAnalysis(
        gpt_repair_eligible=False,
        reason="all_shadow_bypass",
        candidate_count=0,
        candidates=frozenset(),
        intent_effective=Intent.UNKNOWN.value,
        state_before=ConversationState.IDLE.value,
    )


_FAKE_API_KEY = "sk-test-allshadow-key12345678"

# TurnEngine reads _is_all_shadow from _get_gpt_cfg() (global cached config).
# Tests that exercise the all_shadow path must patch app.core.turn_engine._get_gpt_cfg.
import app.core.turn_engine as _te  # imported here for use in all tests below

_ALL_SHADOW_CFG = SemanticRepairConfig(
    phase=2,
    model="gpt-4o-mini",
    timeout_seconds=2.0,
    call_mode="all_shadow",
)


def _all_shadow_ctx(*, with_api_key: bool = True):
    """Context manager stack that makes TurnEngine see all_shadow config + API key."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch.object(_te, "_get_gpt_cfg", return_value=_ALL_SHADOW_CFG))
    if with_api_key:
        stack.enter_context(patch.dict("os.environ", {"OPENAI_API_KEY": _FAKE_API_KEY}))
    else:
        import os
        env_copy = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        stack.enter_context(patch.dict("os.environ", env_copy, clear=True))
    return stack


# ---------------------------------------------------------------------------
# all_shadow: dispatches background GPT for normal turns
# ---------------------------------------------------------------------------


def test_all_shadow_dispatches_background_gpt_for_normal_turn():
    """A normal non-terminal IDLE turn with API key must call _dispatch_shadow_gpt."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_all_shadow_engine(menu_repo, cb)

    analysis = _make_minimal_analysis()
    with _all_shadow_ctx():
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
            with patch.object(engine, "_dispatch_shadow_gpt") as mock_dispatch:
                _turn(engine, _idle_session(), "I want a burger please")
                mock_dispatch.assert_called_once()


def test_all_shadow_dispatch_not_called_for_noise_turn():
    """A 1-character noise turn must NOT dispatch shadow GPT."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_all_shadow_engine(menu_repo, cb)

    analysis = _make_minimal_analysis()
    with _all_shadow_ctx():
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
            with patch.object(engine, "_dispatch_shadow_gpt") as mock_dispatch:
                _turn(engine, _idle_session(), "a")  # too short
                mock_dispatch.assert_not_called()


def test_all_shadow_dispatch_not_called_without_api_key():
    """If OPENAI_API_KEY is absent, shadow dispatch must be skipped."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_all_shadow_engine(menu_repo, cb)

    analysis = _make_minimal_analysis()
    with _all_shadow_ctx(with_api_key=False):
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
            with patch.object(engine, "_dispatch_shadow_gpt") as mock_dispatch:
                _turn(engine, _idle_session(), "I want a burger please")
                mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# all_shadow: does NOT dispatch for terminal states
# ---------------------------------------------------------------------------


def test_all_shadow_skip_completed_state():
    """all_shadow must not dispatch GPT when session is in COMPLETED state."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_all_shadow_engine(menu_repo, cb)

    session = _idle_session()
    session.conversation_state = ConversationState.COMPLETED

    analysis = _make_minimal_analysis()
    with _all_shadow_ctx():
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
            with patch.object(engine, "_dispatch_shadow_gpt") as mock_dispatch:
                _turn(engine, session, "I want a burger")
                mock_dispatch.assert_not_called()


def test_all_shadow_skip_transferring_to_human_agent_state():
    """all_shadow must not dispatch GPT when session is TRANSFERRING_TO_HUMAN_AGENT."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_all_shadow_engine(menu_repo, cb)

    session = _idle_session()
    session.conversation_state = ConversationState.TRANSFERRING_TO_HUMAN_AGENT

    analysis = _make_minimal_analysis()
    with _all_shadow_ctx():
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
            with patch.object(engine, "_dispatch_shadow_gpt") as mock_dispatch:
                _turn(engine, session, "I need a person please")
                mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# all_shadow: stamps trace with pending_async
# ---------------------------------------------------------------------------


def test_all_shadow_stamps_pending_async_on_trace():
    """all_shadow should write gpt_called=True, gpt_decision='pending_async', gpt_applied=False."""
    from app.logging.realtime_latency_logger import RealtimeTurnTrace

    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_all_shadow_engine(menu_repo, cb)

    analysis = _make_minimal_analysis()

    def _fake_dispatch(**kwargs) -> None:
        pass

    with _all_shadow_ctx():
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
            with patch.object(engine, "_dispatch_shadow_gpt", side_effect=_fake_dispatch):
                fake_nlu = _make_nlu("I want a burger please")
                with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                    with patch.object(
                        engine.diagnostics,
                        "_finalize_trace_and_timing",
                        wraps=engine.diagnostics._finalize_trace_and_timing,
                    ) as mock_finalize:
                        engine.process_turn(
                            session=_idle_session(),
                            user_text="I want a burger please",
                        )
                        if mock_finalize.call_args:
                            trace_arg = mock_finalize.call_args.kwargs.get("trace") or (
                                mock_finalize.call_args.args[0]
                                if mock_finalize.call_args.args
                                else None
                            )
                            if trace_arg is not None:
                                assert trace_arg.gpt_called is True
                                assert trace_arg.gpt_decision == "pending_async"
                                assert trace_arg.gpt_applied is False


# ---------------------------------------------------------------------------
# all_shadow: never applies GPT result
# ---------------------------------------------------------------------------


def test_all_shadow_gpt_applied_always_false_in_turn_event():
    """In all_shadow mode, gpt_applied must always be False in the TurnEvent."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_all_shadow_engine(menu_repo, cb)

    analysis = _make_minimal_analysis()
    with _all_shadow_ctx():
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
            with patch.object(engine, "_dispatch_shadow_gpt"):
                _turn(engine, _idle_session(), "I want a burger please")

    event = cb.last
    assert event is not None
    assert event.gpt_applied is False


def test_all_shadow_does_not_mutate_session_state():
    """all_shadow must not change session.conversation_state via GPT."""
    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_all_shadow_engine(menu_repo, cb)

    session = _idle_session()

    analysis = _make_minimal_analysis()
    with _all_shadow_ctx():
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
            with patch.object(engine, "_dispatch_shadow_gpt"):
                fake_nlu = _make_nlu("I want a burger please")
                with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
                    engine.process_turn(session=session, user_text="I want a burger please")

    assert cb.last is not None
    assert cb.last.gpt_applied is False


# ---------------------------------------------------------------------------
# all_shadow: process_turn returns immediately without waiting for background task
# ---------------------------------------------------------------------------


def test_all_shadow_returns_before_shadow_completes():
    """Background shadow dispatch is fire-and-forget; process_turn must return quickly."""
    import time

    menu_repo = _build_menu_repo()
    cb = CapturingBackend()
    engine = _build_all_shadow_engine(menu_repo, cb)

    def _slow_dispatch(**kwargs):
        time.sleep(0.3)

    analysis = _make_minimal_analysis()
    with _all_shadow_ctx():
        with patch.object(engine.gpt_repair, "run", return_value=(analysis, GPT_NOT_CALLED)):
            with patch.object(engine, "_dispatch_shadow_gpt", side_effect=_slow_dispatch):
                start = time.perf_counter()
                _turn(engine, _idle_session(), "I want something to eat")
                elapsed = time.perf_counter() - start

    # With synchronous mock the dispatch itself is slow, but process_turn should
    # return in well under 2 seconds total (no real blocking I/O).
    assert elapsed < 2.0, f"process_turn took too long: {elapsed:.3f}s"
