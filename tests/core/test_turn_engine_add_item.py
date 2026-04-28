from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import types

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

from app.core.response_builder import ResponseBuilder
from app.core.turn_engine import TurnEngine
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


class StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        return SimpleNamespace(ok=False, sid=None, error_code="not_configured", error_message="not configured")


def _build_menu_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "demo"
    return MenuRepository(
        MenuStore(
            menu_path=data_root / "menu.json",
            entity_index_path=data_root / "entity_index.json",
        )
    )


def _build_engine(menu_repo: MenuRepository) -> TurnEngine:
    return TurnEngine(
        router=StateRouter(),
        menu_repo=menu_repo,
        intent_bundle=None,
        slot_bundle=None,
        responder=ResponseBuilder(menu_repo),
        sms_service=StubSmsService(),
        nlu_logger=SimpleNamespace(enabled=False),
    )


def test_turn_engine_add_item_happy_path():
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)

    session = Session(session_id="s1", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE
    session.conversation_context.caller_device_type = "phone"

    assert engine.process_turn(session, "pickup").response_key == "order_type_captured_pickup"

    fake_nlu = SimpleNamespace(
        effective_intent=Intent.ADD_ITEM,
        intent_confidence=0.99,
        raw_text="zinger burger",
        normalized_text=normalize_text("zinger burger"),
        slots=(SlotValue(name="ITEM", value="Burger"),),
        model_main_intent=Intent.ADD_ITEM.value,
        model_sub_intent=Intent.ADD_ITEM.value,
        slot_model_ran=True,
        intent_model_ms=None,
        slot_model_ms=None,
    )

    with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
        out = engine.process_turn(session, "zinger burger")

    assert out.response_key in {"ask_for_modifier", "ask_for_quantity"}
    assert session.last_intent == Intent.ADD_ITEM
