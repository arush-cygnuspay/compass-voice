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
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


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


def _make_nlu(text: str, intent: Intent, slots: tuple[SlotValue, ...]) -> SimpleNamespace:
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


def _turn(
    engine: TurnEngine,
    session: Session,
    text: str,
    *,
    intent: Intent = Intent.UNKNOWN,
    slots: tuple[SlotValue, ...] = (),
):
    fake_nlu = _make_nlu(text, intent, slots)
    with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
        return engine.process_turn(session=session, user_text=text)


def test_turn_engine_accepts_delivery_area_when_nlu_emits_noisy_item_slot():
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)
    session = Session(session_id="delivery-area-regression", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    first = _turn(engine, session, "delivery")
    assert first.response_key == "ask_for_delivery_area"
    assert session.conversation_state == ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY
    assert session.conversation_context.current_prompt_field == "delivery_area"

    second = _turn(
        engine,
        session,
        "washington dc",
        intent=Intent.UNKNOWN,
        slots=(SlotValue(name="ITEM", value="washington dc"),),
    )
    assert second.response_key == "ask_for_delivery_zip"
    assert session.conversation_context.delivery_address.area == "washington dc"
    assert session.conversation_context.current_prompt_field == "delivery_postal_code"


def test_turn_engine_accepts_zip_when_nlu_misclassifies_it_as_ask_price():
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)
    session = Session(session_id="delivery-zip-regression", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    first = _turn(engine, session, "delivery")
    assert first.response_key == "ask_for_delivery_area"

    second = _turn(
        engine,
        session,
        "washington dc",
        intent=Intent.UNKNOWN,
        slots=(SlotValue(name="ITEM", value="washington dc"),),
    )
    assert second.response_key == "ask_for_delivery_zip"

    third = _turn(
        engine,
        session,
        "it's 21000",
        intent=Intent.ASK_PRICE,
        slots=(),
    )
    assert third.response_key == "confirm_delivery_area_zip"
    assert session.conversation_context.delivery_address.postal_code == "21000"
    assert session.conversation_context.current_prompt_field == "delivery_eligibility_confirmation"


def test_turn_engine_accepts_zip_from_mixed_phrase_when_nlu_misclassifies_it() -> None:
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)
    session = Session(session_id="delivery-zip-mixed-regression", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    assert _turn(engine, session, "delivery").response_key == "ask_for_delivery_area"
    assert _turn(engine, session, "washington dc").response_key == "ask_for_delivery_zip"

    third = _turn(
        engine,
        session,
        "my zip code is 30000.",
        intent=Intent.ASK_PRICE,
        slots=(),
    )
    assert third.response_key == "confirm_delivery_area_zip"
    assert session.conversation_context.delivery_address.postal_code == "30000"


def test_turn_engine_accepts_spoken_number_zip_phrase() -> None:
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)
    session = Session(session_id="delivery-zip-spoken-regression", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    assert _turn(engine, session, "delivery").response_key == "ask_for_delivery_area"
    assert _turn(engine, session, "washington dc").response_key == "ask_for_delivery_zip"

    third = _turn(
        engine,
        session,
        "it's twenty one thousand",
        intent=Intent.ASK_PRICE,
        slots=(),
    )
    assert third.response_key == "confirm_delivery_area_zip"


def test_turn_engine_accepts_semantic_affirm_for_delivery_eligibility_confirmation() -> None:
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)
    session = Session(session_id="delivery-confirm-semantic", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

    assert _turn(engine, session, "delivery").response_key == "ask_for_delivery_area"
    assert _turn(engine, session, "washington dc").response_key == "ask_for_delivery_zip"
    assert _turn(engine, session, "21000", intent=Intent.UNKNOWN).response_key == "confirm_delivery_area_zip"

    confirm = _turn(
        engine,
        session,
        "okay",
        intent=Intent.UNKNOWN,
        slots=(),
    )

    assert confirm.response_key == "delivery_area_confirmed"
    assert session.conversation_state == ConversationState.IDLE
    assert session.conversation_context.delivery_address.postal_code == "21000"
