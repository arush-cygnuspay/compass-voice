from pathlib import Path
from types import SimpleNamespace
import sys
import time
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
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


class StubCheckoutService:
    def verify_payment_by_order_number(self, order_number: str) -> dict:
        return {
            "ok": True,
            "paid": False,
            "payment_completed": False,
            "status": "pending",
            "reference": None,
            "session": None,
            "error": None,
        }


class StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        return SimpleNamespace(
            ok=False,
            sid=None,
            error_code="not_configured",
            error_message="not configured",
        )


def _build_menu_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "demo"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def _build_engine(menu_repo: MenuRepository) -> TurnEngine:
    engine = TurnEngine(
        router=StateRouter(),
        menu_repo=menu_repo,
        intent_bundle=None,
        slot_bundle=None,
        responder=ResponseBuilder(menu_repo),
        sms_service=StubSmsService(),
        nlu_logger=SimpleNamespace(enabled=False),
    )
    stub = StubCheckoutService()
    engine.checkout_service = stub
    engine.dispatcher.handlers["waiting_for_payment_handler"].checkout_service = stub
    engine.dispatcher.handlers["waiting_for_checkout_completion_handler"].checkout_service = stub
    # Re-bind nested orchestrators that captured the original checkout_service.
    engine.payment_flow.checkout_service = stub
    engine.dispatcher.checkout_service = stub
    return engine


def test_auto_payment_check_suppresses_repeated_pending_prompt_inside_cooldown() -> None:
    engine = _build_engine(_build_menu_repo())
    session = Session(session_id="payment-cooldown", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
    session.last_response_key = "waiting_for_checkout_completion"
    session.last_response_payload = None
    session.last_response_at_epoch = time.time()
    session.conversation_context.delivery_address.order_number = "ord-123"
    session.conversation_context.delivery_address.payment_status_last_prompt_at_epoch = (
        session.last_response_at_epoch
    )
    session.conversation_context.delivery_address.payment_status_last_response_key = (
        "waiting_for_checkout_completion"
    )

    out = engine.payment_flow._handle_auto_payment_check(session)

    assert out.response_key == "waiting_for_checkout_completion"
    assert out.spoken_response_text == ""
    assert out.internal_response_text == ""
    assert session.conversation_state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION


def test_auto_payment_check_replays_checkout_pending_prompt_after_interval() -> None:
    engine = _build_engine(_build_menu_repo())
    session = Session(session_id="payment-cooldown-expired", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_CHECKOUT_COMPLETION
    session.last_response_key = "waiting_for_checkout_completion"
    session.last_response_payload = None
    session.last_response_at_epoch = time.time() - 31
    session.conversation_context.delivery_address.order_number = "ord-456"
    session.conversation_context.delivery_address.last_checkout_wait_prompt_at_epoch = (
        session.last_response_at_epoch
    )
    session.conversation_context.delivery_address.last_checkout_wait_response_key = (
        "waiting_for_checkout_completion"
    )

    out = engine.payment_flow._handle_auto_payment_check(session)

    assert out.response_key == "waiting_for_checkout_completion"
    assert out.spoken_response_text != ""
    assert session.conversation_state == ConversationState.WAITING_FOR_CHECKOUT_COMPLETION


def test_auto_payment_check_is_noop_after_payment_is_already_completed() -> None:
    engine = _build_engine(_build_menu_repo())
    session = Session(session_id="payment-complete", restaurant_id="demo")
    session.conversation_state = ConversationState.COMPLETED
    session.last_response_key = "order_completed"
    session.last_response_payload = {"order_number": "ord-789"}

    out = engine.payment_flow._handle_auto_payment_check(session)

    assert out.response_key == "order_completed"
    assert out.spoken_response_text == ""
    assert out.internal_response_text == ""
