from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import types

import pytest

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


def _new_session(*, caller_device_type: str = "phone") -> Session:
    session = Session(session_id="test-session", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE
    session.conversation_context.caller_device_type = caller_device_type
    return session


def _assert_cart_contains_item(session: Session, expected_item_name: str):
    cart_items = session.cart.get_items()
    assert len(cart_items) == 1
    assert cart_items[0].quantity == 1
    assert session.conversation_state == ConversationState.IDLE
    assert session.last_response_key == "item_added_successfully"
    assert session.conversation_context.pending_add_item is None
    assert session.conversation_context.current_item_id is None
    assert session.conversation_context.current_item_name is None


@pytest.mark.parametrize(
    ("item_name", "turns"),
    [
        (
            "Bourbon Chicken",
            [
                ("pickup", None, (), "order_type_captured_pickup"),
                (
                    "Bourbon Chicken",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Bourbon Chicken"),),
                    "ask_for_modifier",
                ),
                (
                    "Fresh Mushroom",
                    Intent.ADD_ITEM,
                    (SlotValue(name="MODIFIER", value="Fresh Mushroom"),),
                    "item_added_successfully",
                ),
            ],
        ),
        (
            "Chicken Taco",
            [
                ("pickup", None, (), "order_type_captured_pickup"),
                (
                    "Chicken Taco",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Chicken Taco"),),
                    "ask_for_side",
                ),
                (
                    "Coke (12 oz.)",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Coke (12 oz.)"),),
                    "ask_for_modifier",
                ),
                (
                    "Steak",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Steak"),),
                    "ask_for_modifier",
                ),
                ("no", Intent.DENY, (), "required_modifier_cannot_skip"),
                (
                    "Plain Gravy",
                    Intent.ADD_ITEM,
                    (SlotValue(name="MODIFIER", value="Plain Gravy"),),
                    "item_added_successfully",
                ),
            ],
        ),
        (
            "Iced Mocha",
            [
                ("pickup", None, (), "order_type_captured_pickup"),
                (
                    "Iced Mocha",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Iced Mocha"),),
                    "ask_for_size",
                ),
                ("Small", Intent.UNKNOWN, (SlotValue(name="SIZE", value="Small"),), "ask_for_modifier"),
                (
                    "Cream",
                    Intent.ADD_ITEM,
                    (SlotValue(name="MODIFIER", value="Cream"),),
                    "item_added_successfully",
                ),
            ],
        ),
        (
            "49. Seafood Combo Platter",
            [
                ("pickup", None, (), "order_type_captured_pickup"),
                (
                    "49. Seafood Combo Platter",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="49. Seafood Combo Platter"),),
                    "ask_for_side",
                ),
                (
                    "Rice and Cole Slaw",
                    Intent.ADD_ITEM,
                    (
                        SlotValue(name="ITEM", value="Rice"),
                        SlotValue(name="ITEM", value="Cole Slaw"),
                    ),
                    "item_added_successfully",
                ),
            ],
        ),
        (
            "61. 50 Wings",
            [
                ("pickup", None, (), "order_type_captured_pickup"),
                (
                    "61. 50 Wings",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="61. 50 Wings"),),
                    "ask_for_side",
                ),
                (
                    "Hot",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Hot"),),
                    "ask_for_side",
                ),
                (
                    "Fried",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Fried"),),
                    "item_added_successfully",
                ),
            ],
        ),
        (
            "Crabcake Combo",
            [
                ("pickup", None, (), "order_type_captured_pickup"),
                (
                    "Crabcake Combo",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Crabcake Combo"),),
                    "ask_for_side",
                ),
                (
                    "Coke",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Coke"),),
                    "ask_for_side_size",
                ),
                ("Small", Intent.UNKNOWN, (SlotValue(name="SIZE", value="Small"),), "ask_for_modifier"),
                (
                    "Mustard",
                    Intent.ADD_ITEM,
                    (SlotValue(name="MODIFIER", value="Mustard"),),
                    "item_added_successfully",
                ),
            ],
        ),
    ],
)
def test_real_menu_pickup_add_item_flows(item_name, turns):
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)
    session = _new_session(caller_device_type="phone")

    seen_cancel_confirmation = False

    for text, intent, slots, expected_key in turns:
        if intent is None:
            out = engine.process_turn(session=session, user_text=text)
        else:
            out = _turn(engine, session, text, intent=intent, slots=slots)

        assert out.response_key == expected_key
        if out.response_key == "confirm_cancel_current_item_for_new_request":
            seen_cancel_confirmation = True


def test_no_thats_all_after_item_added_transitions_to_order_review():
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    _turn(
        engine,
        session,
        "Bourbon Chicken",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Bourbon Chicken"),),
    )
    added = _turn(
        engine,
        session,
        "Fresh Mushroom",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="MODIFIER", value="Fresh Mushroom"),),
    )
    assert added.response_key == "item_added_successfully"

    review = _turn(
        engine,
        session,
        "no that's all",
        intent=Intent.UNKNOWN,
        slots=(),
    )

    assert review.response_key == "confirm_order_summary"
    assert session.conversation_state == ConversationState.CONFIRMING_ORDER


@pytest.mark.parametrize(
    ("caller_device_type", "item_name", "item_turns"),
    [
        (
            "phone",
            "Chicken Taco",
            [
                (
                    "Chicken Taco",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Chicken Taco"),),
                    "ask_for_side",
                ),
                (
                    "Coke (12 oz.)",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Coke (12 oz.)"),),
                    "ask_for_modifier",
                ),
                (
                    "Steak",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Steak"),),
                    "ask_for_modifier",
                ),
                ("no", Intent.DENY, (), "required_modifier_cannot_skip"),
                (
                    "Plain Gravy",
                    Intent.ADD_ITEM,
                    (SlotValue(name="MODIFIER", value="Plain Gravy"),),
                    "item_added_successfully",
                ),
            ],
        ),
        (
            "chat",
            "Iced Mocha",
            [
                (
                    "Iced Mocha",
                    Intent.ADD_ITEM,
                    (SlotValue(name="ITEM", value="Iced Mocha"),),
                    "ask_for_size",
                ),
                ("Medium", Intent.UNKNOWN, (SlotValue(name="SIZE", value="Medium"),), "ask_for_modifier"),
                (
                    "Sugar",
                    Intent.ADD_ITEM,
                    (SlotValue(name="MODIFIER", value="Sugar"),),
                    "item_added_successfully",
                ),
            ],
        ),
    ],
)
def test_real_menu_delivery_add_item_flows(caller_device_type, item_name, item_turns):
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)
    session = _new_session(caller_device_type=caller_device_type)

    onboarding_turns = [
        ("delivery", None, (), "ask_for_delivery_area"),
        ("Washington DV", Intent.UNKNOWN, (), "ask_for_delivery_zip"),
        ("21000", Intent.UNKNOWN, (), "confirm_delivery_area_zip"),
        ("yes", Intent.CONFIRM, (), "delivery_area_confirmed"),
    ]

    seen_cancel_confirmation = False

    for text, intent, slots, expected_key in onboarding_turns + item_turns:
        if intent is None:
            out = engine.process_turn(session=session, user_text=text)
        else:
            out = _turn(engine, session, text, intent=intent, slots=slots)

        assert out.response_key == expected_key
        if out.response_key == "confirm_cancel_current_item_for_new_request":
            seen_cancel_confirmation = True

    assert not seen_cancel_confirmation
    assert session.conversation_context.order_type == "delivery"
    assert session.conversation_context.delivery_address.area == "washington dv"
    assert session.conversation_context.delivery_address.postal_code == "21000"
    _assert_cart_contains_item(session, item_name)


def test_real_menu_pickup_add_item_prefills_sparse_side_and_modifiers() -> None:
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo)
    session = _new_session(caller_device_type="phone")

    assert engine.process_turn(session=session, user_text="pickup").response_key == "order_type_captured_pickup"

    out = _turn(
        engine,
        session,
        "a chicken taco with small coke and sausage jelly",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Chicken Taco"),),
    )

    assert out.response_key == "item_added_successfully"
    assert out.response_payload["prefilled_summary"] == "with Coke (12 oz.), Sausage, and Jelly"
    _assert_cart_contains_item(session, "Chicken Taco")
