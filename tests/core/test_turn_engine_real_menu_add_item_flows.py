from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
    with patch("app.core.turn_engine.resolve_nlu", return_value=fake_nlu):
        return engine.process_turn(session=session, user_text=text)


def _new_session(*, caller_device_type: str = "phone") -> Session:
    session = Session(session_id="test-session", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE
    session.conversation_context.caller_device_type = caller_device_type
    return session


def _assert_cart_contains_item(session: Session, expected_item_name: str):
    cart_items = session.cart.get_items()
    assert len(cart_items) == 1
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
                    "ask_for_quantity",
                ),
                ("1", Intent.UNKNOWN, (SlotValue(name="QUANTITY", value="1"),), "item_added_successfully"),
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
                    "repeat_modifier_options",
                ),
                ("no", Intent.DENY, (), "ask_for_modifier"),
                (
                    "Plain Gravy",
                    Intent.ADD_ITEM,
                    (SlotValue(name="MODIFIER", value="Plain Gravy"),),
                    "repeat_modifier_options",
                ),
                ("no", Intent.DENY, (), "ask_for_quantity"),
                ("1", Intent.UNKNOWN, (SlotValue(name="QUANTITY", value="1"),), "item_added_successfully"),
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
                    "repeat_modifier_options",
                ),
                ("no", Intent.DENY, (), "ask_for_quantity"),
                ("1", Intent.UNKNOWN, (SlotValue(name="QUANTITY", value="1"),), "item_added_successfully"),
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
                    "ask_for_quantity",
                ),
                ("1", Intent.UNKNOWN, (SlotValue(name="QUANTITY", value="1"),), "item_added_successfully"),
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
                    "ask_for_quantity",
                ),
                ("1", Intent.UNKNOWN, (SlotValue(name="QUANTITY", value="1"),), "item_added_successfully"),
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
                    "repeat_modifier_options",
                ),
                ("no", Intent.DENY, (), "ask_for_quantity"),
                ("1", Intent.UNKNOWN, (SlotValue(name="QUANTITY", value="1"),), "item_added_successfully"),
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

    assert not seen_cancel_confirmation
    _assert_cart_contains_item(session, item_name)


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
                    "repeat_modifier_options",
                ),
                ("no", Intent.DENY, (), "ask_for_modifier"),
                (
                    "Plain Gravy",
                    Intent.ADD_ITEM,
                    (SlotValue(name="MODIFIER", value="Plain Gravy"),),
                    "repeat_modifier_options",
                ),
                ("no", Intent.DENY, (), "ask_for_quantity"),
                ("1", Intent.UNKNOWN, (SlotValue(name="QUANTITY", value="1"),), "item_added_successfully"),
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
                    "repeat_modifier_options",
                ),
                ("no", Intent.DENY, (), "ask_for_quantity"),
                ("1", Intent.UNKNOWN, (SlotValue(name="QUANTITY", value="1"),), "item_added_successfully"),
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
