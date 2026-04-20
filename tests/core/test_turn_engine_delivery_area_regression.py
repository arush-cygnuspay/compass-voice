from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    with patch("app.core.turn_engine.resolve_nlu", return_value=fake_nlu):
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
