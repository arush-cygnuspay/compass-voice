# D:\Working\Cygnus\compass-voice\tests\core\test_turn_engine_phase2_validation.py
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


class CapturingLogger:
    def __init__(self) -> None:
        self.enabled = True
        self.rows: list[dict] = []

    def log_turn(self, **kwargs) -> None:
        self.rows.append(kwargs)


def _build_menu_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "demo"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def _build_engine(menu_repo: MenuRepository, logger: CapturingLogger) -> TurnEngine:
    return TurnEngine(
        router=StateRouter(),
        menu_repo=menu_repo,
        intent_bundle=None,
        slot_bundle=None,
        responder=ResponseBuilder(menu_repo),
        sms_service=StubSmsService(),
        nlu_logger=logger,
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
    session = Session(session_id="phase2-validation", restaurant_id="demo")
    session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE
    session.conversation_context.caller_device_type = caller_device_type
    return session


def test_turn_logger_records_structured_fields_for_add_item_prompt() -> None:
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    _turn(
        engine,
        session,
        "Bourbon Chicken",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Bourbon Chicken"),),
    )

    row = logger.rows[-1]
    assert row["response_key"] == "ask_for_modifier"
    assert row["response_text"]
    assert row["next_state"] == ConversationState.WAITING_FOR_MODIFIER.value
    assert row["pred_intent"] == Intent.ADD_ITEM.value
    assert len(row["slots"]) == 1
    assert row["normalized_values"]["item_name"] == "Bourbon Chicken"
    assert "quantity" in row["missing_required_fields"]
    assert row["raw_user_text"] == "Bourbon Chicken"
    assert row["reprompt_counts"] == {"modifier": 0}
    assert row["latency_breakdown"]["total_ms"] is not None
    assert row["total_ms"] is not None


def test_required_multi_slot_burger_prefills_modifiers_and_negative_onion() -> None:
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    out = _turn(
        engine,
        session,
        "burger with lettuce, tomato, no onions, extra cheese",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Burger"),),
    )

    # Missing quantity now defaults to 1 — item is added directly.
    assert out.response_key == "item_added_successfully"
    assert session.conversation_state == ConversationState.IDLE

    row = logger.rows[-1]
    assert set(row["normalized_values"]["modifiers"]["Sandwich Condiments"]) == {
        "Tomato",
        "extra Cheese",
        "Lettuce",
        "no Grilled Onions",
    }


def test_required_multi_slot_chicken_burger_prefills_required_sides_without_false_multi_item_split() -> None:
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    out = _turn(
        engine,
        session,
        "chicken burger plain bun beef meat american cheese",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Chicken Burger"),),
    )

    assert out.response_key == "ask_for_modifier"
    assert session.conversation_state == ConversationState.WAITING_FOR_MODIFIER
    assert len(session.conversation_context.pending_item_queue) == 0

    normalized = logger.rows[-1]["normalized_values"]
    assert normalized["sides"] == {
        "Choose Cheese": ["American Cheese"],
        "Choose Meat": ["Beef Meat"],
        "Choose Bun": ["Plain Bun"],
    }


def test_partial_modifier_input_adds_item_with_default_quantity() -> None:
    """A burger ordered with a partial modifier list satisfies the required group
    and is added directly — missing quantity defaults to 1 without prompting."""
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    out = _turn(
        engine,
        session,
        "burger with lettuce",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Burger"),),
    )

    assert out.response_key == "item_added_successfully"
    assert session.conversation_state == ConversationState.IDLE


def test_invalid_modifier_is_reported_without_losing_pending_state() -> None:
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    out = _turn(
        engine,
        session,
        "burger with pineapple",
        intent=Intent.ADD_ITEM,
        slots=(
            SlotValue(name="ITEM", value="Burger"),
            SlotValue(name="MODIFIER", value="Pineapple"),
        ),
    )

    assert out.response_key == "ask_for_modifier"
    assert session.conversation_state == ConversationState.WAITING_FOR_MODIFIER
    assert "prefill_feedback" in (out.response_payload or {})
    assert "pineapple" in out.response_payload["prefill_feedback"].lower()


def test_reprompt_guardrail_lists_options_after_third_invalid_side_attempt() -> None:
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    _turn(
        engine,
        session,
        "Chicken Taco",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Chicken Taco"),),
    )

    first = _turn(engine, session, "maybe", intent=Intent.UNKNOWN)
    second = _turn(engine, session, "still not sure", intent=Intent.UNKNOWN)
    third = _turn(engine, session, "whatever", intent=Intent.UNKNOWN)

    assert first.response_key == "repeat_side_options"
    assert second.response_key == "repeat_side_options"
    assert third.response_key == "list_side_options"
    assert session.reprompt_count_by_field["side"] == 3
    assert session.fallback_count >= 3
    assert session.slot_extraction_failure_count >= 3

    row = logger.rows[-1]
    assert row["response_key"] == "list_side_options"
    assert row["reprompt_field"] == "side"
    assert row["reprompt_count"] == 3
    assert row["fallback_triggered"] is True
    assert row["slot_extraction_failed"] is True


def test_quantity_reprompt_guardrail_changes_guidance_after_third_invalid_attempt() -> None:
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    # Vague ordering ("some") enters WAITING_FOR_QUANTITY — plain "Carrot Cake"
    # now defaults to quantity=1 and adds the item directly.
    _turn(
        engine,
        session,
        "some carrot cakes",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Carrot Cake"),),
    )

    first = _turn(engine, session, "a lot", intent=Intent.UNKNOWN)
    second = _turn(engine, session, "some", intent=Intent.UNKNOWN)
    third = _turn(engine, session, "whatever", intent=Intent.UNKNOWN)

    assert first.response_key == "ask_for_quantity"
    assert second.response_key == "ask_for_quantity"
    assert third.response_key == "invalid_quantity_option"
    assert "like 1 or 2" in third.spoken_response_text

    row = logger.rows[-1]
    assert row["reprompt_field"] == "quantity"
    assert row["reprompt_count"] == 3
    assert row["reprompt_escalated"] is True


def test_add_fries_too_routes_directly_into_next_add_item_flow() -> None:
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    _turn(
        engine,
        session,
        "Bourbon Chicken",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Bourbon Chicken"),),
    )
    _turn(
        engine,
        session,
        "Fresh Mushroom",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="MODIFIER", value="Fresh Mushroom"),),
    )
    _turn(
        engine,
        session,
        "1",
        intent=Intent.UNKNOWN,
        slots=(SlotValue(name="QUANTITY", value="1"),),
    )

    out = _turn(
        engine,
        session,
        "add fries too",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Fries"),),
    )

    assert out.response_key == "ask_for_size"
    assert session.conversation_state == ConversationState.WAITING_FOR_SIZE


def test_remove_item_followup_routes_without_yes_no_bridge() -> None:
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    _turn(
        engine,
        session,
        "Carrot Cake",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Carrot Cake"),),
    )
    _turn(
        engine,
        session,
        "1",
        intent=Intent.UNKNOWN,
        slots=(SlotValue(name="QUANTITY", value="1"),),
    )

    out = _turn(
        engine,
        session,
        "remove the carrot cake",
        intent=Intent.REMOVE_ITEM,
        slots=(SlotValue(name="ITEM", value="Carrot Cake"),),
    )

    assert out.response_key == "confirm_remove_item"
    assert session.conversation_state == ConversationState.REMOVING_ITEM


def test_no_thats_all_routes_to_order_review() -> None:
    logger = CapturingLogger()
    menu_repo = _build_menu_repo()
    engine = _build_engine(menu_repo, logger)
    session = _new_session(caller_device_type="phone")

    engine.process_turn(session=session, user_text="pickup")
    _turn(
        engine,
        session,
        "Bourbon Chicken",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="ITEM", value="Bourbon Chicken"),),
    )
    _turn(
        engine,
        session,
        "Fresh Mushroom",
        intent=Intent.ADD_ITEM,
        slots=(SlotValue(name="MODIFIER", value="Fresh Mushroom"),),
    )

    out = _turn(
        engine,
        session,
        "no that's all",
        intent=Intent.UNKNOWN,
        slots=(),
    )

    assert out.response_key == "confirm_order_summary"
    assert session.conversation_state == ConversationState.CONFIRMING_ORDER
