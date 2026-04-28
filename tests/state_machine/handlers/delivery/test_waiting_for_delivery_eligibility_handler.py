from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
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
from app.nlu.nlu_result import NLUResult
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.handlers.delivery.waiting_for_delivery_eligibility_handler import (
    WaitingForDeliveryEligibilityHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


class StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        return SimpleNamespace(ok=False, sid=None, error_code="not_configured", error_message="not configured")


def _build_menu_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[4] / "app" / "data" / "restaurants" / "demo"
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


def _make_nlu(
    text: str,
    *,
    effective_intent: Intent,
    confidence: float = 0.99,
    model_sub_intent: str | None = None,
) -> NLUResult:
    return NLUResult(
        effective_intent=effective_intent,
        intent_confidence=confidence,
        normalized_text=normalize_text(text),
        raw_text=text,
        model_main_intent=effective_intent.value,
        model_sub_intent=model_sub_intent or effective_intent.value,
        slots=(),
        slot_model_ran=False,
    )


def _turn(
    engine: TurnEngine,
    session: Session,
    text: str,
    *,
    effective_intent: Intent,
    model_sub_intent: str | None = None,
):
    fake_nlu = _make_nlu(
        text,
        effective_intent=effective_intent,
        model_sub_intent=model_sub_intent,
    )
    with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
        return engine.process_turn(session=session, user_text=text)


class WaitingForDeliveryEligibilityHandlerTests(TestCase):
    def _build_confirmation_context(self, text: str, *, effective_intent: Intent, model_sub_intent: str | None = None) -> ConversationContext:
        context = ConversationContext()
        context.current_prompt_field = "delivery_eligibility_confirmation"
        context.delivery_address.area = "washington dc"
        context.delivery_address.postal_code = "21000"
        context.last_nlu = _make_nlu(
            text,
            effective_intent=effective_intent,
            model_sub_intent=model_sub_intent,
        )
        context.last_intent_confidence = context.last_nlu.intent_confidence
        return context

    def test_affirm_confirm_semantics_complete_delivery_eligibility_confirmation(self) -> None:
        cases = (
            ("yeah that's correct", Intent.CONFIRM, "affirm"),
            ("yes that's correct", Intent.CONFIRM, "affirm"),
            ("yep", Intent.AFFIRM, "affirm"),
            ("yeah that's corredct", Intent.CONFIRM, "affirm"),
        )

        for text, effective_intent, model_sub_intent in cases:
            with self.subTest(text=text):
                context = self._build_confirmation_context(
                    text,
                    effective_intent=effective_intent,
                    model_sub_intent=model_sub_intent,
                )

                result = WaitingForDeliveryEligibilityHandler().handle(
                    intent=Intent.UNKNOWN,
                    context=context,
                    user_text=text,
                    session=None,
                )

                self.assertEqual(result.response_key, "delivery_area_confirmed")
                self.assertEqual(result.next_state, ConversationState.IDLE)
                self.assertEqual(context.current_prompt_field, None)
                self.assertEqual(context.delivery_address.area, "washington dc")
                self.assertEqual(context.delivery_address.postal_code, "21000")
                self.assertTrue(context.delivery_address.area_serviceable)
                self.assertTrue(context.onboarding_complete)

    def test_deny_reopens_delivery_area_capture(self) -> None:
        context = self._build_confirmation_context(
            "no that's wrong",
            effective_intent=Intent.DENY,
            model_sub_intent="deny",
        )

        result = WaitingForDeliveryEligibilityHandler().handle(
            intent=Intent.UNKNOWN,
            context=context,
            user_text="no that's wrong",
            session=None,
        )

        self.assertEqual(result.response_key, "ask_for_delivery_area")
        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_DELIVERY_ELIGIBILITY)
        self.assertEqual(context.current_prompt_field, "delivery_area")
        self.assertIsNone(context.delivery_address.area)
        self.assertIsNone(context.delivery_address.postal_code)
        self.assertIsNone(context.delivery_address.area_serviceable)
        self.assertFalse(context.onboarding_complete)

    def test_turn_engine_does_not_repeat_delivery_confirmation_after_first_affirm(self) -> None:
        engine = _build_engine(_build_menu_repo())
        session = Session(session_id="delivery-confirm-loop", restaurant_id="demo")
        session.conversation_state = ConversationState.WAITING_FOR_ORDER_TYPE

        self.assertEqual(_turn(engine, session, "delivery", effective_intent=Intent.UNKNOWN).response_key, "ask_for_delivery_area")
        self.assertEqual(_turn(engine, session, "washington dc", effective_intent=Intent.UNKNOWN).response_key, "ask_for_delivery_zip")
        self.assertEqual(_turn(engine, session, "21000", effective_intent=Intent.UNKNOWN).response_key, "confirm_delivery_area_zip")

        confirm = _turn(
            engine,
            session,
            "yeah that's correct",
            effective_intent=Intent.CONFIRM,
            model_sub_intent="affirm",
        )

        self.assertEqual(confirm.response_key, "delivery_area_confirmed")
        self.assertEqual(session.conversation_state, ConversationState.IDLE)
        self.assertIsNone(session.conversation_context.current_prompt_field)

        followup = _turn(
            engine,
            session,
            "yep",
            effective_intent=Intent.AFFIRM,
            model_sub_intent="affirm",
        )

        self.assertNotEqual(followup.response_key, "repeat_delivery_area_zip_confirmation")
        self.assertEqual(session.conversation_state, ConversationState.IDLE)
