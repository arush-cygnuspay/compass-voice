# D:/Working/Cygnus/compass-voice/tests/support/voice_test_harness.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import types


def install_test_stubs() -> None:
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


install_test_stubs()

from app.core.response_builder import ResponseBuilder
from app.core.turn_engine import TurnEngine, TurnOutput
from app.cart.cart_item import CartItem
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.state_router import StateRouter


@dataclass(frozen=True, slots=True)
class ScriptedTurn:
    utterance: str
    intent: Intent = Intent.UNKNOWN
    slots: tuple[SlotValue, ...] = ()
    confidence: float = 0.99


@dataclass(frozen=True, slots=True)
class SimulatedTurn:
    utterance: str
    response_key: str
    response_text: str
    state_after: ConversationState
    response_payload: dict | None


class StubSmsService:
    def __init__(self, *, configured: bool = True) -> None:
        self.configured = configured
        self.requests: list[object] = []

    def is_configured(self):
        return self.configured

    def send(self, request):
        from app.services.sms_exceptions import PermanentSmsError
        from app.services.sms_service import SmsSendResult
        self.requests.append(request)
        if not self.configured:
            raise PermanentSmsError(
                "Twilio SMS is not configured.",
                error_code="sms_not_configured",
            )
        return SmsSendResult(ok=True, sid="SM-TEST")


class StubCheckoutService:
    def __init__(self) -> None:
        self.ensure_calls: list[dict] = []
        self.create_calls: list[dict] = []
        self.verify_calls: list[str] = []
        self.verify_sequences: dict[str, list[dict]] = {}
        self.default_order_number = "1234567"
        self.default_token = "checkout-token"
        self.default_payment_link = "https://checkout.example/pay"

    def queue_verify_results(self, order_number: str, *results: dict) -> None:
        self.verify_sequences[order_number] = [dict(result) for result in results]

    def create_session(self, **kwargs):
        self.create_calls.append(dict(kwargs))
        order_number = kwargs.get("order_number") or self.default_order_number
        token = kwargs.get("token") or self.default_token
        return SimpleNamespace(order_number=order_number, token=token)

    def build_checkout_url(self, token: str) -> str:
        return f"https://checkout.example/{token}"

    def ensure_payment_link(self, **kwargs) -> dict:
        self.ensure_calls.append(dict(kwargs))
        order_number = kwargs.get("order_number") or self.default_order_number
        return {
            "ok": True,
            "order_number": order_number,
            "payment_completed": False,
            "redirect_url": self.default_payment_link,
            "confirmation_link": f"{self.default_payment_link}/confirm",
            "status": "pending",
        }

    def verify_payment_by_order_number(self, order_number: str) -> dict:
        self.verify_calls.append(order_number)
        queued = self.verify_sequences.get(order_number) or []
        if queued:
            return queued.pop(0)
        return {
            "ok": True,
            "paid": False,
            "payment_completed": False,
            "status": "pending",
            "reference": None,
            "session": None,
            "error": None,
        }


class CapturingNluLogger:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.rows: list[dict] = []

    def log_turn(self, **kwargs) -> None:
        self.rows.append(kwargs)


def build_menu_repo() -> MenuRepository:
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "demo"
    store = MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )
    return MenuRepository(store)


def build_engine(
    *,
    menu_repo: MenuRepository | None = None,
    sms_service: StubSmsService | None = None,
    checkout_service: StubCheckoutService | None = None,
    logger: CapturingNluLogger | None = None,
) -> TurnEngine:
    menu_repo = menu_repo or build_menu_repo()
    sms_service = sms_service or StubSmsService()
    logger = logger or CapturingNluLogger(enabled=False)
    engine = TurnEngine(
        router=StateRouter(),
        menu_repo=menu_repo,
        intent_bundle=None,
        slot_bundle=None,
        responder=ResponseBuilder(menu_repo),
        sms_service=sms_service,
        nlu_logger=logger,
    )
    if checkout_service is not None:
        engine.checkout_service = checkout_service
        engine.dispatcher.handlers["confirming_order_handler"].checkout_service = checkout_service
        engine.dispatcher.handlers["waiting_for_pickup_sms_permission_handler"].checkout_service = checkout_service
        engine.dispatcher.handlers["waiting_for_payment_handler"].checkout_service = checkout_service
        engine.dispatcher.handlers["waiting_for_checkout_completion_handler"].checkout_service = checkout_service
        if "waiting_for_delivery_address_collection_handler" in engine.dispatcher.handlers:
            engine.dispatcher.handlers["waiting_for_delivery_address_collection_handler"].checkout_service = checkout_service
        # Re-bind nested orchestrators that captured the original checkout_service.
        engine.payment_flow.checkout_service = checkout_service
        engine.dispatcher.checkout_service = checkout_service
    return engine


def new_session(
    *,
    state: ConversationState = ConversationState.WAITING_FOR_ORDER_TYPE,
    caller_device_type: str = "phone",
    order_type: str | None = None,
) -> Session:
    session = Session(session_id="voice-test-session", restaurant_id="demo")
    session.conversation_state = state
    session.conversation_context.caller_device_type = caller_device_type
    session.conversation_context.order_type = order_type
    session.conversation_context.delivery_address.customer_phone_number = "+15555550123"
    return session


def seed_cart_item(
    session: Session,
    *,
    item_id: str,
    quantity: int = 1,
    variant_id: str | None = None,
    sides: dict[str, list[str]] | None = None,
    side_variants: dict[str, str] | None = None,
    modifiers: dict[str, list[object]] | None = None,
) -> None:
    menu_repo = build_menu_repo()
    resolved_item_id = item_id
    try:
        menu_repo.get_item(item_id)
    except KeyError:
        resolved = menu_repo.resolve_menu_query(item_id, limit=5)
        if getattr(resolved, "item", None) is not None:
            resolved_item_id = resolved.item.item_id
    session.cart.add_item(
        CartItem.create(
            item_id=resolved_item_id,
            quantity=quantity,
            variant_id=variant_id,
            sides=sides or {},
            side_variants=side_variants or {},
            modifiers=modifiers or {},
        )
    )


def make_slot(name: str, value: str) -> SlotValue:
    return SlotValue(name=name, value=value, raw=value)


def make_nlu(turn: ScriptedTurn) -> SimpleNamespace:
    normalized = normalize_text(turn.utterance)
    return SimpleNamespace(
        effective_intent=turn.intent,
        intent_confidence=turn.confidence,
        normalized_text=normalized,
        slots=turn.slots,
        model_main_intent=turn.intent.value,
        model_sub_intent=turn.intent.value,
        slot_model_ran=bool(turn.slots),
        intent_model_ms=None,
        slot_model_ms=None,
    )


def simulate_turn(
    engine: TurnEngine,
    session: Session,
    turn: ScriptedTurn,
) -> SimulatedTurn:
    if turn.utterance == "__auto_payment_check__":
        output = engine.process_turn(session=session, user_text=turn.utterance)
    else:
        fake_nlu = make_nlu(turn)
        with patch("app.core.nlu_orchestrator.resolve_nlu", return_value=fake_nlu):
            output = engine.process_turn(session=session, user_text=turn.utterance)
    return _to_simulated_turn(output, turn.utterance, session.conversation_state)


def simulate_conversation(
    engine: TurnEngine,
    session: Session,
    turns: list[ScriptedTurn],
) -> list[SimulatedTurn]:
    return [simulate_turn(engine, session, turn) for turn in turns]


def response_text(output: TurnOutput | SimulatedTurn) -> str:
    if isinstance(output, SimulatedTurn):
        return output.response_text
    return str(output.spoken_response_text or output.internal_response_text or "")


def response_mentions(output: TurnOutput | SimulatedTurn, *phrases: str) -> bool:
    text = normalize_text(response_text(output))
    return all(normalize_text(phrase) in text for phrase in phrases)


def _to_simulated_turn(
    output: TurnOutput,
    utterance: str,
    state_after: ConversationState,
) -> SimulatedTurn:
    return SimulatedTurn(
        utterance=utterance,
        response_key=output.response_key,
        response_text=response_text(output),
        state_after=state_after,
        response_payload=output.response_payload,
    )
