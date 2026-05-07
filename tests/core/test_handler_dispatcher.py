"""Smoke tests for HandlerDispatcher (Commit 6 extraction)."""
import sys
import types
import unittest


_intent_module = types.ModuleType("app.ml.intent.inference_intent")
_slot_module = types.ModuleType("app.ml.slot.inference_slot")


class _IntentBundle:
    pass


class _SlotBundle:
    pass


_intent_module.IntentBundle = _IntentBundle
_intent_module.predict_intent = lambda *a, **k: []
_slot_module.SlotBundle = _SlotBundle
_slot_module.predict_slots = lambda *a, **k: []
sys.modules.setdefault("app.ml.intent.inference_intent", _intent_module)
sys.modules.setdefault("app.ml.slot.inference_slot", _slot_module)

for _name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
sys.modules["twilio.rest"].Client = type("_Client", (), {"__init__": lambda *a, **k: None})

_redis_module = types.ModuleType("redis")
_redis_module.Redis = type("_Redis", (), {"__init__": lambda *a, **k: None})
sys.modules.setdefault("redis", _redis_module)


from pathlib import Path

from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
from app.core.command_executor import CommandExecutor
from app.core.handler_dispatcher import HandlerDispatcher
from app.core.response_builder import ResponseBuilder
from app.core.turn_diagnostics import TurnDiagnostics
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.services.checkout_service import CheckoutService


class _StubSmsService:
    def is_configured(self):
        return False

    def send(self, request):
        from types import SimpleNamespace
        return SimpleNamespace(ok=False, sid=None, error_code="x", error_message="x")


def _build_dispatcher() -> HandlerDispatcher:
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "demo"
    menu_repo = MenuRepository(
        MenuStore(
            menu_path=data_root / "menu.json",
            entity_index_path=data_root / "entity_index.json",
        )
    )
    sms = _StubSmsService()
    return HandlerDispatcher(
        menu_repo=menu_repo,
        cart_summary_builder=CartSummaryBuilder(menu_repo),
        sms_service=sms,
        checkout_service=CheckoutService(),
        responder=ResponseBuilder(menu_repo),
        command_executor=CommandExecutor(sms),
        diagnostics=TurnDiagnostics(backends=[]),
    )


_EXPECTED_HANDLER_NAMES = {
    "add_item_handler",
    "waiting_for_side_handler",
    "waiting_for_modifier_handler",
    "waiting_for_size_handler",
    "waiting_for_side_size_handler",
    "waiting_for_quantity_handler",
    "confirming_handler",
    "modifying_item_handler",
    "remove_item_handler",
    "removing_item_handler",
    "start_order_handler",
    "confirming_order_handler",
    "waiting_for_payment_handler",
    "waiting_for_checkout_completion_handler",
    "cart_handler",
    "cancellation_confirmation_handler",
    "ask_menu_info_handler",
    "ask_price_handler",
    "waiting_for_order_type_handler",
    "waiting_for_delivery_eligibility_handler",
    "waiting_for_delivery_address_collection_handler",
}


class HandlerDispatcherSmokeTests(unittest.TestCase):
    def test_registry_contains_every_expected_handler_name(self):
        dispatcher = _build_dispatcher()
        for name in _EXPECTED_HANDLER_NAMES:
            with self.subTest(name=name):
                handler = dispatcher.get_handler(name)
                self.assertIsNotNone(
                    handler, f"missing handler in registry: {name}"
                )

    def test_get_handler_returns_none_for_unknown_name(self):
        dispatcher = _build_dispatcher()
        self.assertIsNone(dispatcher.get_handler("not_a_handler"))


if __name__ == "__main__":
    unittest.main()
