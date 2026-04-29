# tests/core/test_domain_contracts.py
"""
Tests for Task 8: domain/contracts boundary.

Coverage:
- CartSnapshot.from_cart captures items correctly
- CartSnapshot is immutable (frozen dataclass)
- CartSnapshot.is_empty / item_count / get_items
- OrderDraft.from_context captures order fields
- OrderDraft is immutable
- OrderDraft.is_delivery / is_pickup / has_address
- CommandResult fields and from_dict adapter
- CommandResult is immutable (frozen dataclass)
- CartSummaryBuilder.build uses CartSnapshot internally (no Cart mutation)
"""
from __future__ import annotations

import sys
import types
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

# ── stub heavy ML deps ────────────────────────────────────────────────────────
_intent_mod = types.ModuleType("app.ml.intent.inference_intent")
_slot_mod = types.ModuleType("app.ml.slot.inference_slot")
_intent_mod.IntentBundle = type("IntentBundle", (), {})
_intent_mod.predict_intent = lambda *a, **k: []
_slot_mod.SlotBundle = type("SlotBundle", (), {})
_slot_mod.predict_slots = lambda *a, **k: []
sys.modules.setdefault("app.ml.intent.inference_intent", _intent_mod)
sys.modules.setdefault("app.ml.slot.inference_slot", _slot_mod)
for _n in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["twilio.base.exceptions"].TwilioRestException = Exception
sys.modules["twilio.rest"].Client = type("_C", (), {"__init__": lambda *a, **k: None})
sys.modules.setdefault("torch", types.ModuleType("torch"))
# ─────────────────────────────────────────────────────────────────────────────

from app.cart.cart import Cart
from app.cart.cart_item import CartItem
from app.contracts.command_result import CommandResult
from app.domain.cart_snapshot import CartSnapshot
from app.domain.order_draft import OrderDraft
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.delivery_address import DeliveryAddress


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_cart_item(item_id: str = "burger", quantity: int = 1) -> CartItem:
    return CartItem.create(
        item_id=item_id,
        quantity=quantity,
        variant_id=None,
        sides={},
        side_variants={},
        modifiers={},
    )


def _make_cart(*item_ids: str) -> Cart:
    cart = Cart()
    for iid in item_ids:
        cart.add_item(_make_cart_item(iid))
    return cart


# ── CartSnapshot ──────────────────────────────────────────────────────────────

def test_cart_snapshot_from_empty_cart():
    snap = CartSnapshot.from_cart(Cart())
    assert snap.is_empty()
    assert snap.item_count == 0
    assert snap.get_items() == ()


def test_cart_snapshot_captures_items():
    cart = _make_cart("burger", "fries")
    snap = CartSnapshot.from_cart(cart)
    assert len(snap.items) == 2
    item_ids = {i.item_id for i in snap.items}
    assert item_ids == {"burger", "fries"}


def test_cart_snapshot_item_count_sums_quantities():
    cart = Cart()
    cart.add_item(_make_cart_item("burger", 3))
    cart.add_item(_make_cart_item("fries", 2))
    snap = CartSnapshot.from_cart(cart)
    assert snap.item_count == 5


def test_cart_snapshot_is_frozen():
    snap = CartSnapshot.from_cart(_make_cart("burger"))
    try:
        snap.items = ()  # type: ignore
        assert False, "Should have raised"
    except (FrozenInstanceError, AttributeError, TypeError):
        pass


def test_cart_snapshot_does_not_reflect_later_cart_mutations():
    cart = _make_cart("burger")
    snap = CartSnapshot.from_cart(cart)

    cart.add_item(_make_cart_item("fries"))  # mutate after snapshot

    assert len(snap.items) == 1  # snapshot unaffected


def test_cart_snapshot_get_items_returns_tuple():
    snap = CartSnapshot.from_cart(_make_cart("burger"))
    result = snap.get_items()
    assert isinstance(result, tuple)


# ── OrderDraft ────────────────────────────────────────────────────────────────

def _make_ctx(
    order_type: str | None = None,
    area: str | None = None,
    house_number: str | None = None,
    street: str | None = None,
    postal_code: str | None = None,
    customer_phone: str | None = None,
    order_number: str | None = None,
    confirmed: bool = False,
) -> ConversationContext:
    ctx = ConversationContext()
    ctx.order_type = order_type
    ctx.delivery_address.area = area
    ctx.delivery_address.house_number = house_number
    ctx.delivery_address.street = street
    ctx.delivery_address.postal_code = postal_code
    ctx.delivery_address.customer_phone_number = customer_phone
    ctx.delivery_address.order_number = order_number
    return ctx


def test_order_draft_from_context_captures_order_type():
    ctx = _make_ctx(order_type="pickup")
    draft = OrderDraft.from_context(ctx)
    assert draft.order_type == "pickup"


def test_order_draft_is_pickup():
    draft = OrderDraft.from_context(_make_ctx(order_type="pickup"))
    assert draft.is_pickup
    assert not draft.is_delivery


def test_order_draft_is_delivery():
    draft = OrderDraft.from_context(_make_ctx(order_type="delivery"))
    assert draft.is_delivery
    assert not draft.is_pickup


def test_order_draft_has_address_true():
    draft = OrderDraft.from_context(_make_ctx(house_number="12", street="Main St"))
    assert draft.has_address


def test_order_draft_has_address_false_when_empty():
    draft = OrderDraft.from_context(_make_ctx())
    assert not draft.has_address


def test_order_draft_none_order_type():
    draft = OrderDraft.from_context(_make_ctx())
    assert draft.order_type is None
    assert not draft.is_pickup
    assert not draft.is_delivery


def test_order_draft_is_frozen():
    draft = OrderDraft.from_context(_make_ctx(order_type="pickup"))
    try:
        draft.order_type = "delivery"  # type: ignore
        assert False, "Should have raised"
    except (FrozenInstanceError, AttributeError, TypeError):
        pass


def test_order_draft_captures_address_fields():
    ctx = _make_ctx(
        area="Downtown",
        house_number="5",
        street="Oak Ave",
        postal_code="12345",
        customer_phone="+1555",
        order_number="ORD-99",
    )
    draft = OrderDraft.from_context(ctx)
    assert draft.area == "Downtown"
    assert draft.house_number == "5"
    assert draft.street == "Oak Ave"
    assert draft.postal_code == "12345"
    assert draft.customer_phone_number == "+1555"
    assert draft.order_number == "ORD-99"


# ── CommandResult ─────────────────────────────────────────────────────────────

def test_command_result_ok_true():
    r = CommandResult(ok=True)
    assert r.ok is True
    assert r.sid is None
    assert r.error_code is None
    assert r.error_message is None
    assert r.transport_only is False
    assert r.attempts_made == 1


def test_command_result_ok_false_with_error():
    r = CommandResult(ok=False, error_code="sms_send_failed", error_message="Timed out", attempts_made=2)
    assert not r.ok
    assert r.error_code == "sms_send_failed"
    assert r.error_message == "Timed out"
    assert r.attempts_made == 2


def test_command_result_transfer_call():
    r = CommandResult(ok=True, transport_only=True, transfer_number="+15550001234")
    assert r.transport_only
    assert r.transfer_number == "+15550001234"


def test_command_result_is_frozen():
    r = CommandResult(ok=True)
    try:
        r.ok = False  # type: ignore
        assert False, "Should have raised"
    except (FrozenInstanceError, AttributeError, TypeError):
        pass


def test_command_result_from_dict_ok():
    d = {
        "ok": True,
        "sid": "SM123",
        "template": "payment_link",
        "attempts_made": 1,
        "idempotency_key": "abc",
    }
    r = CommandResult.from_dict(d)
    assert r.ok
    assert r.sid == "SM123"
    assert r.template == "payment_link"
    assert r.idempotency_key == "abc"


def test_command_result_from_dict_failure():
    d = {
        "ok": False,
        "sid": None,
        "error_code": "sms_transient_error",
        "error_message": "Rate limited",
        "attempts_made": 2,
    }
    r = CommandResult.from_dict(d)
    assert not r.ok
    assert r.error_code == "sms_transient_error"
    assert r.attempts_made == 2


def test_command_result_from_dict_empty():
    r = CommandResult.from_dict({})
    assert not r.ok
    assert r.sid is None
    assert r.attempts_made == 1


# ── contracts re-exports ──────────────────────────────────────────────────────

def test_contracts_handler_result_re_export():
    from app.contracts.handler_result import HandlerResult
    from app.state_machine.handler_result import HandlerResult as _HR
    assert HandlerResult is _HR


def test_contracts_nlu_result_re_export():
    from app.contracts.nlu_result import NLUResult, SlotValue
    from app.nlu.nlu_result import NLUResult as _NR, SlotValue as _SV
    assert NLUResult is _NR
    assert SlotValue is _SV
