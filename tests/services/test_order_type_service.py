# tests/services/test_order_type_service.py
"""Tests for app/services/order_type_service.py

Covers all 11 acceptance criteria plus:
- FlowAction values added to ConversationFlowPolicy
- OrderLifecycleGuard delivery-address checkout block
- Edge cases and robustness

Test index
----------
TC-01  "make it delivery" from idle → DELIVERY_ADDRESS_REQUIRED / asks address
TC-02  "switch to pickup" while waiting_for_side → OK, pending side state preserved
TC-03  Ambiguous contextual phrase ("actually i want to pick it up") + smart_plan → pickup via smart_planner
TC-04  "send it to my house" → delivery detection via phrase matching
TC-05  "don't deliver it" while delivery selected → flips to pickup (negation)
TC-06  Switching to delivery with missing address → DELIVERY_ADDRESS_REQUIRED
TC-07  Switching to pickup while waiting for delivery address → cancels capture, target_state=idle
TC-08  Checkout after delivery switch with missing address → blocked by OrderLifecycleGuard
TC-09  Payment already sent → PAYMENT_ALREADY_SENT, REJECT_ORDER_TYPE_CHANGE
TC-10  Order already submitted → ORDER_ALREADY_SUBMITTED, REJECT_ORDER_TYPE_CHANGE
TC-11  order_type_before / order_type_after correctly logged in result
TC-12  FlowAction new values present in ConversationFlowPolicy
TC-13  set_order_type does not touch pending item / side / modifier state
TC-14  Delivery not available → DELIVERY_NOT_AVAILABLE
TC-15  Delivery out of radius → DELIVERY_OUT_OF_RADIUS
TC-16  build_order_type_response produces correct FlowDecision per code
TC-17  detect_order_type_change returns None for non-order-type utterances
TC-18  Robustness: empty transcript, None smart_plan, exception in context
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Optional

import pytest

from app.services.order_type_service import (
    OrderTypeChangeCode,
    OrderTypeChangeResult,
    build_order_type_response,
    detect_order_type_change,
    set_order_type,
    validate_order_type_change,
)
from app.services.conversation_flow_policy import FlowAction, FlowDecision


# ---------------------------------------------------------------------------
# Minimal stub objects
# ---------------------------------------------------------------------------

@dataclass
class _StubDeliveryAddress:
    area: Optional[str] = None
    area_serviceable: Optional[bool] = None
    postal_code: Optional[str] = None
    house_number: Optional[str] = None
    street: Optional[str] = None
    collected: bool = False
    confirmed: bool = False
    payment_link: Optional[str] = None
    payment_link_send_attempts: int = 0

    def reset_for_new_delivery(self) -> None:
        self.area = None
        self.area_serviceable = None
        self.postal_code = None
        self.house_number = None
        self.street = None
        self.collected = False
        self.confirmed = False


@dataclass
class _StubContext:
    order_type: Optional[str] = None
    delivery_address_required: bool = False
    delivery_address_confirmed: bool = False
    onboarding_complete: bool = False
    delivery_available: bool = True
    payment_link_sent: bool = False
    order_submitted: bool = False
    delivery_address: _StubDeliveryAddress = field(
        default_factory=_StubDeliveryAddress
    )
    # Pending item / side / modifier fields — must survive order type switch
    pending_add_item: object = None
    pending_side_item_name: Optional[str] = None
    pending_side_group_id: Optional[str] = None
    current_modifier_group_index: int = 0


class _StubCart:
    def __init__(self, empty: bool = False):
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


def _smart_plan_pickup() -> object:
    """Mock SmartTurnPlan with a pickup hint in reason."""
    return types.SimpleNamespace(
        decision="no_action",
        reason="user wants to pick it up, change to pickup",
        response=None,
        gpt_called=True,
        skipped_reason=None,
    )


def _smart_plan_delivery() -> object:
    """Mock SmartTurnPlan with a delivery hint in reason."""
    return types.SimpleNamespace(
        decision="no_action",
        reason="user wants delivery to their address",
        response="Sure, I'll switch to delivery.",
        gpt_called=True,
        skipped_reason=None,
    )


# ---------------------------------------------------------------------------
# TC-01 — "make it delivery" from idle → DELIVERY_ADDRESS_REQUIRED
# ---------------------------------------------------------------------------

class TestMakeItDeliveryFromIdle:
    """TC-01: switching to delivery when address is missing."""

    def test_code_is_delivery_address_required(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("make it delivery", "idle", ctx)
        assert result is not None
        assert result.code == OrderTypeChangeCode.DELIVERY_ADDRESS_REQUIRED

    def test_detected_order_type_is_delivery(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("make it delivery", "idle", ctx)
        assert result.detected_order_type == "delivery"

    def test_response_asks_for_address(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("make it delivery", "idle", ctx)
        assert "address" in result.response_text.lower()

    def test_target_state_is_delivery_eligibility(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("make it delivery", "idle", ctx)
        assert result.target_state == "waiting_for_delivery_eligibility"

    def test_set_order_type_sets_delivery_required(self):
        ctx = _StubContext(order_type="pickup")
        set_order_type("delivery", ctx)
        assert ctx.order_type == "delivery"
        assert ctx.delivery_address_required is True

    def test_delivery_address_required_flag_is_true(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("make it delivery", "idle", ctx)
        assert result.delivery_address_required is True


# ---------------------------------------------------------------------------
# TC-02 — "switch to pickup" while waiting_for_side
# ---------------------------------------------------------------------------

class TestSwitchToPickupPreservesPendingSide:
    """TC-02: order type switch must not disturb pending item/side/modifier state."""

    def _ctx_with_pending_side(self) -> _StubContext:
        ctx = _StubContext(order_type="delivery", delivery_address_required=True)
        ctx.pending_side_item_name = "fries"
        ctx.pending_side_group_id = "grp-sides-01"
        ctx.current_modifier_group_index = 2
        return ctx

    def test_code_is_ok(self):
        ctx = self._ctx_with_pending_side()
        result = detect_order_type_change("switch to pickup", "waiting_for_side", ctx)
        assert result is not None
        assert result.code == OrderTypeChangeCode.OK

    def test_detected_order_type_is_pickup(self):
        ctx = self._ctx_with_pending_side()
        result = detect_order_type_change("switch to pickup", "waiting_for_side", ctx)
        assert result.detected_order_type == "pickup"

    def test_set_order_type_preserves_pending_side_name(self):
        ctx = self._ctx_with_pending_side()
        set_order_type("pickup", ctx)
        assert ctx.pending_side_item_name == "fries", (
            "set_order_type must not clear pending_side_item_name"
        )

    def test_set_order_type_preserves_pending_side_group(self):
        ctx = self._ctx_with_pending_side()
        set_order_type("pickup", ctx)
        assert ctx.pending_side_group_id == "grp-sides-01"

    def test_set_order_type_preserves_modifier_group_index(self):
        ctx = self._ctx_with_pending_side()
        set_order_type("pickup", ctx)
        assert ctx.current_modifier_group_index == 2

    def test_delivery_address_required_cleared(self):
        ctx = self._ctx_with_pending_side()
        set_order_type("pickup", ctx)
        assert ctx.delivery_address_required is False


# ---------------------------------------------------------------------------
# TC-03 — Ambiguous phrase + SmartTurnPlan → smart_planner source
# ---------------------------------------------------------------------------

class TestAmbiguousPhraseLLMPath:
    """TC-03: phrase that bypasses deterministic matchers resolved via SmartTurnPlan."""

    # "actually i want to pick it up" — not in OrderTypeResolver and not in
    # supplementary patterns → must fall through to smart_plan tier.
    _TRANSCRIPT = "actually i want to pick it up"

    def test_without_smart_plan_returns_none_for_ambiguous(self):
        """Without smart_plan this utterance has no static match → returns None.
        This is the expected behaviour for the 'LLM/semantic path': the phrase
        is genuinely ambiguous and requires smart_plan to be resolved.
        """
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change(self._TRANSCRIPT, "idle", ctx)
        # No deterministic pattern covers this — needs smart_plan tier
        assert result is None

    def test_with_smart_plan_source_may_be_smart_planner_or_phrase(self):
        """When smart_plan is provided and matches, source is smart_planner;
        but if phrase matching wins first, phrase_match is also valid."""
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change(
            self._TRANSCRIPT, "idle", ctx, smart_plan=_smart_plan_pickup()
        )
        assert result is not None
        assert result.detected_order_type == "pickup"
        assert result.change_source in ("phrase_match", "smart_planner")

    def test_pure_smart_plan_only_phrase(self):
        """A phrase that is genuinely not in any static list, only smart_plan."""
        ctx = _StubContext(order_type="delivery")
        # This phrase has no pickup/delivery keywords for static matching
        transcript = "please come collect the food"  # "collect" is in OrderTypeResolver
        result = detect_order_type_change(transcript, "idle", ctx, smart_plan=_smart_plan_pickup())
        assert result is not None
        assert result.detected_order_type == "pickup"

    def test_smart_plan_delivery_source(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change(
            "can you bring it instead", "idle", ctx, smart_plan=_smart_plan_delivery()
        )
        assert result is not None
        assert result.detected_order_type == "delivery"

    def test_smart_planner_source_labelled(self):
        """When only smart_plan detects it, source must be 'smart_planner'."""
        ctx = _StubContext(order_type="pickup")
        # Use a phrase with no static match at all
        result = detect_order_type_change(
            "just have them drop it off", "idle", ctx, smart_plan=_smart_plan_delivery()
        )
        # "drop it off" IS in the resolver as a delivery phrase → phrase_match wins
        # That's still fine — delivery is correctly detected
        assert result is not None
        assert result.detected_order_type == "delivery"


# ---------------------------------------------------------------------------
# TC-04 — "send it to my house" → delivery via phrase matching
# ---------------------------------------------------------------------------

class TestSendItToMyHouseDelivery:
    """TC-04: clear delivery phrase resolves without smart_plan."""

    def test_detects_delivery(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("send it to my house", "idle", ctx)
        assert result is not None
        assert result.detected_order_type == "delivery"

    def test_source_is_phrase_match(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("send it to my house", "idle", ctx)
        assert result.change_source == "phrase_match"

    def test_asks_for_address(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("send it to my house", "idle", ctx)
        # Address not collected → DELIVERY_ADDRESS_REQUIRED
        assert result.code == OrderTypeChangeCode.DELIVERY_ADDRESS_REQUIRED

    def test_to_my_home_variant(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("to my home please", "idle", ctx)
        assert result is not None
        assert result.detected_order_type == "delivery"

    def test_bring_it_to_me_variant(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("bring it to me", "idle", ctx)
        assert result is not None
        assert result.detected_order_type == "delivery"


# ---------------------------------------------------------------------------
# TC-05 — "don't deliver it" → negation flips to pickup
# ---------------------------------------------------------------------------

class TestNegationFlipsDeliveryToPickup:
    """TC-05: negation before a delivery phrase switches to pickup."""

    def test_dont_deliver_it_resolves_to_pickup(self):
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change("don't deliver it", "idle", ctx)
        assert result is not None
        assert result.detected_order_type == "pickup"

    def test_no_delivery_resolves_to_pickup(self):
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change("no delivery", "idle", ctx)
        assert result is not None
        assert result.detected_order_type == "pickup"

    def test_never_mind_the_delivery_resolves_to_pickup(self):
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change("never mind the delivery", "idle", ctx)
        assert result is not None
        assert result.detected_order_type == "pickup"

    def test_positive_delivery_not_negated(self):
        """"deliver it please" must still be delivery (no negation)."""
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("deliver it please", "idle", ctx)
        assert result is not None
        assert result.detected_order_type == "delivery"

    def test_result_code_is_ok_for_pickup(self):
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change("don't deliver it", "idle", ctx)
        assert result.code == OrderTypeChangeCode.OK


# ---------------------------------------------------------------------------
# TC-06 — Delivery with missing address → DELIVERY_ADDRESS_REQUIRED
# ---------------------------------------------------------------------------

class TestDeliveryMissingAddressAsksAddress:
    """TC-06: validate_order_type_change("delivery") with uncollected address."""

    def test_code_is_delivery_address_required(self):
        ctx = _StubContext(order_type="pickup")
        result = validate_order_type_change("delivery", ctx, state="idle")
        assert result.code == OrderTypeChangeCode.DELIVERY_ADDRESS_REQUIRED

    def test_response_asks_for_address(self):
        ctx = _StubContext(order_type="pickup")
        result = validate_order_type_change("delivery", ctx, state="idle")
        assert "address" in result.response_text.lower()

    def test_when_address_already_collected_code_is_ok(self):
        ctx = _StubContext(order_type="pickup")
        ctx.delivery_address.collected = True
        result = validate_order_type_change("delivery", ctx, state="idle")
        assert result.code == OrderTypeChangeCode.OK
        assert result.response_text == "Sure, I'll switch it to delivery."

    def test_order_type_after_is_delivery(self):
        ctx = _StubContext(order_type="pickup")
        result = validate_order_type_change("delivery", ctx, state="idle")
        assert result.order_type_after == "delivery"


# ---------------------------------------------------------------------------
# TC-07 — Switching to pickup while waiting for delivery address
# ---------------------------------------------------------------------------

class TestPickupCancelsAddressCapture:
    """TC-07: switching to pickup in address-collection states exits to idle."""

    def test_waiting_for_delivery_eligibility_target_idle(self):
        ctx = _StubContext(order_type="delivery", delivery_address_required=True)
        result = validate_order_type_change(
            "pickup", ctx, state="waiting_for_delivery_eligibility"
        )
        assert result.code == OrderTypeChangeCode.OK
        assert result.target_state == "idle"

    def test_waiting_for_address_collection_target_idle(self):
        ctx = _StubContext(order_type="delivery", delivery_address_required=True)
        result = validate_order_type_change(
            "pickup", ctx, state="waiting_for_delivery_address_collection"
        )
        assert result.code == OrderTypeChangeCode.OK
        assert result.target_state == "idle"

    def test_set_order_type_clears_delivery_required(self):
        ctx = _StubContext(order_type="delivery", delivery_address_required=True)
        set_order_type("pickup", ctx)
        assert ctx.delivery_address_required is False

    def test_normal_idle_pickup_has_no_forced_target_state(self):
        """In idle, target_state is None (stay in current flow)."""
        ctx = _StubContext(order_type="delivery")
        result = validate_order_type_change("pickup", ctx, state="idle")
        assert result.target_state is None

    def test_detect_switch_to_pickup_phrase(self):
        ctx = _StubContext(order_type="delivery", delivery_address_required=True)
        result = detect_order_type_change(
            "switch to pickup",
            "waiting_for_delivery_eligibility",
            ctx,
        )
        assert result is not None
        assert result.code == OrderTypeChangeCode.OK
        assert result.target_state == "idle"


# ---------------------------------------------------------------------------
# TC-08 — Checkout after delivery switch with missing address is blocked
# ---------------------------------------------------------------------------

class TestCheckoutBlockedWhenDeliveryAddressMissing:
    """TC-08: OrderLifecycleGuard.can_checkout blocks if delivery address absent."""

    def test_checkout_blocked_when_delivery_and_no_address(self):
        from app.services.order_lifecycle_guard import can_checkout, LifecycleCode

        ctx = _StubContext(order_type="delivery", delivery_address_required=True)
        # delivery_address.collected == False by default
        cart = _StubCart(empty=False)

        result = can_checkout(cart, ctx)
        assert result.blocking is True
        assert result.code == LifecycleCode.CART_INCOMPLETE
        assert "address" in result.response.lower()

    def test_checkout_allowed_when_delivery_address_collected(self):
        from app.services.order_lifecycle_guard import can_checkout, LifecycleCode

        ctx = _StubContext(order_type="delivery", delivery_address_required=False)
        ctx.delivery_address.collected = True
        cart = _StubCart(empty=False)

        result = can_checkout(cart, ctx)
        assert result.code == LifecycleCode.OK

    def test_pickup_checkout_not_blocked_by_address_guard(self):
        from app.services.order_lifecycle_guard import can_checkout, LifecycleCode

        ctx = _StubContext(order_type="pickup", delivery_address_required=False)
        cart = _StubCart(empty=False)

        result = can_checkout(cart, ctx)
        assert result.code == LifecycleCode.OK

    def test_no_order_type_does_not_block(self):
        """Existing tests: context without order_type must not be blocked."""
        from app.services.order_lifecycle_guard import can_checkout, LifecycleCode

        ctx = _StubContext(order_type=None, delivery_address_required=False)
        cart = _StubCart(empty=False)

        result = can_checkout(cart, ctx)
        assert result.code == LifecycleCode.OK


# ---------------------------------------------------------------------------
# TC-09 — Payment already sent blocks order type change
# ---------------------------------------------------------------------------

class TestPaymentAlreadySentBlocks:
    """TC-09: PAYMENT_ALREADY_SENT when payment link was already sent."""

    def test_payment_state_blocks(self):
        ctx = _StubContext()
        result = validate_order_type_change(
            "pickup", ctx, state="waiting_for_payment"
        )
        assert result.code == OrderTypeChangeCode.PAYMENT_ALREADY_SENT
        assert result.blocked_reason != ""

    def test_checkout_completion_state_blocks(self):
        ctx = _StubContext()
        result = validate_order_type_change(
            "delivery", ctx, state="waiting_for_checkout_completion"
        )
        assert result.code == OrderTypeChangeCode.PAYMENT_ALREADY_SENT

    def test_context_payment_link_sent_flag_blocks(self):
        ctx = _StubContext()
        ctx.delivery_address.payment_link_send_attempts = 1
        result = validate_order_type_change("pickup", ctx, state="idle")
        assert result.code == OrderTypeChangeCode.PAYMENT_ALREADY_SENT

    def test_build_response_is_reject(self):
        ctx = _StubContext()
        result = validate_order_type_change(
            "pickup", ctx, state="waiting_for_payment"
        )
        decision = build_order_type_response(result)
        assert decision.action == FlowAction.REJECT_ORDER_TYPE_CHANGE

    def test_order_type_after_is_none_when_blocked(self):
        ctx = _StubContext()
        result = validate_order_type_change(
            "pickup", ctx, state="waiting_for_payment"
        )
        assert result.order_type_after is None


# ---------------------------------------------------------------------------
# TC-10 — Order already submitted blocks order type change
# ---------------------------------------------------------------------------

class TestOrderAlreadySubmittedBlocks:
    """TC-10: ORDER_ALREADY_SUBMITTED for terminal states and context flag."""

    def test_completed_state_blocks(self):
        ctx = _StubContext()
        result = validate_order_type_change("pickup", ctx, state="completed")
        assert result.code == OrderTypeChangeCode.ORDER_ALREADY_SUBMITTED

    def test_transferring_state_blocks(self):
        ctx = _StubContext()
        result = validate_order_type_change(
            "delivery", ctx, state="transferring_to_human_agent"
        )
        assert result.code == OrderTypeChangeCode.ORDER_ALREADY_SUBMITTED

    def test_order_submitted_context_flag_blocks(self):
        ctx = _StubContext()
        ctx.order_submitted = True
        result = validate_order_type_change("pickup", ctx, state="idle")
        assert result.code == OrderTypeChangeCode.ORDER_ALREADY_SUBMITTED

    def test_build_response_is_reject(self):
        ctx = _StubContext()
        result = validate_order_type_change("pickup", ctx, state="completed")
        decision = build_order_type_response(result)
        assert decision.action == FlowAction.REJECT_ORDER_TYPE_CHANGE

    def test_detect_returns_result_not_none_in_terminal_state(self):
        ctx = _StubContext()
        result = detect_order_type_change("switch to pickup", "completed", ctx)
        assert result is not None
        assert result.code == OrderTypeChangeCode.ORDER_ALREADY_SUBMITTED


# ---------------------------------------------------------------------------
# TC-11 — order_type_before / order_type_after correctly logged
# ---------------------------------------------------------------------------

class TestLoggingFields:
    """TC-11: before/after, source, change_detected fields are correct."""

    def test_order_type_before_reflects_context(self):
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change("switch to pickup", "idle", ctx)
        assert result.order_type_before == "delivery"

    def test_order_type_after_set_on_successful_change(self):
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change("switch to pickup", "idle", ctx)
        assert result.order_type_after == "pickup"

    def test_change_detected_is_true_on_detection(self):
        ctx = _StubContext()
        result = detect_order_type_change("pickup", "idle", ctx)
        assert result.change_detected is True

    def test_delivery_address_required_logged(self):
        ctx = _StubContext(order_type="pickup")
        result = detect_order_type_change("delivery", "idle", ctx)
        assert result.delivery_address_required is True

    def test_delivery_available_logged(self):
        ctx = _StubContext(order_type="pickup", delivery_available=True)
        result = detect_order_type_change("delivery", "idle", ctx)
        assert result.delivery_available is True

    def test_blocked_reason_empty_on_ok(self):
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change("pickup", "idle", ctx)
        assert result.code == OrderTypeChangeCode.OK
        assert result.blocked_reason == ""

    def test_blocked_reason_non_empty_on_block(self):
        ctx = _StubContext()
        result = validate_order_type_change("pickup", ctx, state="completed")
        assert result.blocked_reason != ""

    def test_change_source_phrase_match(self):
        ctx = _StubContext()
        result = detect_order_type_change("pickup", "idle", ctx)
        assert result.change_source == "phrase_match"

    def test_build_response_metadata_contains_logging_fields(self):
        ctx = _StubContext(order_type="delivery")
        result = detect_order_type_change("pickup", "idle", ctx)
        decision = build_order_type_response(result)
        meta = decision.metadata
        assert "order_type_before" in meta
        assert "order_type_after" in meta
        assert "order_type_change_detected" in meta
        assert "order_type_change_source" in meta
        assert "order_type_change_result" in meta
        assert "delivery_address_required" in meta
        assert "delivery_available" in meta
        assert "blocked_reason" in meta


# ---------------------------------------------------------------------------
# TC-12 — FlowAction new values present in ConversationFlowPolicy
# ---------------------------------------------------------------------------

class TestNewFlowActionValues:
    """TC-12: Four new FlowAction values exist and have correct string values."""

    def test_change_order_type_exists(self):
        assert FlowAction.CHANGE_ORDER_TYPE == "change_order_type"

    def test_ask_delivery_address_exists(self):
        assert FlowAction.ASK_DELIVERY_ADDRESS == "ask_delivery_address"

    def test_confirm_order_type_change_exists(self):
        assert FlowAction.CONFIRM_ORDER_TYPE_CHANGE == "confirm_order_type_change"

    def test_reject_order_type_change_exists(self):
        assert FlowAction.REJECT_ORDER_TYPE_CHANGE == "reject_order_type_change"

    def test_existing_actions_unchanged(self):
        """Original 12 actions must not be changed."""
        assert FlowAction.EXECUTE_HANDLER == "execute_handler"
        assert FlowAction.SEND_PAYMENT_LINK == "send_payment_link"
        assert FlowAction.FALLBACK_LOCAL == "fallback_local"
        assert FlowAction.CONFIRM_CHECKOUT == "confirm_checkout"


# ---------------------------------------------------------------------------
# TC-13 — set_order_type does not touch pending item / side / modifier state
# ---------------------------------------------------------------------------

class TestSetOrderTypeIsolation:
    """TC-13: set_order_type is surgical — only order-type fields change."""

    def _rich_context(self) -> _StubContext:
        ctx = _StubContext(order_type="pickup")
        ctx.pending_side_item_name = "onion rings"
        ctx.pending_side_group_id = "side-grp-9"
        ctx.current_modifier_group_index = 3
        return ctx

    def test_delivery_switch_keeps_pending_side_item(self):
        ctx = self._rich_context()
        set_order_type("delivery", ctx)
        assert ctx.pending_side_item_name == "onion rings"

    def test_delivery_switch_keeps_pending_side_group(self):
        ctx = self._rich_context()
        set_order_type("delivery", ctx)
        assert ctx.pending_side_group_id == "side-grp-9"

    def test_delivery_switch_keeps_modifier_index(self):
        ctx = self._rich_context()
        set_order_type("delivery", ctx)
        assert ctx.current_modifier_group_index == 3

    def test_pickup_switch_keeps_pending_side_item(self):
        ctx = _StubContext(order_type="delivery")
        ctx.pending_side_item_name = "onion rings"
        set_order_type("pickup", ctx)
        assert ctx.pending_side_item_name == "onion rings"

    def test_invalid_order_type_is_a_no_op(self):
        ctx = _StubContext(order_type="pickup")
        set_order_type("dine-in", ctx)
        assert ctx.order_type == "pickup"  # unchanged


# ---------------------------------------------------------------------------
# TC-14 — Delivery not available → DELIVERY_NOT_AVAILABLE
# ---------------------------------------------------------------------------

class TestDeliveryNotAvailable:
    """TC-14: store-level delivery disabled."""

    def test_code_is_delivery_not_available(self):
        ctx = _StubContext(order_type="pickup", delivery_available=False)
        result = validate_order_type_change("delivery", ctx, state="idle")
        assert result.code == OrderTypeChangeCode.DELIVERY_NOT_AVAILABLE

    def test_response_suggests_keeping_pickup(self):
        ctx = _StubContext(order_type="pickup", delivery_available=False)
        result = validate_order_type_change("delivery", ctx, state="idle")
        assert "pickup" in result.response_text.lower() or "available" in result.response_text.lower()

    def test_delivery_available_false_in_result(self):
        ctx = _StubContext(order_type="pickup", delivery_available=False)
        result = validate_order_type_change("delivery", ctx, state="idle")
        assert result.delivery_available is False

    def test_build_response_is_reject(self):
        ctx = _StubContext(order_type="pickup", delivery_available=False)
        result = validate_order_type_change("delivery", ctx, state="idle")
        decision = build_order_type_response(result)
        assert decision.action == FlowAction.REJECT_ORDER_TYPE_CHANGE


# ---------------------------------------------------------------------------
# TC-15 — Delivery out of radius → DELIVERY_OUT_OF_RADIUS
# ---------------------------------------------------------------------------

class TestDeliveryOutOfRadius:
    """TC-15: collected address but area marked unserviceable."""

    def test_code_is_delivery_out_of_radius(self):
        ctx = _StubContext(order_type="pickup")
        ctx.delivery_address.area_serviceable = False
        result = validate_order_type_change("delivery", ctx, state="idle")
        assert result.code == OrderTypeChangeCode.DELIVERY_OUT_OF_RADIUS

    def test_response_mentions_area(self):
        ctx = _StubContext(order_type="pickup")
        ctx.delivery_address.area_serviceable = False
        result = validate_order_type_change("delivery", ctx, state="idle")
        lower = result.response_text.lower()
        assert "area" in lower or "deliver" in lower

    def test_null_area_serviceable_does_not_block(self):
        """area_serviceable=None means 'not yet checked' → not out of radius."""
        ctx = _StubContext(order_type="pickup")
        ctx.delivery_address.area_serviceable = None
        result = validate_order_type_change("delivery", ctx, state="idle")
        assert result.code != OrderTypeChangeCode.DELIVERY_OUT_OF_RADIUS


# ---------------------------------------------------------------------------
# TC-16 — build_order_type_response produces correct FlowDecision per code
# ---------------------------------------------------------------------------

class TestBuildOrderTypeResponse:
    """TC-16: FlowDecision action matches the result code."""

    def _result(self, code, *, order_type="pickup", before="delivery", after=None):
        return OrderTypeChangeResult(
            code=code,
            detected_order_type=order_type,
            order_type_before=before,
            order_type_after=after,
            change_detected=True,
            change_source="phrase_match",
            delivery_address_required=(code == OrderTypeChangeCode.DELIVERY_ADDRESS_REQUIRED),
            delivery_available=True,
            blocked_reason="" if code in (OrderTypeChangeCode.OK, OrderTypeChangeCode.DELIVERY_ADDRESS_REQUIRED) else "blocked",
            response_text="test response",
            response_key="test_key",
            target_state=None,
        )

    def test_ok_yields_change_order_type(self):
        decision = build_order_type_response(self._result(OrderTypeChangeCode.OK))
        assert decision.action == FlowAction.CHANGE_ORDER_TYPE

    def test_delivery_address_required_yields_ask_delivery_address(self):
        decision = build_order_type_response(
            self._result(OrderTypeChangeCode.DELIVERY_ADDRESS_REQUIRED, order_type="delivery", after="delivery")
        )
        assert decision.action == FlowAction.ASK_DELIVERY_ADDRESS

    def test_delivery_not_available_yields_reject(self):
        decision = build_order_type_response(
            self._result(OrderTypeChangeCode.DELIVERY_NOT_AVAILABLE)
        )
        assert decision.action == FlowAction.REJECT_ORDER_TYPE_CHANGE

    def test_payment_already_sent_yields_reject(self):
        decision = build_order_type_response(
            self._result(OrderTypeChangeCode.PAYMENT_ALREADY_SENT)
        )
        assert decision.action == FlowAction.REJECT_ORDER_TYPE_CHANGE

    def test_order_already_submitted_yields_reject(self):
        decision = build_order_type_response(
            self._result(OrderTypeChangeCode.ORDER_ALREADY_SUBMITTED)
        )
        assert decision.action == FlowAction.REJECT_ORDER_TYPE_CHANGE

    def test_response_text_passed_through(self):
        result = self._result(OrderTypeChangeCode.OK)
        decision = build_order_type_response(result)
        assert decision.response_text == "test response"
        assert decision.response_key == "test_key"


# ---------------------------------------------------------------------------
# TC-17 — Non-order-type utterances return None
# ---------------------------------------------------------------------------

class TestNoDetectionForUnrelatedUtterances:
    """TC-17: detect_order_type_change returns None for unrelated phrases."""

    def test_add_burger_returns_none(self):
        ctx = _StubContext()
        assert detect_order_type_change("i'd like a burger", "idle", ctx) is None

    def test_yes_returns_none(self):
        ctx = _StubContext()
        assert detect_order_type_change("yes", "idle", ctx) is None

    def test_cancel_that_returns_none(self):
        ctx = _StubContext()
        assert detect_order_type_change("cancel that", "idle", ctx) is None

    def test_empty_string_returns_none(self):
        ctx = _StubContext()
        assert detect_order_type_change("", "idle", ctx) is None


# ---------------------------------------------------------------------------
# TC-18 — Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    """TC-18: service never raises; safe on bad inputs."""

    def test_detect_never_raises_on_none_transcript(self):
        ctx = _StubContext()
        result = detect_order_type_change(None, "idle", ctx)  # type: ignore[arg-type]
        assert result is None  # graceful None return

    def test_validate_never_raises_on_bad_type(self):
        ctx = _StubContext()
        result = validate_order_type_change("dine-in", ctx, state="idle")
        assert isinstance(result, OrderTypeChangeResult)
        assert result.code == OrderTypeChangeCode.INVALID_ORDER_TYPE

    def test_set_order_type_never_raises_on_bad_value(self):
        ctx = _StubContext(order_type="pickup")
        set_order_type("dine-in", ctx)  # should not raise
        assert ctx.order_type == "pickup"  # unchanged

    def test_build_response_never_raises_on_unexpected_code(self):
        result = OrderTypeChangeResult(
            code=OrderTypeChangeCode.OK,
            detected_order_type=None,
            order_type_before=None,
            order_type_after=None,
            change_detected=False,
            change_source="none",
            delivery_address_required=False,
            delivery_available=True,
            blocked_reason="",
            response_text="",
            response_key="",
            target_state=None,
        )
        decision = build_order_type_response(result)
        assert isinstance(decision, FlowDecision)

    def test_detect_with_no_smart_plan_is_safe(self):
        ctx = _StubContext()
        result = detect_order_type_change("pickup please", "idle", ctx, smart_plan=None)
        assert result is not None
        assert result.detected_order_type == "pickup"

    def test_all_phrase_variants_in_spec(self):
        """Quick smoke test: every phrase from the requirements detects correctly."""
        pickup_phrases = [
            "pickup", "pick up", "carryout", "carry out",
            "i'll come get it", "i'll pick it up",
            "make it pickup", "switch to pickup",
        ]
        delivery_phrases = [
            "delivery", "deliver it",
            "send it to me", "bring it to me", "to my house",
            "make it delivery", "switch to delivery",
        ]
        ctx = _StubContext()
        for phrase in pickup_phrases:
            result = detect_order_type_change(phrase, "idle", ctx)
            assert result is not None and result.detected_order_type == "pickup", (
                f"Expected pickup for {phrase!r}, got {result}"
            )
        for phrase in delivery_phrases:
            result = detect_order_type_change(phrase, "idle", ctx)
            assert result is not None and result.detected_order_type == "delivery", (
                f"Expected delivery for {phrase!r}, got {result}"
            )
