# tests/services/test_order_lifecycle_guard.py
"""Focused lifecycle guard tests covering the full order lifecycle.

Test cases
----------
TC-L01  Unknown item "lasagna" → NOT_FOUND, no candidates → generic message, cart unchanged.
TC-L02  Unknown item with close candidates → NOT_FOUND with alternatives offered.
TC-L03  Unavailable item → ITEM_UNAVAILABLE, offers alternative, cart unchanged.
TC-L04  "that's it" with empty cart → CART_EMPTY, does not proceed to checkout.
TC-L05  "that's it" while required side pending → SIDE_REQUIRED, reprompts side.
TC-L06  "checkout" while required modifier pending → MODIFIER_REQUIRED, reprompts modifier.
TC-L07  Required size (variant) pending → SIZE_REQUIRED, reprompts size.
TC-L08  Valid cart, no pending item → OK, checkout proceeds.
TC-L09  Payment link — order_state is None → PAYMENT_NOT_READY, safe handoff response.
TC-L10  Payment link — order_state missing order_number → PAYMENT_NOT_READY.
TC-L11  Payment link — submit_ok=False → PAYMENT_NOT_READY.
TC-L12  Payment link — all signals positive → OK.
TC-L13  build_blocking_response passes through decision.response.
TC-L14  check_pending_requirements — no pending item → OK.
TC-L15  check_pending_requirements — optional modifier not satisfied → OK (does not block).
TC-L16  check_pending_requirements — required modifier satisfied → OK.
TC-L17  check_item_resolution — ITEM_UNAVAILABLE with no candidates.
TC-L18  can_checkout — pending item with all requirements met → CART_INCOMPLETE.
TC-L19  can_checkout — pending item with required side satisfied but modifier not → MODIFIER_REQUIRED.
TC-L20  format_candidate_list edge cases (0, 1, 2, 3 names).

Handler-level lifecycle integration
TC-H01  start_order_handler: "that's it" with empty cart → cart_empty response_key.
TC-H02  waiting_for_modifier: "what do you have?" → list_modifier_options response_key.
TC-H03  waiting_for_side: "what options do I have?" → list_side_options response_key.
TC-H04  item_not_found result → response_key is item_not_found or near_miss, cart unchanged.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from app.services.order_lifecycle_guard import (
    LifecycleCode,
    LifecycleDecision,
    _format_candidate_list,
    build_blocking_response,
    can_checkout,
    can_send_payment_link,
    check_item_resolution,
    check_pending_requirements,
)


# ---------------------------------------------------------------------------
# Minimal stubs for PendingAddItem, PendingModifierGroup, PendingSideGroup,
# ConversationContext, and Cart — tests do not import real models to stay fast.
# ---------------------------------------------------------------------------


def _make_modifier_group(
    group_id: str,
    group_name: str,
    *,
    is_required: bool = True,
    min_selector: int = 1,
    top_choices: list[str] | None = None,
) -> MagicMock:
    g = MagicMock()
    g.group_id = group_id
    g.name = group_name
    g.is_required = is_required
    g.min_selector = min_selector
    g.top_choice_names = top_choices or []
    return g


def _make_side_group(
    group_id: str,
    group_name: str,
    *,
    is_required: bool = True,
    min_selector: int = 1,
    is_suggested_addon: bool = False,
    top_choices: list[str] | None = None,
) -> MagicMock:
    g = MagicMock()
    g.group_id = group_id
    g.name = group_name
    g.is_required = is_required
    g.min_selector = min_selector
    g.is_suggested_addon = is_suggested_addon
    g.top_choice_names = top_choices or []
    return g


def _make_pending(
    item_name: str = "Chicken Burger",
    item_id: str = "item_001",
    modifier_groups: list | None = None,
    side_groups: list | None = None,
    item_variants: list | None = None,
    item_variant_names: tuple = (),
) -> MagicMock:
    p = MagicMock()
    p.item_id = item_id
    p.item_name = item_name
    p.modifier_groups = modifier_groups or []
    p.side_groups = side_groups or []
    p.item_variants = item_variants or []
    p.item_variant_names = item_variant_names
    return p


def _make_context(
    *,
    selected_modifier_groups: dict | None = None,
    skipped_modifier_groups: set | None = None,
    selected_side_groups: dict | None = None,
    skipped_side_groups: set | None = None,
    selected_variant_id: str | None = None,
    pending_add_item=None,
) -> MagicMock:
    ctx = MagicMock()
    ctx.selected_modifier_groups = selected_modifier_groups or {}
    ctx.skipped_modifier_groups = skipped_modifier_groups or set()
    ctx.selected_side_groups = selected_side_groups or {}
    ctx.skipped_side_groups = skipped_side_groups or set()
    ctx.selected_variant_id = selected_variant_id
    ctx.pending_add_item = pending_add_item
    return ctx


def _make_cart(*, empty: bool = False) -> MagicMock:
    cart = MagicMock()
    cart.is_empty.return_value = empty
    return cart


# ---------------------------------------------------------------------------
# TC-L01 – TC-L03: check_item_resolution
# ---------------------------------------------------------------------------


class TestCheckItemResolution(unittest.TestCase):

    def test_l01_no_candidates_returns_generic_not_found(self):
        """TC-L01: 'lasagna' with no candidates → generic not-found message."""
        decision = check_item_resolution("lasagna", [])
        self.assertEqual(decision.code, LifecycleCode.ITEM_NOT_FOUND)
        self.assertTrue(decision.blocking)
        self.assertIn("don't have that on the menu", decision.response)
        self.assertNotIn("lasagna", decision.response.lower().split("don't have")[0])

    def test_l02_with_candidates_offers_alternatives(self):
        """TC-L02: 'lasagna' with close candidates → alternatives in response."""
        decision = check_item_resolution(
            "lasagna", ["Spaghetti Bolognese", "Fettuccine Alfredo", "Penne Arrabbiata"]
        )
        self.assertEqual(decision.code, LifecycleCode.ITEM_NOT_FOUND)
        self.assertTrue(decision.blocking)
        self.assertIn("Spaghetti Bolognese", decision.response)
        self.assertIn("Fettuccine Alfredo", decision.response)
        # Should suggest up to 3
        self.assertIn("Penne Arrabbiata", decision.response)
        # Must not add to cart
        self.assertEqual(decision.details["candidates"][:3], [
            "Spaghetti Bolognese", "Fettuccine Alfredo", "Penne Arrabbiata"
        ])

    def test_l03_unavailable_item_with_alternative(self):
        """TC-L03: Unavailable item with alternative → ITEM_UNAVAILABLE response."""
        decision = check_item_resolution(
            "Clam Chowder", ["Lobster Bisque"], unavailable=True
        )
        self.assertEqual(decision.code, LifecycleCode.ITEM_UNAVAILABLE)
        self.assertTrue(decision.blocking)
        self.assertIn("isn't available right now", decision.response)
        self.assertIn("Lobster Bisque", decision.response)

    def test_unavailable_no_alternatives_uses_generic(self):
        """Unavailable item with no candidates → generic unavailable response."""
        decision = check_item_resolution("Clam Chowder", [], unavailable=True)
        self.assertEqual(decision.code, LifecycleCode.ITEM_UNAVAILABLE)
        self.assertIn("isn't available right now", decision.response)
        self.assertNotIn("We do have", decision.response)

    def test_empty_raw_text_handled_gracefully(self):
        """Empty raw_text does not raise."""
        decision = check_item_resolution("", [])
        self.assertEqual(decision.code, LifecycleCode.ITEM_NOT_FOUND)
        self.assertTrue(decision.blocking)

    def test_candidates_clamped_to_three(self):
        """Only up to 3 candidates appear in decision.details."""
        decision = check_item_resolution(
            "pizza", ["A", "B", "C", "D", "E"]
        )
        self.assertLessEqual(len(decision.details["candidates"]), 3)


# ---------------------------------------------------------------------------
# TC-L04 – TC-L08: can_checkout
# ---------------------------------------------------------------------------


class TestCanCheckout(unittest.TestCase):

    def test_l04_empty_cart_blocks_checkout(self):
        """TC-L04: Empty cart → CART_EMPTY, cannot checkout."""
        cart = _make_cart(empty=True)
        ctx = _make_context()
        decision = can_checkout(cart, ctx)
        self.assertEqual(decision.code, LifecycleCode.CART_EMPTY)
        self.assertTrue(decision.blocking)
        self.assertIn("empty", decision.response.lower())

    def test_l05_required_side_pending_blocks_checkout(self):
        """TC-L05: Required side not selected → SIDE_REQUIRED, reprompts side."""
        side_group = _make_side_group(
            "drinks", "Drink", is_required=True, min_selector=1,
            top_choices=["Coke", "Sprite", "Water"]
        )
        pending = _make_pending("Chicken Taco", side_groups=[side_group])
        # No sides selected
        ctx = _make_context(pending_add_item=pending)
        cart = _make_cart(empty=False)

        decision = can_checkout(cart, ctx)
        self.assertEqual(decision.code, LifecycleCode.SIDE_REQUIRED)
        self.assertTrue(decision.blocking)
        self.assertIn("Drink", decision.response)
        self.assertIn("Chicken Taco", decision.response)
        self.assertIn("Coke", decision.response)

    def test_l06_required_modifier_pending_blocks_checkout(self):
        """TC-L06: Required modifier not selected → MODIFIER_REQUIRED, reprompts."""
        mod_group = _make_modifier_group(
            "cheese", "Cheese", is_required=True, min_selector=1,
            top_choices=["Cheddar", "Mozzarella"]
        )
        pending = _make_pending("Chicken Burger", modifier_groups=[mod_group])
        ctx = _make_context(pending_add_item=pending)
        cart = _make_cart(empty=False)

        decision = can_checkout(cart, ctx)
        self.assertEqual(decision.code, LifecycleCode.MODIFIER_REQUIRED)
        self.assertTrue(decision.blocking)
        self.assertIn("Cheese", decision.response)
        self.assertIn("Chicken Burger", decision.response)
        self.assertIn("Cheddar", decision.response)

    def test_l07_size_required_blocks_checkout(self):
        """TC-L07: Item variants present, none selected → SIZE_REQUIRED."""
        pending = _make_pending(
            "Iced Coffee",
            item_variants=["small", "medium", "large"],
            item_variant_names=("Small", "Medium", "Large"),
        )
        ctx = _make_context(pending_add_item=pending, selected_variant_id=None)
        cart = _make_cart(empty=False)

        decision = can_checkout(cart, ctx)
        self.assertEqual(decision.code, LifecycleCode.SIZE_REQUIRED)
        self.assertTrue(decision.blocking)
        self.assertIn("Iced Coffee", decision.response)

    def test_l08_valid_cart_no_pending_allows_checkout(self):
        """TC-L08: Cart has items, no pending_add_item → OK."""
        cart = _make_cart(empty=False)
        ctx = _make_context(pending_add_item=None)

        decision = can_checkout(cart, ctx)
        self.assertEqual(decision.code, LifecycleCode.OK)
        self.assertFalse(decision.blocking)

    def test_l18_pending_all_requirements_met_is_cart_incomplete(self):
        """TC-L18: Pending item with all requirements met → CART_INCOMPLETE."""
        # No modifier/side groups, no variants → all requirements satisfied
        # but item not yet finalized → CART_INCOMPLETE
        pending = _make_pending("Simple Salad", modifier_groups=[], side_groups=[])
        pending.item_variants = []
        ctx = _make_context(pending_add_item=pending)
        cart = _make_cart(empty=False)

        decision = can_checkout(cart, ctx)
        self.assertEqual(decision.code, LifecycleCode.CART_INCOMPLETE)
        self.assertTrue(decision.blocking)

    def test_modifier_then_side_both_required_modifier_wins(self):
        """Required modifier checked before required side — modifier code returned."""
        mod_group = _make_modifier_group("m1", "Protein", is_required=True)
        side_group = _make_side_group("s1", "Side", is_required=True)
        pending = _make_pending(
            "Bowl", modifier_groups=[mod_group], side_groups=[side_group]
        )
        ctx = _make_context(pending_add_item=pending)
        cart = _make_cart(empty=False)

        decision = can_checkout(cart, ctx)
        self.assertEqual(decision.code, LifecycleCode.MODIFIER_REQUIRED)

    def test_optional_modifier_not_blocking(self):
        """Optional modifier not yet answered does not block checkout."""
        mod_group = _make_modifier_group("m1", "Extras", is_required=False)
        pending = _make_pending("Burger", modifier_groups=[mod_group])
        pending.item_variants = []
        ctx = _make_context(pending_add_item=pending)
        # pending_add_item with no required groups + no variants → CART_INCOMPLETE
        # (pending item exists at all — still incomplete)
        cart = _make_cart(empty=False)
        decision = can_checkout(cart, ctx)
        # should not be MODIFIER_REQUIRED for an optional group
        self.assertNotEqual(decision.code, LifecycleCode.MODIFIER_REQUIRED)

    def test_l19_required_side_met_but_modifier_not_returns_modifier(self):
        """TC-L19: Required side satisfied, required modifier not → MODIFIER_REQUIRED."""
        mod_group = _make_modifier_group("m1", "Sauce", is_required=True, min_selector=1)
        side_group = _make_side_group("s1", "Side", is_required=True, min_selector=1)
        pending = _make_pending(
            "Plate",
            modifier_groups=[mod_group],
            side_groups=[side_group],
        )
        # Side is satisfied, modifier is not
        ctx = _make_context(
            pending_add_item=pending,
            selected_side_groups={"s1": ["side_item_1"]},
        )
        cart = _make_cart(empty=False)

        decision = can_checkout(cart, ctx)
        self.assertEqual(decision.code, LifecycleCode.MODIFIER_REQUIRED)


# ---------------------------------------------------------------------------
# TC-L09 – TC-L12: can_send_payment_link
# ---------------------------------------------------------------------------


class TestCanSendPaymentLink(unittest.TestCase):

    def test_l09_none_order_state_is_not_ready(self):
        """TC-L09: None order_state → PAYMENT_NOT_READY, handoff response."""
        decision = can_send_payment_link(None)
        self.assertEqual(decision.code, LifecycleCode.PAYMENT_NOT_READY)
        self.assertTrue(decision.blocking)
        self.assertIn("connect you with someone", decision.response)

    def test_l10_missing_order_number_is_not_ready(self):
        """TC-L10: order_state without order_number → PAYMENT_NOT_READY."""
        decision = can_send_payment_link({"payment_ready": True, "submit_ok": True})
        self.assertEqual(decision.code, LifecycleCode.PAYMENT_NOT_READY)
        self.assertEqual(decision.details.get("reason"), "no_order_number")

    def test_l11_submit_ok_false_is_not_ready(self):
        """TC-L11: submit_ok=False → PAYMENT_NOT_READY."""
        decision = can_send_payment_link(
            {"order_number": "ORD-001", "submit_ok": False, "payment_ready": True}
        )
        self.assertEqual(decision.code, LifecycleCode.PAYMENT_NOT_READY)

    def test_payment_ready_false_is_not_ready(self):
        """payment_ready=False → PAYMENT_NOT_READY."""
        decision = can_send_payment_link(
            {"order_number": "ORD-001", "submit_ok": True, "payment_ready": False}
        )
        self.assertEqual(decision.code, LifecycleCode.PAYMENT_NOT_READY)

    def test_l12_all_signals_positive_is_ok(self):
        """TC-L12: All signals positive → OK."""
        decision = can_send_payment_link(
            {"order_number": "ORD-001", "submit_ok": True, "payment_ready": True}
        )
        self.assertEqual(decision.code, LifecycleCode.OK)
        self.assertFalse(decision.blocking)

    def test_attribute_style_order_state(self):
        """order_state can be an object with attributes (not just a dict)."""
        order = MagicMock()
        order.__getitem__ = MagicMock(side_effect=TypeError)
        order.order_number = "ORD-42"
        order.submit_ok = True
        order.payment_ready = True
        decision = can_send_payment_link(order)
        self.assertEqual(decision.code, LifecycleCode.OK)

    def test_empty_cart_at_payment_is_not_ready(self):
        """Passing an empty cart at payment time → PAYMENT_NOT_READY."""
        cart = _make_cart(empty=True)
        decision = can_send_payment_link(
            {"order_number": "ORD-001", "submit_ok": True, "payment_ready": True},
            cart=cart,
        )
        self.assertEqual(decision.code, LifecycleCode.PAYMENT_NOT_READY)
        self.assertEqual(decision.details.get("reason"), "cart_empty_at_payment")


# ---------------------------------------------------------------------------
# TC-L13: build_blocking_response
# ---------------------------------------------------------------------------


class TestBuildBlockingResponse(unittest.TestCase):

    def test_l13_returns_decision_response(self):
        """TC-L13: build_blocking_response returns decision.response."""
        decision = LifecycleDecision(
            code=LifecycleCode.CART_EMPTY,
            blocking=True,
            response="Your cart is empty. What would you like?",
        )
        self.assertEqual(
            build_blocking_response(decision),
            "Your cart is empty. What would you like?",
        )

    def test_ok_decision_returns_empty_string(self):
        """OK decision → build_blocking_response returns ''."""
        from app.services.order_lifecycle_guard import _OK
        self.assertEqual(build_blocking_response(_OK), "")


# ---------------------------------------------------------------------------
# TC-L14 – TC-L17: check_pending_requirements
# ---------------------------------------------------------------------------


class TestCheckPendingRequirements(unittest.TestCase):

    def test_l14_none_pending_returns_ok(self):
        """TC-L14: No pending item → OK."""
        ctx = _make_context()
        decision = check_pending_requirements(None, ctx)
        self.assertEqual(decision.code, LifecycleCode.OK)
        self.assertFalse(decision.blocking)

    def test_l15_optional_modifier_not_satisfied_does_not_block(self):
        """TC-L15: Optional modifier not answered → does not block."""
        mod_group = _make_modifier_group("m1", "Extras", is_required=False)
        pending = _make_pending(modifier_groups=[mod_group])
        ctx = _make_context()  # nothing selected, nothing skipped

        decision = check_pending_requirements(pending, ctx)
        self.assertEqual(decision.code, LifecycleCode.OK)

    def test_l16_required_modifier_satisfied_returns_ok(self):
        """TC-L16: Required modifier satisfied → OK."""
        mod_group = _make_modifier_group("m1", "Cheese", is_required=True, min_selector=1)
        pending = _make_pending(modifier_groups=[mod_group])
        mock_selection = MagicMock()
        ctx = _make_context(
            selected_modifier_groups={"m1": [mock_selection]},
        )

        decision = check_pending_requirements(pending, ctx)
        self.assertEqual(decision.code, LifecycleCode.OK)

    def test_required_modifier_skipped_still_blocks(self):
        """Skipped flag does NOT satisfy a required modifier group (mirrors add_item_flow logic).

        Required groups are only satisfied when selected_count >= min_selector.
        The skipped flag applies only to optional groups.  If a required group
        ends up in skipped_modifier_groups the guard still blocks — the handler
        layer should never allow a required group to be skipped.
        """
        mod_group = _make_modifier_group("m1", "Extras", is_required=True)
        pending = _make_pending(modifier_groups=[mod_group])
        ctx = _make_context(skipped_modifier_groups={"m1"})

        decision = check_pending_requirements(pending, ctx)
        # Required group with 0 selections → still blocking, mirrors add_item_flow
        self.assertEqual(decision.code, LifecycleCode.MODIFIER_REQUIRED)
        self.assertTrue(decision.blocking)

    def test_required_side_not_satisfied_returns_side_required(self):
        """Unresolved required side → SIDE_REQUIRED with item and group in response."""
        side_group = _make_side_group(
            "s1", "Fries", is_required=True, min_selector=1,
            top_choices=["Regular Fries", "Sweet Potato Fries"]
        )
        pending = _make_pending("Burger", side_groups=[side_group])
        ctx = _make_context()

        decision = check_pending_requirements(pending, ctx)
        self.assertEqual(decision.code, LifecycleCode.SIDE_REQUIRED)
        self.assertTrue(decision.blocking)
        self.assertIn("Fries", decision.response)
        self.assertIn("Burger", decision.response)
        self.assertEqual(decision.details["group_id"], "s1")

    def test_suggested_addon_side_does_not_block(self):
        """Suggested-addon side groups never block checkout."""
        side_group = _make_side_group(
            "s1", "Add-ons", is_required=True, is_suggested_addon=True
        )
        pending = _make_pending(side_groups=[side_group])
        ctx = _make_context()

        decision = check_pending_requirements(pending, ctx)
        self.assertEqual(decision.code, LifecycleCode.OK)

    def test_l17_size_required_when_variants_not_selected(self):
        """TC-L17: Item has variants but none selected → SIZE_REQUIRED."""
        pending = _make_pending(
            "Coffee",
            item_variants=["sm", "md", "lg"],
            item_variant_names=("Small", "Medium", "Large"),
        )
        ctx = _make_context(selected_variant_id=None)

        decision = check_pending_requirements(pending, ctx)
        self.assertEqual(decision.code, LifecycleCode.SIZE_REQUIRED)
        self.assertTrue(decision.blocking)
        self.assertIn("Coffee", decision.response)
        self.assertIn("Small", decision.response)

    def test_size_required_only_when_variant_selected(self):
        """When variant is already selected, SIZE_REQUIRED is not returned."""
        pending = _make_pending(
            "Coffee",
            item_variants=["sm", "md", "lg"],
        )
        ctx = _make_context(selected_variant_id="sm")

        # modifier/side groups empty → should be OK
        decision = check_pending_requirements(pending, ctx)
        self.assertEqual(decision.code, LifecycleCode.OK)


# ---------------------------------------------------------------------------
# TC-L20: _format_candidate_list
# ---------------------------------------------------------------------------


class TestFormatCandidateList(unittest.TestCase):

    def test_empty_list_returns_empty_string(self):
        self.assertEqual(_format_candidate_list([]), "")

    def test_single_item(self):
        self.assertEqual(_format_candidate_list(["Coke"]), "Coke")

    def test_two_items_uses_or(self):
        self.assertEqual(_format_candidate_list(["Coke", "Sprite"]), "Coke or Sprite")

    def test_three_items_uses_serial_or(self):
        result = _format_candidate_list(["Coke", "Sprite", "Water"])
        self.assertEqual(result, "Coke, Sprite, or Water")

    def test_more_than_three_clamped(self):
        # Even if 4 are passed, only first 3 are used
        result = _format_candidate_list(["A", "B", "C", "D"])
        self.assertEqual(result, "A, B, or C")

    def test_filters_empty_strings(self):
        result = _format_candidate_list(["A", "", "B"])
        self.assertEqual(result, "A or B")


# ---------------------------------------------------------------------------
# Handler-level lifecycle integration tests
# ---------------------------------------------------------------------------


class TestHandlerLevelLifecycle(unittest.TestCase):
    """Integration tests that call real handlers with minimal mocks.

    TC-H01: start_order_handler: empty cart → cart_empty
    TC-H02: waiting_for_modifier: "what do you have?" → list_modifier_options
    TC-H03: waiting_for_side: "what options do I have?" → list_side_options
    TC-H04: item resolution failure → item_not_found response_key, cart unchanged
    """

    def test_h01_empty_cart_checkout_blocked(self):
        """TC-H01: start_order_handler returns cart_empty when cart is empty."""
        from app.state_machine.handlers.order.start_order_handler import StartOrderHandler
        from app.nlu.intent_resolution.intent import Intent
        from app.state_machine.models.conversation_context import ConversationContext

        cart_summary_builder = MagicMock()
        handler = StartOrderHandler(cart_summary_builder=cart_summary_builder)

        session = MagicMock()
        session.cart.is_empty.return_value = True

        ctx = ConversationContext()
        result = handler.handle(
            intent=Intent.END_ADDING,
            context=ctx,
            user_text="that's it",
            session=session,
        )
        self.assertEqual(result.response_key, "cart_empty")
        # cart_summary_builder must NOT be called (no summary when empty)
        cart_summary_builder.build.assert_not_called()

    def test_h01_non_empty_cart_proceeds_to_confirming_order(self):
        """Non-empty cart → confirm_order_summary, CONFIRMING_ORDER state."""
        from app.state_machine.handlers.order.start_order_handler import StartOrderHandler
        from app.nlu.intent_resolution.intent import Intent
        from app.state_machine.models.conversation_context import ConversationContext
        from app.state_machine.models.conversation_state import ConversationState

        cart_summary_builder = MagicMock()
        cart_summary_builder.build.return_value = {"items": ["Burger x1"]}
        handler = StartOrderHandler(cart_summary_builder=cart_summary_builder)

        session = MagicMock()
        session.cart.is_empty.return_value = False

        ctx = ConversationContext()
        result = handler.handle(
            intent=Intent.CHECKOUT,
            context=ctx,
            user_text="checkout",
            session=session,
        )
        self.assertEqual(result.response_key, "confirm_order_summary")
        self.assertEqual(result.next_state, ConversationState.CONFIRMING_ORDER)

    def test_h02_what_do_you_have_in_waiting_for_modifier(self):
        """TC-H02: 'what do you have?' in waiting_for_modifier → list_modifier_options."""
        from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
            WaitingForModifierHandler,
        )
        from app.nlu.intent_resolution.intent import Intent
        from app.state_machine.models.conversation_context import ConversationContext
        from app.state_machine.models.pending_item_models import (
            PendingAddItem,
            PendingModifierGroup,
            PendingModifierChoice,
        )

        # Build real pending item with a modifier group
        choice = PendingModifierChoice(
            modifier_id="mod_ched",
            name="Cheddar",
            group_id="cheese",
            normalized_name="cheddar",
            match_texts=("cheddar",),
        )
        group = PendingModifierGroup(
            group_id="cheese",
            name="Cheese",
            is_required=True,
            min_selector=1,
            max_selector=1,
            choices=[choice],
            choices_by_modifier_id={"mod_ched": choice},
            choice_names=("Cheddar",),
        )
        pending = PendingAddItem(
            item_id="burger_01",
            item_name="Chicken Burger",
            modifier_groups=[group],
            modifier_groups_by_id={"cheese": group},
        )

        ctx = ConversationContext()
        ctx.pending_add_item = pending
        ctx.current_modifier_group_index = 0
        ctx.last_intent_confidence = 0.9  # high confidence — planner won't fire

        handler = WaitingForModifierHandler(menu_repo=None)
        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="what do you have?",
            session=None,
        )
        # "what do you have?" maps to OPTIONS_REQUEST control intent → lists options
        self.assertIn(result.response_key, {
            "list_modifier_options",
            "repeat_modifier_options",
        })

    def test_h03_what_options_in_waiting_for_side(self):
        """TC-H03: Options question in waiting_for_side → list_side_options."""
        from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
            WaitingForSideHandler,
        )
        from app.nlu.intent_resolution.intent import Intent
        from app.state_machine.models.conversation_context import ConversationContext
        from app.state_machine.models.pending_item_models import (
            PendingAddItem,
            PendingSideGroup,
            PendingSideChoice,
        )

        choice = PendingSideChoice(
            item_id="side_fries",
            name="Regular Fries",
            pricing_mode="flat",
            normalized_name="regular fries",
            match_texts=("regular fries",),
        )
        group = PendingSideGroup(
            group_id="sides",
            name="Sides",
            is_required=True,
            min_selector=1,
            max_selector=1,
            choices=[choice],
            choices_by_item_id={"side_fries": choice},
            choice_names=("Regular Fries",),
        )
        pending = PendingAddItem(
            item_id="taco_01",
            item_name="Chicken Taco",
            side_groups=[group],
            side_groups_by_id={"sides": group},
        )

        ctx = ConversationContext()
        ctx.pending_add_item = pending
        ctx.current_side_group_index = 0
        ctx.last_intent_confidence = 0.9

        handler = WaitingForSideHandler(menu_repo=None)
        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="what options do I have?",
            session=None,
        )
        self.assertIn(result.response_key, {
            "list_side_options",
            "repeat_side_options",
            "ask_for_side",
        })

    def test_h04_item_not_found_leaves_cart_unchanged(self):
        """TC-H04: item_not_found path sets response_key, never mutates cart."""
        from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
        from app.nlu.intent_resolution.intent import Intent
        from app.state_machine.models.conversation_context import ConversationContext

        menu_repo = MagicMock()
        query_result = MagicMock()
        # No items → resolver returns empty
        query_result.items = []
        query_result.item = None
        query_result.type = None
        query_result.query = "lasagna"
        menu_repo.resolve_menu_query_normalized.return_value = query_result

        handler = AddItemHandler(menu_repo=menu_repo, gpt_planner=None)
        ctx = ConversationContext()
        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=ctx,
            user_text="I'd like a lasagna",
            session=None,
        )
        # Response key must be a not-found variant
        self.assertIn(result.response_key, {
            "item_not_found",
            "item_not_found_escalation",
            "item_not_found_near_miss",
            "confirm_item_ambiguous",
            "confirm_item_from_category",
        })
        # context.pending_add_item must NOT be set
        self.assertIsNone(ctx.pending_add_item)


# ---------------------------------------------------------------------------
# Correction/cancel guard policy tests
# (verify existing cancel behavior and lifecycle guard aids correction logic)
# ---------------------------------------------------------------------------


class TestCorrectionCancelPolicy(unittest.TestCase):
    """TC-H05, TC-H06: Cancel behavior and last_cart_diff guard."""

    def test_cancel_order_in_idle_returns_no_active_order(self):
        """TC-H05: 'cancel the order' from IDLE → no_active_order_to_cancel."""
        from app.state_machine.handlers.order.start_order_handler import StartOrderHandler
        from app.nlu.intent_resolution.intent import Intent
        from app.state_machine.models.conversation_context import ConversationContext

        handler = StartOrderHandler(cart_summary_builder=MagicMock())
        session = MagicMock()
        session.cart.is_empty.return_value = True

        ctx = ConversationContext()
        result = handler.handle(
            intent=Intent.CANCEL_ORDER,
            context=ctx,
            user_text="cancel the order",
            session=session,
        )
        self.assertEqual(result.response_key, "no_active_order_to_cancel")

    def test_lifecycle_guard_check_item_resolution_never_adds_to_cart(self):
        """TC-H06: Lifecycle guard decisions never contain a cart mutation signal."""
        decision = check_item_resolution("lasagna", ["Spaghetti"])
        # No "add_to_cart" key in details
        self.assertNotIn("add_to_cart", decision.details)
        self.assertNotIn("item_id", decision.details)
        # The response never says "added"
        self.assertNotIn("added", decision.response.lower())

    def test_last_cart_diff_semantics_in_check_item_resolution(self):
        """check_item_resolution with candidates → safe for correction context."""
        decision = check_item_resolution(
            "wrong item", ["Right Item"]
        )
        # candidates should be in details for use by correction handler
        self.assertIn("candidates", decision.details)
        self.assertEqual(decision.details["candidates"], ["Right Item"])


if __name__ == "__main__":
    unittest.main()
