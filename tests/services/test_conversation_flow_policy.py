# tests/services/test_conversation_flow_policy.py
"""15 focused tests for ConversationFlowPolicy.

Tests cover:
  TC-01  Off-menu item with alternatives      → SUGGEST_ALTERNATIVES, cart unchanged
  TC-02  Off-menu item without alternatives   → SUGGEST_ALTERNATIVES, "not on menu"
  TC-03  "that's it" empty cart               → ASK_CLARIFICATION (no payment link)
  TC-04  "that's it" pending side             → ASK_MISSING_REQUIREMENT (no link)
  TC-05  "that's it" complete cart            → CONFIRM_CHECKOUT (not SEND_PAYMENT_LINK)
  TC-06  "yeah do it" in confirming_order     → SEND_PAYMENT_LINK
  TC-07  "yeah do it" in idle, no context     → ASK_CLARIFICATION
  TC-08  "no coke" when Coke in cart          → REMOVE_SPECIFIC_ITEM
  TC-09  "no coke" when Coke was suggested    → ASK_CLARIFICATION (reject suggestion)
  TC-10  "cancel that" after last item added  → REMOVE_LAST_ITEM
  TC-11  "cancel the order"                   → CLEAR_ORDER_CONFIRM (confirmation required)
  TC-12  "okay fine" after payment-link ask   → SEND_PAYMENT_LINK
  TC-13  "okay fine" in waiting_for_side      → EXECUTE_HANDLER
  TC-14  "no" in confirming_order             → FALLBACK_LOCAL (no payment link sent)
  TC-15  "no" in idle                         → ASK_CLARIFICATION
"""
from __future__ import annotations

import pytest

from app.services.conversation_flow_policy import (
    FlowAction,
    FlowDecision,
    build_checkout_confirmation,
    decide_off_menu_flow,
    decide_short_utterance_flow,
    build_reprompt_for_lifecycle_decision,
    _is_affirm,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _lc_ok():
    """Return a non-blocking LifecycleDecision (cart cleared for checkout)."""
    from app.services.order_lifecycle_guard import LifecycleCode, LifecycleDecision
    return LifecycleDecision(
        code=LifecycleCode.OK,
        blocking=False,
        response="",
        details={},
    )


def _lc_side_required(response: str = "Please choose a side for your burger."):
    """Return a blocking LifecycleDecision for SIDE_REQUIRED."""
    from app.services.order_lifecycle_guard import LifecycleCode, LifecycleDecision
    return LifecycleDecision(
        code=LifecycleCode.SIDE_REQUIRED,
        blocking=True,
        response=response,
        details={"item": "burger"},
    )


# ---------------------------------------------------------------------------
# TC-01 — Off-menu item WITH alternatives
# ---------------------------------------------------------------------------

class TestOffMenuWithAlternatives:
    """TC-01: off-menu item + alternatives → SUGGEST_ALTERNATIVES, cart unchanged."""

    def test_action_is_suggest_alternatives(self):
        d = decide_off_menu_flow(
            "lobster roll",
            alternatives=["crab sandwich", "fish tacos"],
        )
        assert d.action == FlowAction.SUGGEST_ALTERNATIVES

    def test_response_contains_requested_item(self):
        d = decide_off_menu_flow(
            "lobster roll",
            alternatives=["crab sandwich", "fish tacos"],
        )
        assert "lobster roll" in d.response_text

    def test_response_mentions_at_least_one_alternative(self):
        d = decide_off_menu_flow(
            "lobster roll",
            alternatives=["crab sandwich", "fish tacos"],
        )
        assert "crab sandwich" in d.response_text or "fish tacos" in d.response_text

    def test_metadata_cart_unchanged_is_true(self):
        d = decide_off_menu_flow(
            "lobster roll",
            alternatives=["crab sandwich", "fish tacos"],
        )
        assert d.metadata.get("cart_unchanged") is True

    def test_metadata_alternatives_present(self):
        d = decide_off_menu_flow(
            "lobster roll",
            alternatives=["crab sandwich", "fish tacos"],
        )
        alts = d.metadata.get("alternatives", [])
        assert len(alts) >= 1


# ---------------------------------------------------------------------------
# TC-02 — Off-menu item WITHOUT alternatives
# ---------------------------------------------------------------------------

class TestOffMenuNoAlternatives:
    """TC-02: off-menu item, no alternatives → SUGGEST_ALTERNATIVES, 'not on menu'."""

    def test_action_is_suggest_alternatives(self):
        d = decide_off_menu_flow("unicorn burger")
        assert d.action == FlowAction.SUGGEST_ALTERNATIVES

    def test_response_says_not_available(self):
        d = decide_off_menu_flow("unicorn burger")
        lower = d.response_text.lower()
        assert "don't have" in lower or "not on the menu" in lower or "not available" in lower

    def test_metadata_cart_unchanged_is_true(self):
        d = decide_off_menu_flow("unicorn burger")
        assert d.metadata.get("cart_unchanged") is True

    def test_metadata_alternatives_empty(self):
        d = decide_off_menu_flow("unicorn burger")
        assert d.metadata.get("alternatives", []) == []


# ---------------------------------------------------------------------------
# TC-03 — "that's it" with empty cart
# ---------------------------------------------------------------------------

class TestCheckoutPhraseEmptyCart:
    """TC-03: "that's it" with empty cart → asks what to order, no payment link."""

    def test_action_is_not_send_payment_link(self):
        d = decide_short_utterance_flow("that's it", "idle", cart_snapshot=[])
        assert d.action != FlowAction.SEND_PAYMENT_LINK

    def test_action_is_clarification(self):
        d = decide_short_utterance_flow("that's it", "idle", cart_snapshot=[])
        assert d.action == FlowAction.ASK_CLARIFICATION

    def test_response_mentions_empty_or_what_to_order(self):
        d = decide_short_utterance_flow("that's it", "idle", cart_snapshot=[])
        lower = d.response_text.lower()
        assert "empty" in lower or "what" in lower or "order" in lower


# ---------------------------------------------------------------------------
# TC-04 — "that's it" with pending side requirement
# ---------------------------------------------------------------------------

class TestCheckoutPhrasePendingSide:
    """TC-04: checkout phrase when side not yet chosen → reprompts for side, no link."""

    def test_action_is_ask_missing_requirement(self):
        d = decide_short_utterance_flow(
            "that's it", "idle",
            cart_snapshot=["Burger"],
            lifecycle_decision=_lc_side_required(),
        )
        assert d.action == FlowAction.ASK_MISSING_REQUIREMENT

    def test_action_is_not_send_payment_link(self):
        d = decide_short_utterance_flow(
            "that's it", "idle",
            cart_snapshot=["Burger"],
            lifecycle_decision=_lc_side_required(),
        )
        assert d.action != FlowAction.SEND_PAYMENT_LINK

    def test_response_text_comes_from_lifecycle_decision(self):
        lc = _lc_side_required(response="Please choose a side.")
        d = decide_short_utterance_flow(
            "that's it", "idle",
            cart_snapshot=["Burger"],
            lifecycle_decision=lc,
        )
        assert "side" in d.response_text.lower()


# ---------------------------------------------------------------------------
# TC-05 — "that's it" with complete cart
# ---------------------------------------------------------------------------

class TestCheckoutPhraseCompleteCart:
    """TC-05: checkout phrase, cart complete → CONFIRM_CHECKOUT summary (not send link yet)."""

    def test_action_is_confirm_checkout(self):
        d = decide_short_utterance_flow(
            "that's it", "idle",
            cart_snapshot=["Burger", "Fries"],
            lifecycle_decision=_lc_ok(),
        )
        assert d.action == FlowAction.CONFIRM_CHECKOUT

    def test_action_is_not_send_payment_link(self):
        d = decide_short_utterance_flow(
            "that's it", "idle",
            cart_snapshot=["Burger", "Fries"],
            lifecycle_decision=_lc_ok(),
        )
        assert d.action != FlowAction.SEND_PAYMENT_LINK

    def test_response_contains_cart_items(self):
        d = decide_short_utterance_flow(
            "that's it", "idle",
            cart_snapshot=["Burger", "Fries"],
            lifecycle_decision=_lc_ok(),
        )
        assert "Burger" in d.response_text or "Fries" in d.response_text

    def test_response_asks_for_confirmation_not_sends_link(self):
        d = decide_short_utterance_flow(
            "that's it", "idle",
            cart_snapshot=["Burger", "Fries"],
            lifecycle_decision=_lc_ok(),
        )
        lower = d.response_text.lower()
        assert "should i send" in lower or "payment link" in lower

    def test_requires_confirmation_true(self):
        d = decide_short_utterance_flow(
            "that's it", "idle",
            cart_snapshot=["Burger", "Fries"],
            lifecycle_decision=_lc_ok(),
        )
        assert d.requires_confirmation is True


# ---------------------------------------------------------------------------
# TC-06 — "yeah do it" in confirming_order → SEND_PAYMENT_LINK
# ---------------------------------------------------------------------------

class TestYeahDoItInConfirmingOrder:
    """TC-06: compound affirm during order confirmation → payment link."""

    def test_action_is_send_payment_link(self):
        d = decide_short_utterance_flow("yeah do it", "confirming_order")
        assert d.action == FlowAction.SEND_PAYMENT_LINK

    def test_response_key_correct(self):
        d = decide_short_utterance_flow("yeah do it", "confirming_order")
        assert d.response_key == "send_payment_link"

    def test_yeah_alone_also_sends_payment_link(self):
        """Single "yeah" in confirming_order must also trigger link."""
        d = decide_short_utterance_flow("yeah", "confirming_order")
        assert d.action == FlowAction.SEND_PAYMENT_LINK


# ---------------------------------------------------------------------------
# TC-07 — "yeah do it" in idle, no pending action → ASK_CLARIFICATION
# ---------------------------------------------------------------------------

class TestYeahDoItIdleNoContext:
    """TC-07: compound affirm in idle with no context → asks clarification."""

    def test_action_is_ask_clarification(self):
        d = decide_short_utterance_flow("yeah do it", "idle")
        assert d.action == FlowAction.ASK_CLARIFICATION

    def test_action_is_not_send_payment_link(self):
        d = decide_short_utterance_flow("yeah do it", "idle")
        assert d.action != FlowAction.SEND_PAYMENT_LINK

    def test_reason_is_affirm_no_pending_context(self):
        d = decide_short_utterance_flow("yeah do it", "idle")
        assert d.reason == "affirm_no_pending_context"


# ---------------------------------------------------------------------------
# TC-08 — "no coke" when Coke in cart → REMOVE_SPECIFIC_ITEM
# ---------------------------------------------------------------------------

class TestNoItemInCart:
    """TC-08: "no [item]" where item is in cart → remove specific item."""

    def test_action_is_remove_specific_item(self):
        d = decide_short_utterance_flow("no coke", "idle", cart_snapshot=["Coke"])
        assert d.action == FlowAction.REMOVE_SPECIFIC_ITEM

    def test_tool_name_is_remove_item_by_name(self):
        d = decide_short_utterance_flow("no coke", "idle", cart_snapshot=["Coke"])
        assert d.tool_name == "remove_item_by_name"

    def test_metadata_contains_item_name(self):
        d = decide_short_utterance_flow("no coke", "idle", cart_snapshot=["Coke"])
        item = d.metadata.get("item_name", "")
        assert "coke" in item.lower() or "Coke" in item

    def test_response_confirms_removal(self):
        d = decide_short_utterance_flow("no coke", "idle", cart_snapshot=["Coke"])
        assert "removed" in d.response_text.lower() or "got it" in d.response_text.lower()

    def test_case_insensitive_match(self):
        """Cart item stored as "Diet Coke"; user says "no diet coke"."""
        d = decide_short_utterance_flow(
            "no diet coke", "idle", cart_snapshot=["Diet Coke"]
        )
        assert d.action == FlowAction.REMOVE_SPECIFIC_ITEM


# ---------------------------------------------------------------------------
# TC-09 — "no coke" when Coke was suggested (not in cart)
# ---------------------------------------------------------------------------

class TestNoItemWhenSuggested:
    """TC-09: "no coke" when Coke was suggested, not in cart → reject suggestion."""

    def test_action_is_ask_clarification(self):
        d = decide_short_utterance_flow(
            "no coke", "idle",
            cart_snapshot=[],
            previous_assistant_text="Would you like a Coke with that?",
        )
        assert d.action == FlowAction.ASK_CLARIFICATION

    def test_response_key_is_ask_alternative(self):
        d = decide_short_utterance_flow(
            "no coke", "idle",
            cart_snapshot=[],
            previous_assistant_text="Would you like a Coke with that?",
        )
        assert d.response_key == "ask_alternative"

    def test_metadata_rejected_item_set(self):
        d = decide_short_utterance_flow(
            "no coke", "idle",
            cart_snapshot=[],
            previous_assistant_text="Would you like a Coke with that?",
        )
        assert d.metadata.get("rejected_item") == "coke"

    def test_response_offers_alternative(self):
        d = decide_short_utterance_flow(
            "no coke", "idle",
            cart_snapshot=[],
            previous_assistant_text="Would you like a Coke with that?",
        )
        lower = d.response_text.lower()
        assert "instead" in lower or "what" in lower or "else" in lower


# ---------------------------------------------------------------------------
# TC-10 — "cancel that" after last item added
# ---------------------------------------------------------------------------

class TestCancelThatRemovesLastItem:
    """TC-10: "cancel that" with cart diff → removes only the last added item."""

    def test_action_is_remove_last_item(self):
        d = decide_short_utterance_flow(
            "cancel that", "idle",
            last_cart_diff=["Cheeseburger"],
        )
        assert d.action == FlowAction.REMOVE_LAST_ITEM

    def test_tool_name_is_remove_last_cart_diff(self):
        d = decide_short_utterance_flow(
            "cancel that", "idle",
            last_cart_diff=["Cheeseburger"],
        )
        assert d.tool_name == "remove_last_cart_diff"

    def test_metadata_removed_items_correct(self):
        d = decide_short_utterance_flow(
            "cancel that", "idle",
            last_cart_diff=["Cheeseburger"],
        )
        assert "Cheeseburger" in d.metadata.get("removed_items", [])

    def test_response_mentions_removed_item(self):
        d = decide_short_utterance_flow(
            "cancel that", "idle",
            last_cart_diff=["Cheeseburger"],
        )
        assert "Cheeseburger" in d.response_text

    def test_without_cart_diff_asks_what_to_remove(self):
        """No diff tracked → can't undo; ask what to remove."""
        d = decide_short_utterance_flow("cancel that", "idle")
        assert d.action == FlowAction.ASK_CLARIFICATION


# ---------------------------------------------------------------------------
# TC-11 — "cancel the order" → CLEAR_ORDER_CONFIRM
# ---------------------------------------------------------------------------

class TestCancelOrderAsksConfirmation:
    """TC-11: "cancel the order" → asks confirmation before clearing, not silent clear."""

    def test_action_is_clear_order_confirm(self):
        d = decide_short_utterance_flow("cancel the order", "idle")
        assert d.action == FlowAction.CLEAR_ORDER_CONFIRM

    def test_requires_confirmation_is_true(self):
        d = decide_short_utterance_flow("cancel the order", "idle")
        assert d.requires_confirmation is True

    def test_response_asks_to_confirm_cancel(self):
        d = decide_short_utterance_flow("cancel the order", "idle")
        lower = d.response_text.lower()
        assert "sure" in lower or "confirm" in lower or "cancel" in lower

    def test_cancel_my_order_variant_also_works(self):
        d = decide_short_utterance_flow("cancel my order", "idle")
        assert d.action == FlowAction.CLEAR_ORDER_CONFIRM

    def test_start_over_also_triggers_confirmation(self):
        d = decide_short_utterance_flow("start over", "idle")
        assert d.action == FlowAction.CLEAR_ORDER_CONFIRM


# ---------------------------------------------------------------------------
# TC-12 — "okay fine" after assistant asks to send payment link
# ---------------------------------------------------------------------------

class TestOkayFineAfterPaymentLinkPrompt:
    """TC-12: compound affirm after payment-link prompt → SEND_PAYMENT_LINK."""

    def test_action_is_send_payment_link(self):
        d = decide_short_utterance_flow(
            "okay fine", "idle",
            previous_assistant_text="Should I send the payment link to complete your order?",
        )
        assert d.action == FlowAction.SEND_PAYMENT_LINK

    def test_okay_alone_after_payment_link_also_sends(self):
        d = decide_short_utterance_flow(
            "okay", "idle",
            previous_assistant_text="Ready to send the payment link?",
        )
        assert d.action == FlowAction.SEND_PAYMENT_LINK

    def test_is_affirm_recognises_okay_fine(self):
        """Unit-level check that the helper classifies "okay fine" as affirm."""
        assert _is_affirm("okay fine") is True

    def test_is_affirm_recognises_yeah_do_it(self):
        """Unit-level check that the helper classifies "yeah do it" as affirm."""
        assert _is_affirm("yeah do it") is True


# ---------------------------------------------------------------------------
# TC-13 — "okay fine" while waiting for required side
# ---------------------------------------------------------------------------

class TestOkayFineInWaitingForSide:
    """TC-13: compound affirm in waiting_for_side → EXECUTE_HANDLER (let handler pick side)."""

    def test_action_is_execute_handler(self):
        d = decide_short_utterance_flow("okay fine", "waiting_for_side")
        assert d.action == FlowAction.EXECUTE_HANDLER

    def test_reason_is_affirm_in_waiting_state(self):
        d = decide_short_utterance_flow("okay fine", "waiting_for_side")
        assert d.reason == "affirm_in_waiting_state"

    def test_action_is_not_send_payment_link(self):
        d = decide_short_utterance_flow("okay fine", "waiting_for_side")
        assert d.action != FlowAction.SEND_PAYMENT_LINK

    def test_works_for_all_waiting_states(self):
        """Compound affirm in any waiting state falls through to handler."""
        waiting_states = [
            "waiting_for_modifier",
            "waiting_for_size",
            "waiting_for_quantity",
            "confirming_item",
        ]
        for state in waiting_states:
            d = decide_short_utterance_flow("okay fine", state)
            assert d.action == FlowAction.EXECUTE_HANDLER, (
                f"Expected EXECUTE_HANDLER for state={state!r}, got {d.action!r}"
            )


# ---------------------------------------------------------------------------
# TC-14 — "no" in checkout confirmation → return to ordering
# ---------------------------------------------------------------------------

class TestNoInConfirmingOrder:
    """TC-14: "no" in confirming_order → FALLBACK_LOCAL (return to ordering, no link)."""

    def test_action_is_fallback_local(self):
        d = decide_short_utterance_flow("no", "confirming_order")
        assert d.action == FlowAction.FALLBACK_LOCAL

    def test_action_is_not_send_payment_link(self):
        d = decide_short_utterance_flow("no", "confirming_order")
        assert d.action != FlowAction.SEND_PAYMENT_LINK

    def test_reason_is_deny_in_confirming_order(self):
        d = decide_short_utterance_flow("no", "confirming_order")
        assert d.reason == "deny_in_confirming_order"

    def test_nope_variant_also_denies(self):
        d = decide_short_utterance_flow("nope", "confirming_order")
        assert d.action == FlowAction.FALLBACK_LOCAL


# ---------------------------------------------------------------------------
# TC-15 — "no" in idle → ASK_CLARIFICATION
# ---------------------------------------------------------------------------

class TestNoInIdle:
    """TC-15: "no" in idle with no pending context → asks clarification."""

    def test_action_is_ask_clarification(self):
        d = decide_short_utterance_flow("no", "idle")
        assert d.action == FlowAction.ASK_CLARIFICATION

    def test_action_is_not_send_payment_link(self):
        d = decide_short_utterance_flow("no", "idle")
        assert d.action != FlowAction.SEND_PAYMENT_LINK

    def test_nope_also_asks_clarification(self):
        d = decide_short_utterance_flow("nope", "idle")
        assert d.action == FlowAction.ASK_CLARIFICATION

    def test_nah_also_asks_clarification(self):
        d = decide_short_utterance_flow("nah", "idle")
        assert d.action == FlowAction.ASK_CLARIFICATION


# ---------------------------------------------------------------------------
# Bonus: edge-case / robustness checks
# ---------------------------------------------------------------------------

class TestRobustness:
    """Policy must never raise and must always return a FlowDecision."""

    def test_empty_transcript_returns_fallback(self):
        d = decide_short_utterance_flow("", "idle")
        assert isinstance(d, FlowDecision)
        assert d.action == FlowAction.FALLBACK_LOCAL

    def test_none_like_state_is_handled(self):
        d = decide_short_utterance_flow("yes", "")
        assert isinstance(d, FlowDecision)

    def test_build_checkout_confirmation_empty_cart_is_safe(self):
        d = build_checkout_confirmation([])
        assert isinstance(d, FlowDecision)
        assert d.action != FlowAction.SEND_PAYMENT_LINK

    def test_decide_off_menu_empty_item_is_safe(self):
        d = decide_off_menu_flow("")
        assert isinstance(d, FlowDecision)

    def test_build_reprompt_none_decision_is_safe(self):
        d = build_reprompt_for_lifecycle_decision(None)  # type: ignore[arg-type]
        assert isinstance(d, FlowDecision)
