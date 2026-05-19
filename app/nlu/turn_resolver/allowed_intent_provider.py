# app/nlu/turn_resolver/allowed_intent_provider.py
"""State-specific allowed intent provider for GPT context building.

Kept independent from GPT code so deterministic validators can reuse it.
All state names are normalised to lowercase before lookup.
Unknown states return a conservative safe-only set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.state_machine.models.conversation_context import ConversationContext


@dataclass(frozen=True, slots=True)
class AllowedIntent:
    """Descriptor for one intent that is valid in a given conversation state."""

    name: str
    description: str
    apply_mode_allowed: bool = True
    notes: str | None = None


# ── State → tuple[AllowedIntent, ...] registry ──────────────────────────────
# Immutable tuples mean no allocation on lookup after first class load.

_SAFE_FALLBACK: tuple[AllowedIntent, ...] = (
    AllowedIntent("no_action", "No applicable action for this state.", apply_mode_allowed=False),
    AllowedIntent("handoff_request", "Transfer to a human agent."),
)

_STATE_INTENTS: dict[str, tuple[AllowedIntent, ...]] = {
    "idle": (
        AllowedIntent("add_item", "Add a menu item to the cart."),
        AllowedIntent("remove_item", "Remove a menu item from the cart."),
        AllowedIntent("modify_item", "Modify a cart item (modifier, size, quantity)."),
        AllowedIntent("checkout", "Proceed to checkout / confirm the order."),
        AllowedIntent("ask_menu", "Ask what is on the menu."),
        AllowedIntent("ask_item_info", "Ask about a specific item."),
        AllowedIntent("change_order_type", "Switch between pickup and delivery."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    "waiting_for_order_type": (
        AllowedIntent("select_order_type", "Choose pickup or delivery."),
        AllowedIntent("change_order_type", "Restate order type preference."),
        AllowedIntent("ask_clarification", "Ask the bot to clarify the question."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    "waiting_for_modifier": (
        AllowedIntent("select_modifier", "Choose a modifier/add-on for the pending item."),
        AllowedIntent("list_options", "List available modifier options."),
        AllowedIntent("skip_optional_modifier", "Skip an optional modifier group."),
        AllowedIntent("cancel_pending_item", "Cancel the current pending item."),
        AllowedIntent("checkout_request", "Stop adding items and go to checkout."),
        AllowedIntent("change_order_type", "Switch between pickup and delivery."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    "waiting_for_side": (
        AllowedIntent("select_side", "Choose a side/drink for the pending item."),
        AllowedIntent("list_options", "List available side options."),
        AllowedIntent("skip_optional_side", "Skip an optional side group."),
        AllowedIntent("cancel_pending_item", "Cancel the current pending item."),
        AllowedIntent("checkout_request", "Stop adding items and go to checkout."),
        AllowedIntent("change_order_type", "Switch between pickup and delivery."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    "waiting_for_size": (
        AllowedIntent("select_size", "Choose a size or variant for the pending item."),
        AllowedIntent("list_options", "List available size options."),
        AllowedIntent("cancel_pending_item", "Cancel the current pending item."),
        AllowedIntent("checkout_request", "Stop adding items and go to checkout."),
        AllowedIntent("change_order_type", "Switch between pickup and delivery."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    "waiting_for_side_size": (
        AllowedIntent("select_side_size", "Choose a size variant for the current side item."),
        AllowedIntent("list_options", "List available size options for the side."),
        AllowedIntent("cancel_pending_item", "Cancel the current pending item."),
        AllowedIntent("checkout_request", "Stop adding items and go to checkout."),
        AllowedIntent("change_order_type", "Switch between pickup and delivery."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    "confirming_item": (
        AllowedIntent("affirm", "Confirm the item as stated."),
        AllowedIntent("deny", "Reject the proposed item."),
        AllowedIntent("correction", "Correct something about the item."),
        AllowedIntent("cancel_pending_item", "Cancel the pending item entirely."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    "confirming_order": (
        AllowedIntent("affirm", "Confirm and place the order."),
        AllowedIntent("deny", "Reject or modify the order before placement."),
        AllowedIntent("checkout", "Explicitly proceed to payment."),
        AllowedIntent("change_order_type", "Switch between pickup and delivery."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    "waiting_for_pickup_sms_permission": (
        AllowedIntent("affirm", "Agree to receive an SMS."),
        AllowedIntent("deny", "Decline the SMS."),
        AllowedIntent("payment_preference", "State a payment or delivery preference."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    # Terminal / completed states — only safe read-only actions
    "completed": (
        AllowedIntent("no_action", "Order is complete; no further actions.", apply_mode_allowed=False),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
    "transferring_to_human_agent": (
        AllowedIntent("no_action", "Call is being transferred; no further actions.", apply_mode_allowed=False),
    ),
    "cancellation_confirmation": (
        AllowedIntent("affirm", "Confirm the cancellation."),
        AllowedIntent("deny", "Cancel the cancellation — continue the order."),
        AllowedIntent("handoff_request", "Transfer to a human agent."),
    ),
}


class AllowedIntentProvider:
    """Returns state-specific allowed intents for GPT and validator use.

    Thread-safe: all state is class-level read-only dicts.
    """

    def get_allowed_intents_for_state(
        self,
        state: str,
        context: "ConversationContext | None" = None,
    ) -> tuple[AllowedIntent, ...]:
        """Return allowed intents for *state*.

        Unknown states return the conservative safe fallback set only.
        Context is accepted for future context-aware filtering but not used now.
        """
        key = (state or "").strip().lower()
        return _STATE_INTENTS.get(key, _SAFE_FALLBACK)

    def get_allowed_intent_names(
        self,
        state: str,
        context: "ConversationContext | None" = None,
    ) -> tuple[str, ...]:
        """Convenience: return just the name strings for *state*."""
        return tuple(ai.name for ai in self.get_allowed_intents_for_state(state, context))
