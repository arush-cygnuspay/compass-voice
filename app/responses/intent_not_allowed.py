# app/responses/intent_not_allowed.py
"""Tier-aware renderer for the global ``intent_not_allowed`` response key.

The TurnEngine emits this key whenever the StateRouter rejects a
(state, intent) pair. The payload may include a ``tier`` value from
``NoInputEscalationPolicy`` — earlier tiers re-anchor with state-specific
copy, later tiers escalate to a help offer or to a human-agent handoff.

When ``tier`` is absent (older payloads / programmatic call sites), we
preserve the previous behavior so legacy tests do not break.
"""
from __future__ import annotations

from app.nlu.intent_resolution.intent import Intent
from app.policies.no_input_escalation_policy import NoInputTier
from app.state_machine.flow_sets import ADD_ITEM_FLOW_STATES
from app.state_machine.models.conversation_state import ConversationState


# State -> short anchor (tier 0/1)
_STATE_ANCHOR: dict[ConversationState, str] = {
    ConversationState.WAITING_FOR_ORDER_TYPE: "Please say pickup or delivery.",
    ConversationState.CONFIRMING_ORDER: "Please confirm the order, or say cancel.",
    ConversationState.WAITING_FOR_PAYMENT: "Please complete payment, or say cancel.",
    ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION: (
        "Want me to text the payment link, or pay at the counter?"
    ),
}

# State -> concrete options hint (tier 2)
_STATE_HINT: dict[ConversationState, str] = {
    ConversationState.WAITING_FOR_ORDER_TYPE: (
        "Just say pickup if you're picking it up, or delivery if you'd like it brought to you."
    ),
    ConversationState.CONFIRMING_ORDER: (
        "Say yes to place the order, no to keep editing, or cancel to start over."
    ),
    ConversationState.WAITING_FOR_PAYMENT: (
        "You can say I paid once you're done, or say cancel to stop."
    ),
    ConversationState.WAITING_FOR_PICKUP_SMS_PERMISSION: (
        "Say yes to text you the payment link, or no to pay when you arrive."
    ),
}

_OFFER_HELP = (
    "I'm having trouble understanding. Want me to read the options, "
    "or would you like to speak with a team member?"
)

_HANDOFF = "I'm having trouble. Let me connect you with someone who can help."


def handle_intent_not_allowed(payload: dict) -> str:
    state = payload.get("state")
    intent = payload.get("intent")
    tier = payload.get("tier")

    # Coerce tier - accept enum, raw string, or None for backwards compat.
    if isinstance(tier, NoInputTier):
        tier_value = tier.value
    elif isinstance(tier, str):
        tier_value = tier
    else:
        tier_value = None

    # Tier 3+: caller is being handed off - TurnEngine has already set the
    # state to TRANSFERRING_TO_HUMAN_AGENT and emits this string while the
    # transfer is queued.
    if tier_value == NoInputTier.HANDOFF.value:
        return _HANDOFF

    # Tier 2: explicit help offer.
    if tier_value == NoInputTier.OFFER_HELP.value:
        return _OFFER_HELP

    # Add-item flow always wins state-specific copy regardless of tier.
    if state in ADD_ITEM_FLOW_STATES and intent == Intent.SHOW_CART:
        return "Finish this item first, or say cancel."
    if state in ADD_ITEM_FLOW_STATES:
        return "Please finish this item, or say cancel."

    # Tier 1: hint with concrete options.
    if tier_value == NoInputTier.REPROMPT_WITH_HINT.value:
        return _STATE_HINT.get(state) or (
            "I didn't catch that. You can add an item, hear the menu, or say checkout."
        )

    # Tier 0 / unspecified: state-specific anchor (preserves legacy behavior).
    return _STATE_ANCHOR.get(state) or "I didn't catch that. Could you say that again?"
