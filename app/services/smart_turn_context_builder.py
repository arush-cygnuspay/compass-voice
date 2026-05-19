# app/services/smart_turn_context_builder.py
"""Build compact SmartTurnContext from live handler/session state.

The resulting SmartTurnContext is the only data structure passed to
plan_smart_turn().  It contains:
  - normalized transcript + FSM state + local NLU signals
  - turn memory (up to 3 prior bot/user pairs)
  - compact cart snapshot (item names only, no prices)
  - allowed options for the current group (modifier/side choices)
  - pending item name and group name
  - reprompt count for this field
  - names of items most recently added (last_cart_diff, for correction context)

Does NOT include:
  - full menu content
  - cart prices, line totals, discounts
  - PII (phone, address, payment link)
  - API keys or credentials
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.session.session import Session
    from app.state_machine.models.conversation_context import ConversationContext

logger = logging.getLogger(__name__)

# Safety caps — prevent oversized payloads
_MAX_CART_ITEMS = 8
_MAX_ALLOWED_OPTIONS = 20
_MAX_PREVIOUS_TURNS = 3
_MAX_CART_DIFF_ITEMS = 4


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmartTurnContext:
    """Compact, LLM-safe snapshot of live turn state.

    All string fields are safe to serialize and forward to the planner.
    No prices, PII, or full menu data is ever stored here.
    """

    # Core turn signals
    transcript: str
    state: str
    local_intent: str
    local_confidence: float

    # Caller-provided relevant menu candidates (NOT full menu)
    menu_context: list[str]

    # Cart item names only (no prices, no item IDs)
    cart_snapshot: list[str]

    # Valid choices for the currently-active group (modifier/side choices)
    allowed_options: list[str]

    # Recent (bot_text, user_text) turn pairs — newest last
    previous_turns: list[tuple[str, str]]

    # Names of items most recently added to cart — used for correction context.
    # None when not tracked (caller must supply it or it falls back to None).
    last_cart_diff: list[str] | None

    # Currently-being-built item name (from pending_add_item)
    pending_item_name: str | None

    # Current modifier/side group name (from current_prompt_field or group obj)
    pending_group_name: str | None

    # How many times this field has been reprompted (0 = first ask)
    reprompt_count: int

    # Which fields were populated — used for structured logging only
    context_keys: list[str]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_smart_turn_context(
    *,
    transcript: str,
    state: str,
    local_intent: str,
    local_confidence: float,
    context: "ConversationContext",
    session: "Session | None" = None,
    menu_context: list[str] | None = None,
    allowed_options: list[str] | None = None,
    last_cart_diff: list[str] | None = None,
) -> SmartTurnContext:
    """Build a SmartTurnContext from live handler state.

    Parameters
    ----------
    transcript:
        Normalized customer utterance for this turn.
    state:
        Current FSM state name (e.g. "WAITING_FOR_MODIFIER").
    local_intent:
        Intent string predicted by local NLU (e.g. "ADD_ITEM").
    local_confidence:
        Local NLU confidence score (0.0–1.0).
    context:
        Live ConversationContext — provides turn memory, pending item,
        available choices, reprompt counts.
    session:
        Optional Session — provides cart access.
    menu_context:
        Caller-supplied relevant menu item names (≤12; NOT full menu).
        If None, the context builder returns an empty list.
    allowed_options:
        Caller-supplied valid choices for the current group.
        If None, the builder tries context.available_choices_values as a
        fallback, then gives up (returns empty list).
    last_cart_diff:
        Names of items most recently added to cart.  When None, the builder
        tries to infer from session, but will return None if unavailable.
        Callers in correction scenarios should supply this explicitly.
    """
    context_keys: list[str] = []

    # ── Cart snapshot ─────────────────────────────────────────────────────
    cart_snapshot: list[str] = []
    if session is not None:
        try:
            cart = getattr(session, "cart", None)
            if cart is not None:
                for ci in list(cart.get_items())[:_MAX_CART_ITEMS]:
                    name = getattr(ci, "name", None) or str(ci)
                    if name:
                        cart_snapshot.append(str(name))
                if cart_snapshot:
                    context_keys.append("cart_snapshot")
        except Exception:
            pass  # cart unavailable — safe to continue

    # ── Turn memory ───────────────────────────────────────────────────────
    previous_turns: list[tuple[str, str]] = []
    try:
        raw_memory = context.get_turn_memory(_MAX_PREVIOUS_TURNS)
        for entry in raw_memory:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                previous_turns.append((str(entry[0]), str(entry[1])))
        if previous_turns:
            context_keys.append("previous_turns")
    except Exception:
        pass

    # ── Pending item name ──────────────────────────────────────────────────
    pending_item_name: str | None = None
    try:
        pending = getattr(context, "pending_add_item", None)
        if pending is not None:
            item_name = getattr(pending, "item_name", None)
            if item_name:
                pending_item_name = str(item_name)
                context_keys.append("pending_item_name")
    except Exception:
        pass

    # ── Pending group name + allowed options fallback ─────────────────────
    pending_group_name: str | None = None
    try:
        # Use context.current_prompt_field as the group name indicator
        prompt_field = getattr(context, "current_prompt_field", None)
        if prompt_field:
            pending_group_name = str(prompt_field)
            context_keys.append("pending_group_name")

        # If caller didn't supply allowed_options, fall back to
        # context.available_choices_values which is set by the handler
        # when entering WAITING_FOR_MODIFIER / WAITING_FOR_SIDE.
        if not allowed_options:
            avail = tuple(getattr(context, "available_choices_values", ()) or ())
            if avail:
                allowed_options = list(avail)[:_MAX_ALLOWED_OPTIONS]
    except Exception:
        pass

    if allowed_options:
        context_keys.append("allowed_options")

    # ── Reprompt count ────────────────────────────────────────────────────
    reprompt_count = 0
    try:
        field_name = pending_group_name or (
            "modifier" if "MODIFIER" in state else
            "side" if "SIDE" in state else
            ""
        )
        if field_name:
            reprompt_count = int(context.reprompt_count(field_name))
            if reprompt_count > 0:
                context_keys.append("reprompt_count")
    except Exception:
        pass

    # ── last_cart_diff ────────────────────────────────────────────────────
    # Caller should supply this for correction scenarios.  We do not attempt
    # to compute it here (no timestamp ordering on cart items).
    resolved_diff = list(last_cart_diff[:_MAX_CART_DIFF_ITEMS]) if last_cart_diff else None
    if resolved_diff:
        context_keys.append("last_cart_diff")

    if menu_context:
        context_keys.append("menu_context")

    return SmartTurnContext(
        transcript=transcript,
        state=state,
        local_intent=local_intent,
        local_confidence=local_confidence,
        menu_context=list(menu_context or []),
        cart_snapshot=cart_snapshot,
        allowed_options=list(allowed_options or []),
        previous_turns=previous_turns,
        last_cart_diff=resolved_diff,
        pending_item_name=pending_item_name,
        pending_group_name=pending_group_name,
        reprompt_count=reprompt_count,
        context_keys=context_keys,
    )
