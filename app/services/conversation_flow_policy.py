# app/services/conversation_flow_policy.py
"""ConversationFlowPolicy — state-aware flow decisions for Compass Voice.

Answers four questions the handler layer does not need to answer individually:

  1. decide_short_utterance_flow(...)
     What does "yes / no / okay / cancel that" mean *right now*?

  2. build_checkout_confirmation(cart_snapshot)
     Build the staged checkout confirmation message (never sends payment link directly).

  3. decide_off_menu_flow(raw_item_text, resolver_result, alternatives)
     Respond to an unresolvable menu query without mutating the cart.

  4. build_reprompt_for_lifecycle_decision(decision, context)
     Translate a blocking LifecycleDecision into an exact reprompt.

Design principles
-----------------
* Pure functions — no handler imports, no I/O, no side effects.
* Returns FlowDecision(action, …) which callers can act on or ignore.
* Never raises.  Any error returns FALLBACK_LOCAL so the handler runs unchanged.
* Deterministic local handlers remain the fallback path; this policy only
  overrides when it is confident about the user's intent.
* LLM (SmartTurnPlanner) output may be passed in but is never applied directly;
  FlowDecision is always the final router.
* OrderLifecycleGuard.can_checkout result must be passed by the caller before
  a checkout-phrase decision is made — this policy never calls the guard itself.

Integration note
----------------
Call ``decide_short_utterance_flow`` at the start of any handler whose state
accepts short affirm/deny/checkout/cancel utterances.  If the returned action
is not FALLBACK_LOCAL / EXECUTE_HANDLER, skip the normal handler logic and
return the policy's response.  See integration examples in handlers.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.order_lifecycle_guard import LifecycleDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FlowAction
# ---------------------------------------------------------------------------

class FlowAction(str, Enum):
    """Machine-readable action the caller should take based on this decision."""

    EXECUTE_HANDLER         = "execute_handler"         # fall through to normal handler
    ASK_MISSING_REQUIREMENT = "ask_missing_requirement" # lifecycle guard blocked checkout
    ASK_CLARIFICATION       = "ask_clarification"       # utterance ambiguous in current context
    SUGGEST_ALTERNATIVES    = "suggest_alternatives"    # off-menu item, offer nearby items
    APPLY_CORRECTION        = "apply_correction"        # correction phrase detected
    REMOVE_LAST_ITEM        = "remove_last_item"        # "cancel that" → undo last add
    REMOVE_SPECIFIC_ITEM    = "remove_specific_item"    # "no coke" → remove named item
    CLEAR_ORDER_CONFIRM     = "clear_order_confirm"     # "cancel order" → ask confirmation
    CONFIRM_CHECKOUT        = "confirm_checkout"        # staged: show summary, ask confirm
    SEND_PAYMENT_LINK       = "send_payment_link"       # after confirmation → send link
    HANDOFF                 = "handoff"                 # transfer to human agent
    FALLBACK_LOCAL          = "fallback_local"          # policy has no opinion; use handler
    # ── Order type switching ───────────────────────────────────────────────
    CHANGE_ORDER_TYPE       = "change_order_type"       # apply the detected type switch
    ASK_DELIVERY_ADDRESS    = "ask_delivery_address"    # delivery chosen; address missing
    CONFIRM_ORDER_TYPE_CHANGE = "confirm_order_type_change"  # confirm ambiguous switch
    REJECT_ORDER_TYPE_CHANGE  = "reject_order_type_change"   # switch not allowed (payment/done)


# ---------------------------------------------------------------------------
# FlowDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FlowDecision:
    """Immutable result of a ConversationFlowPolicy function.

    Attributes
    ----------
    action:
        The recommended next step.
    reason:
        Short machine-readable code explaining why this action was chosen.
    response_text:
        Voice-ready response string for the caller.  Empty when
        action=EXECUTE_HANDLER or FALLBACK_LOCAL.
    response_key:
        Symbolic response key for ResponseBuilder lookup.  May be empty.
    target_state:
        Desired next ConversationState (lowercase value string) or None.
    handler_hint:
        Optional hint to the caller about which handler to invoke.
    tool_name:
        Optional tool / command name for side effects (e.g. "remove_item").
    tool_args:
        Arguments for tool_name.
    requires_confirmation:
        True when the action needs an explicit user confirm before execution.
    metadata:
        Structured extras (item names, alternatives, lifecycle code, etc.)
        for richer caller logic.  Never contains PII.
    """

    action: FlowAction | str
    reason: str = ""
    response_text: str = ""
    response_key: str = ""
    target_state: str | None = None
    handler_hint: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# Singleton for "no opinion — let handler decide"
_FALLBACK = FlowDecision(action=FlowAction.FALLBACK_LOCAL, reason="no_policy_match")


# ---------------------------------------------------------------------------
# Utterance pattern sets (normalized, lowercase)
# ---------------------------------------------------------------------------

_AFFIRM_WORDS: frozenset[str] = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "fine",
    "do it", "do that", "that's fine", "thats fine", "that works",
    "sounds good", "go ahead", "alright", "absolutely", "great",
    "perfect", "right", "correct", "that's right", "agreed",
    "of course", "definitely", "please do",
})

_DENY_WORDS: frozenset[str] = frozenset({
    "no", "nope", "nah", "not that", "don't want that",
    "i don't want that", "no thank you", "no thanks", "never mind",
    "not really",
})

_CHECKOUT_PHRASES: tuple[str, ...] = (
    "done", "that's it", "thats it", "that's all", "thats all",
    "check out", "checkout", "place my order", "place order",
    "submit order", "i'm done", "im done", "i am done",
    "nothing else", "no more", "that'll be all", "that will be all",
    "wrap it up", "finalize",
)

_CANCEL_LAST_PHRASES: tuple[str, ...] = (
    "cancel that", "scratch that", "remove that", "undo that",
    "take that off", "take it off", "forget that", "never mind that",
    "undo last", "undo",
)

_CLEAR_ORDER_PHRASES: tuple[str, ...] = (
    "cancel the order", "cancel my order", "cancel order",
    "clear my order", "clear the order", "clear order",
    "start over", "start again", "forget it all", "forget everything",
    "remove everything", "empty my cart", "empty cart",
)

# States where an affirm/deny directly applies to a pending system prompt.
_CONFIRMING_STATES: frozenset[str] = frozenset({
    "confirming_order",
    "cancellation_confirmation",
})

# States where the handler should process affirm/deny (pass-through).
_WAITING_STATES: frozenset[str] = frozenset({
    "waiting_for_modifier",
    "waiting_for_side",
    "waiting_for_side_size",
    "waiting_for_size",
    "waiting_for_quantity",
    "confirming_item",
})


# ---------------------------------------------------------------------------
# 1. decide_short_utterance_flow
# ---------------------------------------------------------------------------

def decide_short_utterance_flow(
    transcript: str,
    state: str,
    *,
    previous_assistant_text: str = "",
    pending_action: str | None = None,
    last_cart_diff: Sequence[str] | None = None,
    cart_snapshot: Sequence[str] | None = None,
    lifecycle_decision: "LifecycleDecision | None" = None,
    available_choices: Sequence[str] | None = None,
    smart_plan: Any = None,
) -> FlowDecision:
    """Return a FlowDecision for a short / ambiguous customer utterance.

    Parameters
    ----------
    transcript:
        Normalized (lowercased, whitespace-collapsed) customer utterance.
    state:
        Current ConversationState value string (lowercase).
    previous_assistant_text:
        The last thing the bot said to the customer (for context).
    pending_action:
        Current PendingAction value string, or None.  E.g. "add_item".
    last_cart_diff:
        Names of items added / modified in the *last* cart action.
        Used to resolve "cancel that".  None = not tracked.
    cart_snapshot:
        Current cart item names (compact, no PII).
    lifecycle_decision:
        Pre-computed result of can_checkout().  Required when the utterance
        is a checkout phrase; None otherwise.
    available_choices:
        Current available option names (e.g. modifier / side options).
    smart_plan:
        Optional SmartTurnPlan — used as a hint when local classification
        is uncertain.  Never applied directly.

    Returns
    -------
    FlowDecision with the recommended action.  Action=FALLBACK_LOCAL means
    the caller should continue with the normal handler unchanged.
    """
    try:
        return _decide_short(
            transcript=transcript,
            state=state,
            previous_assistant_text=previous_assistant_text,
            pending_action=pending_action,
            last_cart_diff=last_cart_diff,
            cart_snapshot=cart_snapshot or [],
            lifecycle_decision=lifecycle_decision,
            available_choices=available_choices or [],
            smart_plan=smart_plan,
        )
    except Exception as exc:
        logger.warning(
            "conversation_flow_policy.decide_short_utterance_flow_error",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return _FALLBACK


def _decide_short(
    *,
    transcript: str,
    state: str,
    previous_assistant_text: str,
    pending_action: str | None,
    last_cart_diff: Sequence[str] | None,
    cart_snapshot: Sequence[str],
    lifecycle_decision: "LifecycleDecision | None",
    available_choices: Sequence[str],
    smart_plan: Any,
) -> FlowDecision:
    text = (transcript or "").strip().lower()
    state = (state or "").lower()

    if not text:
        return _FALLBACK

    # ── Priority 1: clear-order phrases (destructive, must confirm) ───────
    if _matches_any(text, _CLEAR_ORDER_PHRASES):
        return FlowDecision(
            action=FlowAction.CLEAR_ORDER_CONFIRM,
            reason="clear_order_phrase",
            response_text=(
                "Are you sure you want to cancel your entire order? "
                "Just say yes to confirm."
            ),
            response_key="confirm_clear_order",
            requires_confirmation=True,
            metadata={"utterance": text},
        )

    # ── Priority 2: "cancel that" / "scratch that" (undo last add) ────────
    if _matches_any(text, _CANCEL_LAST_PHRASES):
        if last_cart_diff:
            items_str = _and_join(list(last_cart_diff)[:2])
            return FlowDecision(
                action=FlowAction.REMOVE_LAST_ITEM,
                reason="cancel_that",
                response_text=f"Okay, I removed the {items_str}.",
                response_key="item_removed_last",
                tool_name="remove_last_cart_diff",
                tool_args={"items": list(last_cart_diff)},
                metadata={"removed_items": list(last_cart_diff)},
            )
        else:
            # No diff tracked — ask what to remove
            return FlowDecision(
                action=FlowAction.ASK_CLARIFICATION,
                reason="cancel_that_no_diff",
                response_text="What would you like me to remove?",
                response_key="ask_what_to_remove",
            )

    # ── Priority 3: "no [item]" — item removal or suggestion rejection ────
    no_item = _extract_no_item(text)
    if no_item:
        cart_lower = [c.lower() for c in cart_snapshot]
        in_cart = any(no_item in c or c in no_item for c in cart_lower)
        if in_cart:
            # Item is in cart → remove it
            matched = next(
                (c for c in cart_snapshot if no_item in c.lower() or c.lower() in no_item),
                no_item,
            )
            return FlowDecision(
                action=FlowAction.REMOVE_SPECIFIC_ITEM,
                reason="no_item_in_cart",
                response_text=f"Got it, I removed the {matched}.",
                response_key="item_removed",
                tool_name="remove_item_by_name",
                tool_args={"item_name": matched},
                metadata={"item_name": matched},
            )
        # Item was suggested (mentioned in previous bot text)?
        if no_item in previous_assistant_text.lower():
            return FlowDecision(
                action=FlowAction.ASK_CLARIFICATION,
                reason="no_item_was_suggested",
                response_text="No problem. What would you like instead?",
                response_key="ask_alternative",
                metadata={"rejected_item": no_item},
            )
        # Unknown item — generic deny
        return FlowDecision(
            action=FlowAction.ASK_CLARIFICATION,
            reason="no_item_not_in_cart_not_suggested",
            response_text="What would you like?",
            response_key="ask_clarification",
        )

    # ── Priority 4: checkout / done phrases ───────────────────────────────
    if _matches_any(text, _CHECKOUT_PHRASES) or _starts_with_any(text, _CHECKOUT_PHRASES):
        return _handle_checkout_phrase(
            lifecycle_decision=lifecycle_decision,
            cart_snapshot=cart_snapshot,
        )

    # ── Priority 5: affirm utterances ─────────────────────────────────────
    if _is_affirm(text):
        return _handle_affirm(
            text=text,
            state=state,
            previous_assistant_text=previous_assistant_text,
            pending_action=pending_action,
            available_choices=available_choices,
            lifecycle_decision=lifecycle_decision,
            cart_snapshot=cart_snapshot,
        )

    # ── Priority 6: deny utterances ───────────────────────────────────────
    if _is_deny(text):
        return _handle_deny(
            text=text,
            state=state,
            previous_assistant_text=previous_assistant_text,
        )

    # ── No confident classification → let handler decide ──────────────────
    return _FALLBACK


# ---------------------------------------------------------------------------
# 2. build_checkout_confirmation
# ---------------------------------------------------------------------------

def build_checkout_confirmation(
    cart_snapshot: Sequence[str],
    *,
    total: str = "",
) -> FlowDecision:
    """Build the staged checkout confirmation message.

    The cart must be non-empty (caller's responsibility to check via
    OrderLifecycleGuard.can_checkout before calling this).

    Returns CONFIRM_CHECKOUT (never SEND_PAYMENT_LINK directly).
    The caller must wait for a subsequent AFFIRM before sending the link.
    """
    try:
        items = list(cart_snapshot)
        if not items:
            return FlowDecision(
                action=FlowAction.ASK_CLARIFICATION,
                reason="empty_cart_at_confirmation",
                response_text="Your cart is empty. What would you like to order?",
                response_key="cart_empty",
            )

        summary = _format_cart_summary(items)
        total_part = f" The total is {total}." if total else ""
        response = (
            f"Your order is {summary}.{total_part} "
            "Should I send the payment link?"
        )
        return FlowDecision(
            action=FlowAction.CONFIRM_CHECKOUT,
            reason="checkout_confirmation",
            response_text=response,
            response_key="confirm_checkout_summary",
            target_state="confirming_order",
            requires_confirmation=True,
            metadata={
                "cart_items": items,
                "total": total,
            },
        )
    except Exception as exc:
        logger.warning(
            "conversation_flow_policy.build_checkout_confirmation_error",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return _FALLBACK


# ---------------------------------------------------------------------------
# 3. decide_off_menu_flow
# ---------------------------------------------------------------------------

def decide_off_menu_flow(
    raw_item_text: str,
    resolver_result: Any = None,
    alternatives: Sequence[str] | None = None,
) -> FlowDecision:
    """Return a response when a menu item cannot be resolved.

    Parameters
    ----------
    raw_item_text:
        The exact item name or phrase the user requested.
    resolver_result:
        Optional resolver outcome (duck-typed; may carry unavailable flag).
        Pass None when not available.
    alternatives:
        Nearby item names the menu *does* carry.  Empty list = none found.

    The cart is NEVER mutated by this function.  The caller must ensure
    the item was not added before calling this function.
    """
    try:
        return _off_menu(
            raw_item_text=raw_item_text,
            resolver_result=resolver_result,
            alternatives=list(alternatives or []),
        )
    except Exception as exc:
        logger.warning(
            "conversation_flow_policy.decide_off_menu_flow_error",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return FlowDecision(
            action=FlowAction.SUGGEST_ALTERNATIVES,
            reason="off_menu_error_fallback",
            response_text="Sorry, we don't have that on the menu. What else would you like?",
            response_key="item_not_found",
            metadata={"raw_item": raw_item_text},
        )


def _off_menu(
    *,
    raw_item_text: str,
    resolver_result: Any,
    alternatives: list[str],
) -> FlowDecision:
    item_label = (raw_item_text or "").strip()
    unavailable = bool(getattr(resolver_result, "unavailable", False))
    close = [a for a in alternatives if a][:3]

    if unavailable:
        if close:
            alts = _format_alt_list(close)
            response = (
                f"Sorry, {item_label or 'that item'} isn't available right now. "
                f"We do have {alts}. What would you like?"
            )
        else:
            response = (
                f"Sorry, {item_label or 'that item'} isn't available right now. "
                "What else can I get you?"
            )
        response_key = "item_unavailable"
    else:
        if close:
            alts = _format_alt_list(close)
            response = (
                f"Sorry, we don't have {item_label or 'that'}. "
                f"You can try {alts}."
            )
        else:
            response = (
                "Sorry, we don't have that on the menu. "
                "What else would you like?"
            )
        response_key = "item_not_found"

    return FlowDecision(
        action=FlowAction.SUGGEST_ALTERNATIVES,
        reason="off_menu_with_alternatives" if close else "off_menu_no_alternatives",
        response_text=response,
        response_key=response_key,
        metadata={
            "raw_item": item_label,
            "alternatives": close,
            "unavailable": unavailable,
            "cart_unchanged": True,
        },
    )


# ---------------------------------------------------------------------------
# 4. build_reprompt_for_lifecycle_decision
# ---------------------------------------------------------------------------

def build_reprompt_for_lifecycle_decision(
    decision: "LifecycleDecision",
    context: Any = None,
) -> FlowDecision:
    """Translate a blocking LifecycleDecision into an exact reprompt.

    Parameters
    ----------
    decision:
        A blocking LifecycleDecision from OrderLifecycleGuard.
    context:
        Optional duck-typed context object for richer response building.
        Not required — the LifecycleDecision already carries the response text.

    Returns
    -------
    FlowDecision with action=ASK_MISSING_REQUIREMENT and the exact
    voice-ready prompt from the LifecycleDecision.
    """
    try:
        from app.services.order_lifecycle_guard import LifecycleCode

        if decision is None:
            return _FALLBACK

        code_val = getattr(decision, "code", None)
        code_str = code_val.value if hasattr(code_val, "value") else str(code_val or "")
        response_text = getattr(decision, "response", "") or ""
        details = dict(getattr(decision, "details", {}) or {})

        # Map lifecycle code to a response key
        _code_to_key: dict[str, str] = {
            "size_required": "ask_for_size",
            "side_required": "ask_for_side",
            "modifier_required": "ask_for_modifier",
            "cart_empty": "cart_empty",
            "cart_incomplete": "cart_incomplete",
            "item_not_found": "item_not_found",
            "item_unavailable": "item_unavailable",
            "payment_not_ready": "payment_not_ready",
            "order_not_ready": "order_not_ready",
        }
        response_key = _code_to_key.get(code_str, "ask_missing_requirement")

        return FlowDecision(
            action=FlowAction.ASK_MISSING_REQUIREMENT,
            reason=f"lifecycle_guard_{code_str}",
            response_text=response_text,
            response_key=response_key,
            metadata={
                "lifecycle_code": code_str,
                "lifecycle_details": details,
            },
        )
    except Exception as exc:
        logger.warning(
            "conversation_flow_policy.build_reprompt_error",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return FlowDecision(
            action=FlowAction.ASK_MISSING_REQUIREMENT,
            reason="reprompt_error_fallback",
            response_text="I still need a bit more information. What would you like?",
            response_key="ask_missing_requirement",
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_affirm(text: str) -> bool:
    """Return True when the utterance is a short affirmative."""
    t = text.strip().rstrip(".,!")
    if t in _AFFIRM_WORDS:
        return True
    # Short compound affirmatives whose first word is affirm: "yeah do it", "okay fine"
    words = t.split()
    if 1 < len(words) <= 3 and words[0] in _AFFIRM_WORDS:
        return True
    # Multi-word phrase exact match or startswith
    return any(text.startswith(w) and len(text.split()) <= 4 for w in _AFFIRM_WORDS if " " in w)


def _is_deny(text: str) -> bool:
    """Return True when the utterance is a short denial."""
    t = text.strip().rstrip(".,!")
    return t in _DENY_WORDS or any(
        t.startswith(w) and len(text.split()) <= 4 for w in _DENY_WORDS if " " in w
    )


def _matches_any(text: str, phrases: tuple[str, ...] | frozenset[str]) -> bool:
    t = text.strip()
    return t in phrases


def _starts_with_any(text: str, phrases: tuple[str, ...]) -> bool:
    t = text.strip()
    return any(t.startswith(p) for p in phrases)


# Pattern: "no " followed by 1–4 words (item name)
_NO_ITEM_RE = re.compile(r"^no\s+([a-z0-9 ]{2,40})$")


def _extract_no_item(text: str) -> str | None:
    """Extract item name from 'no [item]' patterns.  Returns None if no match."""
    m = _NO_ITEM_RE.match(text.strip())
    if m:
        candidate = m.group(1).strip()
        # Avoid matching plain "no thank you" / "no thanks"
        if candidate not in {"thank you", "thanks", "way", "more", "problem"}:
            return candidate
    return None


def _handle_checkout_phrase(
    *,
    lifecycle_decision: "LifecycleDecision | None",
    cart_snapshot: Sequence[str],
) -> FlowDecision:
    """Handle a checkout / done utterance."""
    if lifecycle_decision is None:
        # No lifecycle result supplied — build a minimal confirmation
        return build_checkout_confirmation(cart_snapshot)

    blocking = bool(getattr(lifecycle_decision, "blocking", False))
    if blocking:
        return build_reprompt_for_lifecycle_decision(lifecycle_decision)

    # Lifecycle guard cleared — proceed to staged confirmation
    return build_checkout_confirmation(cart_snapshot)


def _handle_affirm(
    *,
    text: str,
    state: str,
    previous_assistant_text: str,
    pending_action: str | None,
    available_choices: Sequence[str],
    lifecycle_decision: "LifecycleDecision | None",
    cart_snapshot: Sequence[str],
) -> FlowDecision:
    """Handle an affirm utterance given the current context."""

    # In CONFIRMING_ORDER state: affirm → payment link
    if state == "confirming_order":
        return FlowDecision(
            action=FlowAction.SEND_PAYMENT_LINK,
            reason="affirm_in_confirming_order",
            response_text="",  # caller builds payment message
            response_key="send_payment_link",
        )

    # In CANCELLATION_CONFIRMATION state: affirm → execute the cancel
    if state == "cancellation_confirmation":
        return FlowDecision(
            action=FlowAction.EXECUTE_HANDLER,
            reason="affirm_in_cancellation_confirmation",
        )

    # In WAITING states: affirm → let the specific waiting handler decide
    # (it knows about current options and what AFFIRM means in context)
    if state in _WAITING_STATES:
        return FlowDecision(
            action=FlowAction.EXECUTE_HANDLER,
            reason="affirm_in_waiting_state",
        )

    # A pending action exists → affirm applies to it
    if pending_action:
        return FlowDecision(
            action=FlowAction.EXECUTE_HANDLER,
            reason="affirm_with_pending_action",
        )

    # Previous bot text mentioned "payment link" → user confirms payment
    prev_lower = previous_assistant_text.lower()
    if "payment link" in prev_lower or "send the payment link" in prev_lower:
        return FlowDecision(
            action=FlowAction.SEND_PAYMENT_LINK,
            reason="affirm_after_payment_link_prompt",
            response_text="",
            response_key="send_payment_link",
        )

    # Previous bot text was a checkout confirmation → affirm = confirm checkout
    if "should i send" in prev_lower or "confirm" in prev_lower:
        return FlowDecision(
            action=FlowAction.SEND_PAYMENT_LINK,
            reason="affirm_after_checkout_confirmation",
            response_text="",
            response_key="send_payment_link",
        )

    # No clear confirmable context → ask clarification
    return FlowDecision(
        action=FlowAction.ASK_CLARIFICATION,
        reason="affirm_no_pending_context",
        response_text="What would you like to do?",
        response_key="ask_clarification",
    )


def _handle_deny(
    *,
    text: str,
    state: str,
    previous_assistant_text: str,
) -> FlowDecision:
    """Handle a deny utterance given the current context."""

    # In CONFIRMING_ORDER: deny → return to ordering (no payment)
    if state == "confirming_order":
        return FlowDecision(
            action=FlowAction.FALLBACK_LOCAL,
            reason="deny_in_confirming_order",
            response_text="No problem. What else would you like?",
            response_key="return_to_ordering",
        )

    # In CANCELLATION_CONFIRMATION: deny → cancel the cancellation
    if state == "cancellation_confirmation":
        return FlowDecision(
            action=FlowAction.EXECUTE_HANDLER,
            reason="deny_in_cancellation_confirmation",
        )

    # In WAITING states: deny → let the handler process it
    if state in _WAITING_STATES:
        return FlowDecision(
            action=FlowAction.EXECUTE_HANDLER,
            reason="deny_in_waiting_state",
        )

    # IDLE with no pending context → ask clarification
    return FlowDecision(
        action=FlowAction.ASK_CLARIFICATION,
        reason="deny_no_context",
        response_text="What would you like?",
        response_key="ask_clarification",
    )


def _format_cart_summary(items: Sequence[str]) -> str:
    """Format cart item names into a spoken summary."""
    unique = list(dict.fromkeys(items))  # preserve order, deduplicate
    if not unique:
        return "nothing yet"
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    # 3+: "A, B, and C" (serial comma)
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def _format_alt_list(names: list[str]) -> str:
    """Format up to 3 alternative names into a natural list."""
    items = [n for n in names if n][:3]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return f"{items[0]}, {items[1]}, or {items[2]}"


def _and_join(names: list[str]) -> str:
    """Format a short list of names with 'and'."""
    if not names:
        return "that item"
    if len(names) == 1:
        return names[0]
    return f"{names[0]} and {names[1]}"
