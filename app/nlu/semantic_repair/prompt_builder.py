# app/nlu/semantic_repair/prompt_builder.py
"""Build the compact GPT verify_extract prompt for semantic repair.

Safety contract
---------------
* Never include: full menu data, full cart JSON, phone numbers, payment
  links, the full Intent enum, or the OPENAI_API_KEY.
* Candidates list is a curated short set derived from the current FSM state —
  never the complete enum.
* Cart summary is item count + name list only (no prices, no PII).
* Slot values in output are validated against the utterance + choices before use.
"""
from __future__ import annotations

import json

from app.state_machine.models.conversation_state import ConversationState


# Curated per-state candidate sets.  Must be short (<=10 values) and must
# never expose the full Intent enum to the model.
_STATE_CANDIDATES: dict[str, frozenset[str]] = {
    ConversationState.IDLE.value: frozenset({
        "add_item", "checkout", "show_cart", "show_total",
        "show_menu", "ask_item_info", "ask_price",
        "ask_options", "cancel_order", "unknown",
    }),
    ConversationState.CONFIRMING_ORDER.value: frozenset({
        "confirm", "deny", "show_cart", "cancel_order", "unknown",
    }),
    ConversationState.CONFIRMING_ITEM.value: frozenset({
        "confirm", "deny", "cancel", "unknown",
    }),
    ConversationState.WAITING_FOR_SIDE.value: frozenset({
        "add_item", "confirm", "deny", "cancel", "unknown",
    }),
    ConversationState.WAITING_FOR_MODIFIER.value: frozenset({
        "add_item", "confirm", "deny", "cancel", "unknown",
    }),
    ConversationState.WAITING_FOR_SIZE.value: frozenset({
        "add_item", "confirm", "deny", "cancel", "unknown",
    }),
    ConversationState.WAITING_FOR_SIDE_SIZE.value: frozenset({
        "add_item", "confirm", "deny", "cancel", "unknown",
    }),
    ConversationState.WAITING_FOR_QUANTITY.value: frozenset({
        "add_item", "confirm", "deny", "cancel", "unknown",
    }),
    ConversationState.CANCELLATION_CONFIRMATION.value: frozenset({
        "confirm", "deny", "unknown",
    }),
    ConversationState.WAITING_FOR_PAYMENT.value: frozenset({
        "payment_status", "payment_done", "confirm", "deny", "unknown",
    }),
    ConversationState.GREETING.value: frozenset({
        "add_item", "show_menu", "greeting", "unknown",
    }),
}

_DEFAULT_CANDIDATES: frozenset[str] = frozenset({
    "add_item", "checkout", "confirm", "deny", "cancel", "unknown",
})

_CONTROL_INTENTS: frozenset[str] = frozenset({
    "cancel", "confirm", "deny", "checkout", "unknown",
})

_SYSTEM_PROMPT = (
    "You verify restaurant-order NLU. "
    "Pick only allowed intent/control/slots. "
    "Extract slots only from text or choices. "
    "Return JSON only. No customer text."
)

# Compact schema shown inline with the payload
_OUTPUT_SCHEMA = (
    '{"decision":"ok|repair|missing_info|fallback|no_repair",'
    '"intent":"<one of allowed.intents or null>",'
    '"control":null,'
    '"slots":[{"n":"SLOT","v":"value","op":"add|replace|remove"}],'
    '"missing":["SLOT_NAME"],'
    '"fallback_type":"none|off_topic|restaurant_question|user_frustrated'
    '|request_human|unclear|unsupported_request|back_to_order",'
    '"conf":0.9,"why":"<max 20 words>","rhv":true}'
)

_RULES = (
    "Rules: "
    "ok=local model correct, no changes. "
    "repair=use a different intent from allowed.intents. "
    "missing_info=required slot absent from utterance; list in missing[]. "
    "fallback=utterance unrelated to ordering (off-topic, frustration, etc). "
    "Slot values must appear verbatim in text or choices. "
    "rhv must be true. "
    "No customer-facing text in output."
)


def get_candidates(state_name: str) -> frozenset[str]:
    """Return the curated candidate set for the given FSM state value."""
    return _STATE_CANDIDATES.get(state_name, _DEFAULT_CANDIDATES)


def build_messages(
    *,
    utterance: str,
    state_name: str,
    candidates: frozenset[str],
    # Extended context (all optional for backward compat)
    current_prompt_field: str | None = None,
    current_item_name: str | None = None,
    intent_candidates: tuple = (),
    cart_summary: dict | None = None,
    repeat_count: int = 0,
    slots: tuple = (),
    selected_intent: str | None = None,
    selected_confidence: float | None = None,
    # New context fields
    choices: tuple[str, ...] = (),
    required_missing: tuple[str, ...] = (),
    previous_turns: tuple[tuple[str, str], ...] = (),
) -> list[dict]:
    """Return the messages list for openai.chat.completions.create().

    Builds a compact JSON payload with short keys to minimise token cost.
    The utterance is the normalized text from NLU — never raw STT output.
    """
    payload: dict = {
        "t": "verify_extract",
        "state": state_name,
        "text": utterance,
    }

    if current_prompt_field:
        payload["prompt"] = current_prompt_field
    if current_item_name:
        payload["item"] = current_item_name
    if choices:
        payload["choices"] = list(choices)
    if required_missing:
        payload["required"] = list(required_missing)

    # Local NLU snapshot — short keys
    local_block: dict = {
        "intent": selected_intent or "unknown",
        "conf": round(selected_confidence, 3) if selected_confidence is not None else 0.0,
    }

    if intent_candidates:
        local_block["top_k"] = [
            {
                "intent": getattr(c, "canonical_intent", "") or getattr(c, "intent_sub_intent", ""),
                "conf": round(getattr(c, "confidence", 0.0), 3),
            }
            for c in intent_candidates
        ]

    if slots:
        local_block["slots"] = [
            {
                "n": getattr(s, "name", ""),
                "v": str(getattr(s, "value", "")),
            }
            for s in slots
        ]

    payload["local"] = local_block

    # Allowed candidate sets (separated so model doesn't confuse repair vs control)
    repair_intents = sorted(candidates - _CONTROL_INTENTS) or sorted(candidates)
    control_intents = sorted(candidates & _CONTROL_INTENTS)
    payload["allowed"] = {
        "intents": repair_intents,
        "control": control_intents,
    }

    # Cart summary (item count + names only, never prices or PII)
    if cart_summary is not None:
        payload["cart"] = {
            "n": cart_summary.get("count", 0),
            "items": cart_summary.get("items", [])[:10],
        }

    # Conversation history (last 3 turns, oldest first)
    if previous_turns:
        payload["history"] = [[role, text] for role, text in previous_turns]

    if repeat_count:
        payload["fb_count"] = repeat_count

    user_content = (
        json.dumps(payload, ensure_ascii=False)
        + "\n\nOutput schema:\n" + _OUTPUT_SCHEMA
        + "\n\n" + _RULES
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
