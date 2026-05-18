# app/nlu/semantic_repair/add_item_extractor.py
"""GPT ADD_ITEM shadow extractor — dataclasses, eligibility gate, payload builder.

Shadow-only contract
--------------------
* This module extracts item structure for logging and future training data.
* It NEVER mutates cart, session state, intent_result, slots, or the
  customer-facing response.
* GPT receives only a compact safe payload — no full menu, no full cart JSON,
  no full Intent enum, no API key, no phone/address/payment PII.

Supported states for extraction (add-item flow states):
  IDLE, CONFIRMING_ITEM, WAITING_FOR_SIDE, WAITING_FOR_MODIFIER,
  WAITING_FOR_SIZE, WAITING_FOR_SIDE_SIZE, WAITING_FOR_QUANTITY
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.state_machine.models.conversation_state import ConversationState

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig
    from app.nlu.intent_resolution.intent_result import IntentResult


# ---------------------------------------------------------------------------
# Frozen dataclasses — parsed GPT output (log-only, never applied to cart)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GptAddItemChild:
    """A side dish or modifier attached to a parent item."""

    name: str
    operation: str = "add"          # "add" | "remove" | "replace"
    quantity: int | None = None
    size: str | None = None
    variant: str | None = None
    modifiers: tuple[str, ...] = ()
    parse_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GptAddItem:
    """One structured add-item entry extracted by GPT."""

    item: str
    quantity: int | None = None
    size: str | None = None
    variant: str | None = None
    sides: tuple[GptAddItemChild, ...] = ()
    modifiers: tuple[GptAddItemChild, ...] = ()
    missing: tuple[str, ...] = ()
    parse_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GptAddItemPlan:
    """Full result of one GPT ADD_ITEM extractor call (or a no-op when skipped)."""

    decision: str = "no_repair"
    intent: str | None = None
    items: tuple[GptAddItem, ...] = ()
    global_slots: tuple[Any, ...] = ()
    missing: tuple[str, ...] = ()
    fallback_type: str = "none"
    confidence: float | None = None
    reason: str | None = None
    parse_error: str | None = None
    latency_ms: float | None = None
    total_ms: float | None = None
    timeout: bool = False
    prompt_chars: int = 0
    completion_chars: int = 0
    model: str | None = None
    eligible: bool = False
    skipped_reason: str | None = None
    parse_notes: tuple[str, ...] = ()


# Sentinel returned when the extractor was not called at all.
ADD_ITEM_NOT_CALLED = GptAddItemPlan(
    decision="no_repair",
    eligible=False,
    skipped_reason="not_called",
)

# ---------------------------------------------------------------------------
# Supported states for the ADD_ITEM extractor
# ---------------------------------------------------------------------------

_ADD_ITEM_SUPPORTED_STATES: frozenset[str] = frozenset({
    ConversationState.IDLE.value,
    ConversationState.CONFIRMING_ITEM.value,
    ConversationState.WAITING_FOR_SIDE.value,
    ConversationState.WAITING_FOR_MODIFIER.value,
    ConversationState.WAITING_FOR_SIZE.value,
    ConversationState.WAITING_FOR_SIDE_SIZE.value,
    ConversationState.WAITING_FOR_QUANTITY.value,
})

_TERMINAL_STATES: frozenset[str] = frozenset({
    ConversationState.COMPLETED.value,
    ConversationState.TRANSFERRING_TO_HUMAN_AGENT.value,
    ConversationState.ERROR_RECOVERY.value,
})


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------


class AddItemEligibilityGate:
    """Decide whether the ADD_ITEM extractor should run for a given turn."""

    def check(
        self,
        *,
        intent_result: "IntentResult",
        state: ConversationState,
        normalized_text: str,
        gpt_shadow_decision: str | None = None,
        gpt_shadow_repaired_intent: str | None = None,
        config: "SemanticRepairConfig",
    ) -> tuple[bool, str]:
        """Return (eligible: bool, reason: str).

        Reasons:
          mode_disabled            – add_item_mode is "disabled"
          terminal_state           – conversation is in a terminal/transfer state
          intent_not_add_item      – post-coercion intent is not ADD_ITEM
          state_not_supported      – state is not in the supported add-item states
          text_too_short           – normalized text is below min_text_len
          coalesce_existing_repair – existing semantic repair already repaired to add_item
          eligible_add_item        – all checks passed
        """
        from app.nlu.intent_resolution.intent import Intent

        if config.add_item_mode == "disabled":
            return False, "mode_disabled"

        if state.value in _TERMINAL_STATES:
            return False, "terminal_state"

        # Post-coercion intent must be ADD_ITEM
        if intent_result.intent != Intent.ADD_ITEM:
            return False, "intent_not_add_item"

        if state.value not in _ADD_ITEM_SUPPORTED_STATES:
            return False, "state_not_supported"

        text = (normalized_text or "").strip()
        if len(text) < config.add_item_min_text_len:
            return False, "text_too_short"

        # If the upstream semantic repair already identified add_item, the
        # ADD_ITEM extractor would be redundant — skip to avoid double-spend.
        if (
            gpt_shadow_decision in ("repair", "repair_intent", "repair_intent_and_slots")
            and gpt_shadow_repaired_intent == "add_item"
        ):
            return False, "coalesce_existing_repair"

        return True, "eligible_add_item"


# ---------------------------------------------------------------------------
# Compact payload builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You extract restaurant add-item requests. "
    "Attach size/variant to the exact entity named. "
    "Use only allowed slots. "
    "Return JSON only. No customer text."
)

# Slot names the extractor is allowed to populate
_ALLOWED_SLOT_NAMES: list[str] = [
    "ITEM", "MODIFIER", "SIDE", "QUANTITY", "SIZE", "VARIANT",
]

# Compact size/variant scoping rules embedded in the payload
_SIZE_SCOPE_RULES = (
    "Size/variant rules: "
    "'large pizza with coke'→pizza.size=large,coke.size=null. "
    "'pizza with large coke'→pizza.size=null,coke.size=large. "
    "'large pizza with large coke'→both sizes set. "
    "Modifiers do not support size/variant in current schema; "
    "output size for modifier anyway but parser will strip it."
)

# Output schema shown inline in every payload
_OUTPUT_SCHEMA = (
    '{"decision":"ok|repair|missing_info|fallback|no_repair",'
    '"intent":"add_item",'
    '"items":[{"item":"string","quantity":1,"size":"string|null","variant":"string|null",'
    '"sides":[{"name":"string","operation":"add|remove|replace","quantity":1,'
    '"size":"string|null","variant":"string|null","modifiers":[]}],'
    '"modifiers":[{"name":"string","operation":"add|remove|replace","quantity":1,'
    '"size":"string|null","variant":"string|null"}],'
    '"missing":[]}],'
    '"global_slots":[],'
    '"missing":[],'
    '"fallback_type":"none|off_topic|restaurant_question|user_frustrated|'
    'request_human|unclear|unsupported_request|back_to_order",'
    '"confidence":0.0,"reason":"max 20 words",'
    '"requires_handler_validation":true}'
)

# States where choices should be included in the payload
_CHOICES_STATES: frozenset[str] = frozenset({
    ConversationState.WAITING_FOR_SIDE.value,
    ConversationState.WAITING_FOR_MODIFIER.value,
    ConversationState.WAITING_FOR_SIZE.value,
    ConversationState.WAITING_FOR_SIDE_SIZE.value,
    ConversationState.WAITING_FOR_QUANTITY.value,
})


class AddItemPayloadBuilder:
    """Build compact JSON payloads for the GPT ADD_ITEM extractor.

    Safety contract — the payload must never contain:
      * Full menu data
      * Full cart JSON (compact summary only: names + count)
      * Full Intent enum (only curated add-item candidates)
      * API key or any PII (phone, address, payment links)
      * Prices or tax data
      * The current final bot response text
    """

    MAX_CART_ITEMS: int = 10
    MAX_HISTORY_TURNS: int = 3
    MAX_TOP_K: int = 4
    MAX_CHOICES: int = 12

    def build_messages(
        self,
        *,
        state: ConversationState,
        normalized_text: str,
        current_item: str,
        prompt_field: str,
        local_intent: str,
        local_confidence: float,
        top_k_intents: list[dict[str, Any]],
        local_slots: list[dict[str, Any]],
        choices: list[str],
        required_missing: list[str],
        cart_item_names: list[str],
        previous_turns: list[tuple[str, str]],
    ) -> list[dict[str, str]]:
        """Return the [system, user] messages list for OpenAI chat completions."""
        payload = self._build_payload(
            state=state,
            normalized_text=normalized_text,
            current_item=current_item,
            prompt_field=prompt_field,
            local_intent=local_intent,
            local_confidence=local_confidence,
            top_k_intents=top_k_intents,
            local_slots=local_slots,
            choices=choices,
            required_missing=required_missing,
            cart_item_names=cart_item_names,
            previous_turns=previous_turns,
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _build_payload(
        self,
        *,
        state: ConversationState,
        normalized_text: str,
        current_item: str,
        prompt_field: str,
        local_intent: str,
        local_confidence: float,
        top_k_intents: list[dict[str, Any]],
        local_slots: list[dict[str, Any]],
        choices: list[str],
        required_missing: list[str],
        cart_item_names: list[str],
        previous_turns: list[tuple[str, str]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "t": "extract_add_items",
            "state": state.value,
        }

        # Current FSM prompt field (e.g. "size", "side") — only when non-empty
        if prompt_field:
            payload["prompt"] = prompt_field

        # Current item being assembled (e.g. "Hawaiian pizza")
        if current_item:
            payload["current_item"] = current_item

        payload["text"] = normalized_text

        # Local NLU snapshot
        local_block: dict[str, Any] = {
            "intent": local_intent,
            "conf": round(local_confidence, 4),
        }
        top_k_capped = top_k_intents[: self.MAX_TOP_K]
        if top_k_capped:
            local_block["top_k"] = top_k_capped
        if local_slots:
            local_block["slots"] = local_slots
        payload["local"] = local_block

        # Allowed intents/control/slot names (curated, never full enum)
        payload["allowed"] = {
            "intents": ["add_item"],
            "control": ["confirm", "deny", "cancel"],
            "slot_names": _ALLOWED_SLOT_NAMES,
        }

        # Choices for waiting states (e.g. available sides, modifiers, sizes)
        if state.value in _CHOICES_STATES and choices:
            payload["choices"] = choices[: self.MAX_CHOICES]

        # Required but missing slots (from FSM state context)
        if required_missing:
            payload["required"] = required_missing

        # Compact cart summary (names + count, no prices)
        if cart_item_names:
            capped = cart_item_names[: self.MAX_CART_ITEMS]
            payload["cart"] = {
                "n": len(cart_item_names),
                "items": capped,
            }

        # Recent conversation history (bot/user pairs, capped)
        if previous_turns:
            capped_turns = list(previous_turns[-self.MAX_HISTORY_TURNS:])
            payload["history"] = [
                [role, text] for role, text in capped_turns
            ]

        # Size/variant scope rules (always included for add-item turns)
        payload["rules"] = _SIZE_SCOPE_RULES

        # Output schema (always last so GPT sees it near the instruction)
        payload["schema"] = _OUTPUT_SCHEMA

        return payload
