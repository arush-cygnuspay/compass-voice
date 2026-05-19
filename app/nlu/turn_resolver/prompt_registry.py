# app/nlu/turn_resolver/prompt_registry.py
"""Per-task GPT prompt registry for the turn-resolution layer.

Each task mode has a separate, focused prompt. No giant universal prompt.
No GPT API calls here — only static string definitions.
PromptRegistry must not import handler, cart, or FSM implementation.
"""
from __future__ import annotations

# ── Task mode constants ───────────────────────────────────────────────────────

TASK_PICKUP_DELIVERY_INITIAL = "pickup_delivery_initial"
TASK_ORDER_TYPE_CHANGE = "order_type_change"
TASK_IDLE_ADD_ITEM_OR_MENU_QUERY = "idle_add_item_or_menu_query"
TASK_MULTI_ITEM_ADD_PLANNING = "multi_item_add_planning"
TASK_MODIFIER_OPTION_RESOLUTION = "modifier_option_resolution"
TASK_SIDE_OPTION_RESOLUTION = "side_option_resolution"
TASK_SIZE_OPTION_RESOLUTION = "size_option_resolution"
TASK_CORRECTION_CANCEL_RESOLUTION = "correction_cancel_resolution"
TASK_CHECKOUT_CONFIRMATION_RESOLUTION = "checkout_confirmation_resolution"
TASK_PAYMENT_PERMISSION_RESOLUTION = "payment_permission_resolution"
TASK_GENERIC_UNKNOWN_REPAIR = "generic_unknown_repair"

_ALL_TASK_MODES: frozenset[str] = frozenset({
    TASK_PICKUP_DELIVERY_INITIAL,
    TASK_ORDER_TYPE_CHANGE,
    TASK_IDLE_ADD_ITEM_OR_MENU_QUERY,
    TASK_MULTI_ITEM_ADD_PLANNING,
    TASK_MODIFIER_OPTION_RESOLUTION,
    TASK_SIDE_OPTION_RESOLUTION,
    TASK_SIZE_OPTION_RESOLUTION,
    TASK_CORRECTION_CANCEL_RESOLUTION,
    TASK_CHECKOUT_CONFIRMATION_RESOLUTION,
    TASK_PAYMENT_PERMISSION_RESOLUTION,
    TASK_GENERIC_UNKNOWN_REPAIR,
})

# ── Shared output contract ────────────────────────────────────────────────────

_OUTPUT_CONTRACT_BASE = (
    "Output ONLY a single JSON object. No markdown, no explanation, no extra text.\n"
    "Choose intent and options ONLY from the allowed lists provided.\n"
    "Do NOT invent menu items, option names, or response keys not in the context.\n"
    "The deterministic FSM validates and applies your suggestion — it is the final authority.\n"
    "If you are uncertain, set decision to 'clarify' and explain briefly in reason."
)

# ── System prompts per task ───────────────────────────────────────────────────

_SYSTEM_PROMPTS: dict[str, str] = {
    TASK_PICKUP_DELIVERY_INITIAL: (
        "You are a voice ordering assistant classifier. "
        "Determine whether the customer wants pickup or delivery based on their response. "
        "Output strict JSON only."
    ),
    TASK_ORDER_TYPE_CHANGE: (
        "You are a voice ordering assistant classifier. "
        "Determine whether the customer wants to switch from their current order type "
        "(pickup ↔ delivery) and to which type. Output strict JSON only."
    ),
    TASK_IDLE_ADD_ITEM_OR_MENU_QUERY: (
        "You are a voice ordering assistant. "
        "Determine whether the customer is adding a menu item, asking a menu question, "
        "asking about a specific item, or saying something unclear. "
        "Use only the menu_candidates provided. Output strict JSON only."
    ),
    TASK_MULTI_ITEM_ADD_PLANNING: (
        "You are a voice ordering assistant. "
        "Extract a structured plan for a compound multi-item utterance. "
        "Match sides, drinks, modifiers, and sizes to their parent items. "
        "Output strict JSON only."
    ),
    TASK_MODIFIER_OPTION_RESOLUTION: (
        "You are a voice ordering assistant. "
        "The customer is choosing a modifier or add-on for a pending item. "
        "Resolve their selection from the allowed_options list only. "
        "Output strict JSON only."
    ),
    TASK_SIDE_OPTION_RESOLUTION: (
        "You are a voice ordering assistant. "
        "The customer is choosing a side dish or drink for a pending item. "
        "Resolve their selection from the allowed_options list only. "
        "Preserve any size (small/medium/large) the customer mentions. "
        "Output strict JSON only."
    ),
    TASK_SIZE_OPTION_RESOLUTION: (
        "You are a voice ordering assistant. "
        "The customer is choosing a size or variant for a pending item. "
        "Resolve their selection from the allowed_options list only. "
        "Output strict JSON only."
    ),
    TASK_CORRECTION_CANCEL_RESOLUTION: (
        "You are a voice ordering assistant. "
        "Determine whether the customer is correcting something about the current item "
        "or cancelling the pending item entirely. Output strict JSON only."
    ),
    TASK_CHECKOUT_CONFIRMATION_RESOLUTION: (
        "You are a voice ordering assistant. "
        "Determine whether the customer is confirming their order for placement "
        "or rejecting/modifying it. Do not skip required order lifecycle rules. "
        "Output strict JSON only."
    ),
    TASK_PAYMENT_PERMISSION_RESOLUTION: (
        "You are a voice ordering assistant. "
        "Determine whether the customer wants to receive a payment link via SMS "
        "or pay on arrival / in person. Output strict JSON only."
    ),
    TASK_GENERIC_UNKNOWN_REPAIR: (
        "You are a voice ordering assistant fallback classifier. "
        "Given context and a user utterance that the local NLU could not resolve, "
        "determine the most likely intent from the allowed_intents list. "
        "Output strict JSON only."
    ),
}

# ── Task instructions per task ────────────────────────────────────────────────

_TASK_INSTRUCTIONS: dict[str, str] = {
    TASK_PICKUP_DELIVERY_INITIAL: (
        "Classify the customer's response as 'pickup' or 'delivery'.\n"
        "If unclear, return decision='clarify'."
    ),
    TASK_ORDER_TYPE_CHANGE: (
        "Determine if the customer wants to switch order type mid-order.\n"
        "Return the new type ('pickup' or 'delivery') or decision='clarify'."
    ),
    TASK_IDLE_ADD_ITEM_OR_MENU_QUERY: (
        "The customer may state a bare item name without 'I want' or 'add'.\n"
        "Classify as one of: add_item | ask_menu | ask_item_info | unclear.\n"
        "Match item names only to the menu_candidates list provided.\n"
        "Do NOT invent items not in menu_candidates.\n"
        "If multiple items are present, classify as add_item with all names."
    ),
    TASK_MULTI_ITEM_ADD_PLANNING: (
        "Produce a paired item plan for a compound multi-item utterance.\n"
        "Attach side/drink/modifier/size to the nearest valid parent item.\n"
        "Do NOT merge multiple real items into one.\n"
        "'6 piece wings' is a variant name, not a quantity, unless the customer "
        "says 'two 6 piece wings' (then quantity=2).\n"
        "Use only items in menu_candidates.\n"
        "Each item entry must have: item_name, quantity, and optionally "
        "size, variant, sides[], modifiers[]."
    ),
    TASK_MODIFIER_OPTION_RESOLUTION: (
        "The customer is selecting a modifier/add-on for the current pending item.\n"
        "Choose only from the allowed_options list.\n"
        "If the customer's choice is ambiguous between options, return decision='clarify'.\n"
        "If the customer asks to see the options, return decision='list_options'.\n"
        "If the modifier group is optional and the customer wants to skip, "
        "return decision='skip'."
    ),
    TASK_SIDE_OPTION_RESOLUTION: (
        "The customer is selecting a side dish or drink for the current pending item.\n"
        "Choose only from the allowed_options list.\n"
        "If the customer mentions a size (small/medium/large), include it in the response.\n"
        "If ambiguous, return decision='clarify'.\n"
        "If the side group is optional and the customer wants to skip, "
        "return decision='skip'."
    ),
    TASK_SIZE_OPTION_RESOLUTION: (
        "The customer is selecting a size or variant for the current pending item.\n"
        "Choose only from the allowed_options list.\n"
        "If ambiguous, return decision='clarify'."
    ),
    TASK_CORRECTION_CANCEL_RESOLUTION: (
        "Determine whether the customer is:\n"
        "  - correcting a field on the current pending item (decision='correction')\n"
        "  - cancelling the entire pending item (decision='cancel')\n"
        "  - doing something else (decision='clarify')\n"
        "Return what is being corrected in the 'correction_field' key if applicable."
    ),
    TASK_CHECKOUT_CONFIRMATION_RESOLUTION: (
        "Determine whether the customer confirms or rejects the order for placement.\n"
        "  - affirm: customer confirms the order as stated\n"
        "  - deny: customer wants to change something before checkout\n"
        "  - clarify: ambiguous\n"
        "Do NOT skip required order lifecycle rules. "
        "The FSM validates all payment lifecycle steps."
    ),
    TASK_PAYMENT_PERMISSION_RESOLUTION: (
        "Determine the customer's payment preference:\n"
        "  - sms_link: customer wants a payment link via SMS\n"
        "  - pay_on_arrival: customer will pay in person / on arrival\n"
        "  - clarify: ambiguous\n"
        "Do NOT collect or repeat any payment card data."
    ),
    TASK_GENERIC_UNKNOWN_REPAIR: (
        "The local NLU could not confidently resolve this utterance.\n"
        "Given the conversation context and allowed_intents, determine the most "
        "likely intent.\n"
        "If no intent in allowed_intents fits, return decision='no_match'.\n"
        "Never invent intents not in the allowed_intents list."
    ),
}

# ── Output contracts per task ─────────────────────────────────────────────────

_OUTPUT_CONTRACTS: dict[str, str] = {
    TASK_PICKUP_DELIVERY_INITIAL: (
        '{"decision": "pickup"|"delivery"|"clarify", "confidence": 0.0-1.0, '
        '"reason": "optional string"}'
    ),
    TASK_ORDER_TYPE_CHANGE: (
        '{"decision": "pickup"|"delivery"|"no_change"|"clarify", '
        '"confidence": 0.0-1.0, "reason": "optional string"}'
    ),
    TASK_IDLE_ADD_ITEM_OR_MENU_QUERY: (
        '{"decision": "add_item"|"ask_menu"|"ask_item_info"|"unclear", '
        '"items": [{"item_name": str, "quantity": int}], '
        '"confidence": 0.0-1.0, "reason": "optional string"}'
    ),
    TASK_MULTI_ITEM_ADD_PLANNING: (
        '{"decision": "add_items"|"clarify"|"no_match", '
        '"items": [{"item_name": str, "quantity": int, "size": str|null, '
        '"variant": str|null, "sides": [...], "modifiers": [...]}], '
        '"confidence": 0.0-1.0, "reason": "optional string"}'
    ),
    TASK_MODIFIER_OPTION_RESOLUTION: (
        '{"decision": "select"|"list_options"|"skip"|"clarify"|"no_match", '
        '"selected_option": str|null, "confidence": 0.0-1.0, '
        '"reason": "optional string"}'
    ),
    TASK_SIDE_OPTION_RESOLUTION: (
        '{"decision": "select"|"list_options"|"skip"|"clarify"|"no_match", '
        '"selected_option": str|null, "size_hint": str|null, '
        '"confidence": 0.0-1.0, "reason": "optional string"}'
    ),
    TASK_SIZE_OPTION_RESOLUTION: (
        '{"decision": "select"|"clarify"|"no_match", '
        '"selected_option": str|null, "confidence": 0.0-1.0, '
        '"reason": "optional string"}'
    ),
    TASK_CORRECTION_CANCEL_RESOLUTION: (
        '{"decision": "correction"|"cancel"|"clarify", '
        '"correction_field": str|null, "confidence": 0.0-1.0, '
        '"reason": "optional string"}'
    ),
    TASK_CHECKOUT_CONFIRMATION_RESOLUTION: (
        '{"decision": "affirm"|"deny"|"clarify", '
        '"confidence": 0.0-1.0, "reason": "optional string"}'
    ),
    TASK_PAYMENT_PERMISSION_RESOLUTION: (
        '{"decision": "sms_link"|"pay_on_arrival"|"clarify", '
        '"confidence": 0.0-1.0, "reason": "optional string"}'
    ),
    TASK_GENERIC_UNKNOWN_REPAIR: (
        '{"decision": "intent_resolved"|"no_match"|"clarify", '
        '"resolved_intent": str|null, "confidence": 0.0-1.0, '
        '"reason": "optional string"}'
    ),
}


class PromptRegistry:
    """Returns per-task prompts for GPT turn-resolution calls.

    Each task has a separate focused prompt. Unknown task modes return
    the generic_unknown_repair prompt rather than raising, to avoid
    crashing callers at runtime.
    """

    def get_system_prompt(self, task_mode: str) -> str:
        """Return the system prompt for *task_mode*."""
        return _SYSTEM_PROMPTS.get(task_mode, _SYSTEM_PROMPTS[TASK_GENERIC_UNKNOWN_REPAIR])

    def get_task_instructions(self, task_mode: str) -> str:
        """Return task-specific instructions for *task_mode*."""
        return _TASK_INSTRUCTIONS.get(task_mode, _TASK_INSTRUCTIONS[TASK_GENERIC_UNKNOWN_REPAIR])

    def get_output_contract(self, task_mode: str) -> str:
        """Return the expected JSON output schema string for *task_mode*."""
        base = _OUTPUT_CONTRACT_BASE
        specific = _OUTPUT_CONTRACTS.get(task_mode, _OUTPUT_CONTRACTS[TASK_GENERIC_UNKNOWN_REPAIR])
        return f"{base}\nExpected output format:\n{specific}"

    def is_known_task_mode(self, task_mode: str) -> bool:
        """Return True if *task_mode* is a registered task mode constant."""
        return task_mode in _ALL_TASK_MODES
