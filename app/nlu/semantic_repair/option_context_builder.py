# app/nlu/semantic_repair/option_context_builder.py
"""Build compact GPT payloads for the Phase 3 option resolver.

Safety contract
---------------
The payload must NEVER contain:
  * Full menu data or the item catalogue
  * Full cart JSON (only the current item name is used)
  * API keys or any PII (phone numbers, addresses, payment links)
  * Prices or tax figures
  * The current bot response text (only the response_key label is allowed)
  * More than MAX_CHOICES options from the current modifier group
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.state_machine.models.pending_item_models import PendingModifierGroup

# System prompt — short and action-focused.
_SYSTEM_PROMPT = (
    "You resolve modifier option selections for a restaurant voice ordering bot. "
    "The user is responding to a question about which modifier to add to their item. "
    "Match the user's text to one or more of the given choices using phonetic and "
    "fuzzy reasoning. "
    "If the text does not reasonably match any choice, return decision=no_match. "
    "Return compact JSON only — no extra text, no markdown."
)

# Output schema embedded in every payload so GPT sees it near the instruction.
_OUTPUT_SCHEMA = (
    '{"decision":"select_option|no_match",'
    '"selected_names":["exact choice name from choices list"],'
    '"confidence":0.0,'
    '"reason_code":"exact_match|phonetic_match|fuzzy_match|no_match"}'
)

# Safety caps — never send more than these to GPT.
MAX_CHOICES = 20
MAX_HISTORY_TURNS = 3
MAX_LOCAL_SLOTS = 4
MAX_TOP_INTENTS = 3


class GptOptionContextBuilder:
    """Build [system, user] message pairs for the GPT option resolver.

    The user message is a compact JSON object containing only the information
    GPT needs to resolve a modifier option name — no full menu, no full cart.

    Payload shape
    -------------
    {
      "t": "select_modifier",
      "item": "<item_name>",            // current item being assembled
      "group": "<group_name>",          // e.g. "Cheese", "Sauce"
      "text": "<user utterance>",       // normalized user text
      "choices": ["..."],               // allowed option names (capped at MAX_CHOICES)
      "selected": ["..."],              // already-selected names (optional)
      "history": [["bot|user", "…"]],  // recent turns (optional, capped at 3)
      "last_prompt": "ask_for_modifier", // last bot response key (optional)
      "local_slots": [{"n":"SLOT","v":"val"},...],  // local NLU slots (optional)
      "top_intents": [{"i":"intent","c":0.9},...],  // top-K local intents (optional)
      "schema": "…"                     // output schema (always last)
    }
    """

    def build_messages(
        self,
        *,
        user_text: str,
        item_name: str,
        group_name: str,
        choice_names: list[str],
        already_selected_names: list[str] | None = None,
        previous_turns: list[tuple[str, str]] | None = None,
        last_response_key: str | None = None,
        local_slots: list[dict[str, Any]] | None = None,
        top_intents: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """Return the [system, user] messages list for OpenAI chat completions.

        Parameters
        ----------
        user_text:
            The normalized customer utterance for this turn.
        item_name:
            The name of the item currently being assembled (e.g. "Cheeseburger").
        group_name:
            The name of the modifier group being resolved (e.g. "Cheese").
        choice_names:
            All allowed modifier option names in the current group.
        already_selected_names:
            Names of options the user has already selected in this group (optional).
        previous_turns:
            Recent bot/user turn pairs [(role, text), …] capped to last 3 (optional).
        last_response_key:
            The last bot response key (e.g. "ask_for_modifier") for context (optional).
            Only the key label is sent — never the rendered response text.
        local_slots:
            Local NLU slot values from the current turn (optional, capped at 4).
            Each dict should have {"n": slot_name, "v": slot_value}.
        top_intents:
            Top-K local NLU intent candidates (optional, capped at 3).
            Each dict should have {"i": intent_label, "c": confidence_float}.
        """
        payload: dict[str, Any] = {
            "t": "select_modifier",
            "item": item_name,
            "group": group_name,
            "text": user_text,
            "choices": choice_names[:MAX_CHOICES],
        }

        if already_selected_names:
            payload["selected"] = already_selected_names

        if previous_turns:
            capped = list(previous_turns[-MAX_HISTORY_TURNS:])
            payload["history"] = [[role, text] for role, text in capped]

        # Last bot prompt key (enum-safe label only — never response text).
        if last_response_key:
            payload["last_prompt"] = last_response_key

        # Local NLU snapshot — gives GPT context about what the model thought.
        if local_slots:
            payload["local_slots"] = local_slots[:MAX_LOCAL_SLOTS]
        if top_intents:
            payload["top_intents"] = top_intents[:MAX_TOP_INTENTS]

        # Output schema always last so GPT sees it close to the instruction.
        payload["schema"] = _OUTPUT_SCHEMA

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    @staticmethod
    def extract_choice_names(group: "PendingModifierGroup") -> list[str]:
        """Return the canonical option names for the given modifier group."""
        return [choice.name for choice in group.choices if choice.name]
