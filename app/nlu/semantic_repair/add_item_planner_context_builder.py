# app/nlu/semantic_repair/add_item_planner_context_builder.py
"""Phase 4 GPT Add-Item Planner — compact context/payload builder.

Safety contract
---------------
The payload must NEVER contain:
  * Full menu data or the item catalogue
  * Full cart JSON (compact name list only — no prices, no raw JSON)
  * API keys or any PII (phone numbers, addresses, payment links)
  * Prices or tax figures
  * Other modifier/side groups not relevant to the candidate items
  * More than MAX_ITEM_CANDIDATES items, MAX_OPTION_CANDIDATES options per group

Candidate items are passed in pre-resolved by the caller (planner service).
The builder does not do menu lookups — it only serialises what it receives.
"""
from __future__ import annotations

import json
from typing import Any

# System prompt — compact and action-focused.
_SYSTEM_PROMPT = (
    "You parse complex restaurant add-item utterances into structured item plans. "
    "Use only candidate items and options listed in the payload. "
    "If an entity doesn't match any candidate, put it in unresolved[]. "
    "Return compact JSON only — no extra text, no markdown."
)

# Output schema embedded in every payload.
_OUTPUT_SCHEMA = (
    '{"decision":"add_items|clarify|no_repair|unclear",'
    '"items":['
    '{"candidate_item_id":"string|null","item_name":"string","quantity":1,'
    '"size":"string|null","variant":"string|null",'
    '"modifiers":[{"name":"string","operation":"add|remove|extra|light","quantity":1}],'
    '"sides":[{"name":"string","quantity":1,"size":"string|null"}],'
    '"special_instructions":"string|null"}],'
    '"unresolved":[{"text":"string","reason":"not_on_menu|ambiguous|belongs_to_unknown_group|unsupported"}],'
    '"confidence":0.0,'
    '"reason_code":"complex_with_phrase|multi_item|slot_grouping_repair|unknown_with_item_evidence|unclear",'
    '"safe_to_apply":false}'
)

# Safety caps — never send more than these to GPT.
MAX_ITEM_CANDIDATES: int = 10
MAX_OPTION_CANDIDATES: int = 20
MAX_HISTORY_TURNS: int = 3
MAX_TOP_K: int = 4
MAX_LOCAL_SLOTS: int = 6
MAX_CART_ITEMS: int = 10


class GptAddItemPlannerContextBuilder:
    """Build [system, user] message pairs for the GPT add-item planner.

    The user message is a compact JSON object. The caller (planner service)
    is responsible for resolving candidate menu items from the menu store and
    passing them as ``candidate_items``.

    Payload shape
    -------------
    {
      "t": "add_item_plan",
      "text": "<normalized user utterance>",
      "local": {
        "intent": "<intent_label>",
        "conf": 0.0,
        "top_k": [{"i": "...", "c": 0.0}, ...],
        "slots": [{"n": "ITEM", "v": "burger"}, ...]
      },
      "candidates": [
        {
          "id": "<candidate_item_id>",
          "name": "<item_name>",
          "sizes": ["Regular", "Large"],
          "modifier_groups": [
            {"name": "Cheese", "choices": ["American Cheese", "Swiss Cheese"]}
          ],
          "side_groups": [
            {"name": "Drink", "choices": ["Coke", "Sprite", "Water"]}
          ]
        }
      ],
      "cart": {"n": 2, "items": ["Pizza", "Wings"]},   // compact only
      "history": [["bot", "..."], ["user", "..."]],
      "rules": "...",
      "schema": "..."
    }
    """

    def build_messages(
        self,
        *,
        user_text: str,
        local_intent: str | None = None,
        local_confidence: float = 0.0,
        top_k_intents: list[dict[str, Any]] | None = None,
        local_slots: list[dict[str, Any]] | None = None,
        candidate_items: list[dict[str, Any]] | None = None,
        cart_item_names: list[str] | None = None,
        previous_turns: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Return the [system, user] messages list for OpenAI chat completions.

        Parameters
        ----------
        user_text:
            Normalized customer utterance.
        local_intent:
            Local NLU intent label (string, e.g. "add_item").
        local_confidence:
            Local NLU intent confidence (0.0–1.0).
        top_k_intents:
            Top-K local NLU candidates; each {"i": label, "c": conf}.
            Capped to MAX_TOP_K.
        local_slots:
            Local NLU slot values; each {"n": slot_name, "v": value}.
            Capped to MAX_LOCAL_SLOTS.
        candidate_items:
            Pre-resolved menu item candidates from the menu store.
            Each entry is a dict with keys:
              id         — menu item ID (string)
              name       — display name (string)
              sizes      — list of size/variant labels (list[str], may be empty)
              modifier_groups — list of {name, choices[]} (list[dict])
              side_groups     — list of {name, choices[]} (list[dict])
            Capped to MAX_ITEM_CANDIDATES.
            Options within each group capped to MAX_OPTION_CANDIDATES.
        cart_item_names:
            Current cart item names (compact — no prices, no raw JSON).
            Capped to MAX_CART_ITEMS.
        previous_turns:
            Recent (role, text) pairs capped to MAX_HISTORY_TURNS.
        """
        payload: dict[str, Any] = {
            "t": "add_item_plan",
            "text": user_text,
        }

        # Local NLU snapshot
        local_block: dict[str, Any] = {
            "intent": local_intent or "",
            "conf": round(max(0.0, min(1.0, float(local_confidence))), 4),
        }
        if top_k_intents:
            local_block["top_k"] = top_k_intents[:MAX_TOP_K]
        if local_slots:
            local_block["slots"] = local_slots[:MAX_LOCAL_SLOTS]
        payload["local"] = local_block

        # Candidate items (capped + options capped)
        if candidate_items:
            capped: list[dict[str, Any]] = []
            for item in candidate_items[:MAX_ITEM_CANDIDATES]:
                entry: dict[str, Any] = {
                    "id": item.get("id") or item.get("item_id") or "",
                    "name": item.get("name") or "",
                }
                sizes = item.get("sizes") or []
                if sizes:
                    entry["sizes"] = sizes

                mod_groups = item.get("modifier_groups") or []
                if mod_groups:
                    entry["modifier_groups"] = [
                        {
                            "name": mg.get("name", ""),
                            "choices": (mg.get("choices") or [])[:MAX_OPTION_CANDIDATES],
                        }
                        for mg in mod_groups
                    ]

                side_groups = item.get("side_groups") or []
                if side_groups:
                    entry["side_groups"] = [
                        {
                            "name": sg.get("name", ""),
                            "choices": (sg.get("choices") or [])[:MAX_OPTION_CANDIDATES],
                        }
                        for sg in side_groups
                    ]

                capped.append(entry)
            payload["candidates"] = capped

        # Compact cart summary — names and count only (no prices)
        if cart_item_names:
            cap_cart = cart_item_names[:MAX_CART_ITEMS]
            payload["cart"] = {"n": len(cart_item_names), "items": cap_cart}

        # Recent conversation history (bot/user pairs, capped)
        if previous_turns:
            capped_history = list(previous_turns[-MAX_HISTORY_TURNS:])
            payload["history"] = [[role, text] for role, text in capped_history]

        # Size/variant scoping rule reminder
        payload["rules"] = (
            "Attach size to the specific entity named. "
            "'large coke with burger' → coke.size=large, burger.size=null. "
            "Use only options listed in candidates. "
            "Unknown entities go in unresolved[]."
        )

        # Output schema always last so GPT sees it close to the instruction.
        payload["schema"] = _OUTPUT_SCHEMA

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    @staticmethod
    def build_candidate_from_menu_item(
        menu_item: Any,
        *,
        max_options: int = MAX_OPTION_CANDIDATES,
    ) -> dict[str, Any]:
        """Convert a MenuItem (from app.menu.models) to a candidate dict.

        Extracts: id, name, sizes, modifier_groups, side_groups.
        Applies max_options cap per group.
        Does NOT include prices or full enum data.

        Parameters
        ----------
        menu_item:
            A MenuItem instance from the live menu store.
        max_options:
            Maximum option names per modifier/side group.
        """
        entry: dict[str, Any] = {
            "id": getattr(menu_item, "item_id", "") or "",
            "name": getattr(menu_item, "name", "") or "",
        }

        # Sizes from variant pricing
        pricing = getattr(menu_item, "pricing", None)
        if pricing is not None and getattr(pricing, "mode", "") == "variant":
            variants = getattr(pricing, "variants", ()) or ()
            sizes = [getattr(v, "label", "") for v in variants if getattr(v, "label", "")]
            if sizes:
                entry["sizes"] = sizes

        # Modifier groups
        mod_groups_raw = getattr(menu_item, "modifier_groups", ()) or ()
        if mod_groups_raw:
            mod_groups: list[dict[str, Any]] = []
            for mg in mod_groups_raw:
                choices_raw = getattr(mg, "choices", ()) or ()
                choice_names = [
                    getattr(c, "name", "") for c in choices_raw
                    if getattr(c, "name", "")
                ][:max_options]
                mod_groups.append({
                    "name": getattr(mg, "name", "") or "",
                    "choices": choice_names,
                })
            entry["modifier_groups"] = mod_groups

        # Side groups
        side_groups_raw = getattr(menu_item, "side_groups", ()) or ()
        if side_groups_raw:
            side_groups: list[dict[str, Any]] = []
            for sg in side_groups_raw:
                choices_raw = getattr(sg, "choices", ()) or ()
                choice_names = [
                    getattr(c, "name", "") or getattr(c, "item_name", "") or ""
                    for c in choices_raw
                ]
                choice_names = [n for n in choice_names if n][:max_options]
                side_groups.append({
                    "name": getattr(sg, "name", "") or "",
                    "choices": choice_names,
                })
            entry["side_groups"] = side_groups

        return entry
