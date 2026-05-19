# app/nlu/turn_resolver/gpt_context_builder.py
"""Compact, PII-safe GPT context packet builder for the turn-resolution layer.

Builds a structured dict suitable for GPT prompt construction.
No GPT API calls here — pure context assembly.

Safety contract
---------------
* NEVER includes: API keys, payment links, card data, raw phone numbers,
  full menu JSON, full cart raw JSON, or full conversation history.
* Cart: item count + item names only (no prices, IDs, or payment state).
* Previous turns: capped at 6 entries total.
* Menu candidates: capped at 12 entries.
* Allowed options: capped at 12 entries.
* All string fields are sanitised before inclusion.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.nlu.turn_resolver.allowed_intent_provider import AllowedIntentProvider
from app.nlu.turn_resolver.allowed_response_key_provider import AllowedResponseKeyProvider
from app.nlu.turn_resolver.allowed_option_extractor import AllowedOptionExtractor

if TYPE_CHECKING:
    from app.state_machine.models.conversation_context import ConversationContext
    from app.state_machine.models.turn_memory import TurnMemoryEntry

_MAX_PREV_TURNS = 6
_MAX_MENU_CANDIDATES = 12
_MAX_ALLOWED_OPTIONS = 12
_MAX_CART_NAMES = 15
_MAX_TEXT_LEN = 512

# ── PII sanitizer (lightweight, no external dependency) ──────────────────────

_PAYMENT_LINK_RE = re.compile(r"https?://\S*(?:pay|checkout)\S*", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(
    r"\b\d{7,}\b"
    r"|\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b",
    re.ASCII,
)
_ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


class GptContextBuilder:
    """Builds compact, PII-safe GPT context dicts for all task modes.

    Parameters
    ----------
    intent_provider:
        Provides allowed intents per state. Defaults to AllowedIntentProvider().
    response_key_provider:
        Provides allowed response keys per state. Defaults to AllowedResponseKeyProvider().
    option_extractor:
        Extracts waiting-state options from context. Defaults to AllowedOptionExtractor().
    """

    def __init__(
        self,
        intent_provider: AllowedIntentProvider | None = None,
        response_key_provider: AllowedResponseKeyProvider | None = None,
        option_extractor: AllowedOptionExtractor | None = None,
    ) -> None:
        self._intents = intent_provider or AllowedIntentProvider()
        self._response_keys = response_key_provider or AllowedResponseKeyProvider()
        self._options = option_extractor or AllowedOptionExtractor()

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        context: "ConversationContext",
        user_text: str,
        normalized_text: str | None,
        local_intent: str | None,
        local_confidence: float | None,
        local_candidates: list | tuple | None,
        local_slots: list | tuple | None,
        task_mode: str,
        state: str | None = None,
        menu_candidates: list | tuple | None = None,
        allowed_options: list | tuple | None = None,
        cart_item_names: tuple[str, ...] | None = None,
    ) -> dict:
        """Build and return a PII-safe GPT context dict.

        Parameters
        ----------
        context:
            Current ConversationContext (not the full session).
        user_text:
            Raw user utterance for this turn.
        normalized_text:
            Preprocessed/normalized user utterance.
        local_intent:
            Effective intent string from local NLU.
        local_confidence:
            Intent confidence from local NLU (0.0–1.0).
        local_candidates:
            Top-K intent candidates from local NLU.
        local_slots:
            Slot values from local NLU.
        task_mode:
            One of the TASK_* constants from prompt_registry.py.
        state:
            Current conversation state value string.
        menu_candidates:
            Menu item candidates (for idle item resolution). Capped at 12.
        allowed_options:
            Pre-built option list for waiting states. If None, extracted from context.
        cart_item_names:
            Item names from cart (names only, no prices). Capped at 15.

        Returns
        -------
        dict — JSON-serialisable. Safe to pass to prompt builder or logger.
        """
        current_state = (state or "").strip().lower()

        # Options: use caller-supplied list or extract from context
        if allowed_options is None:
            allowed_options = self._options.extract(context, current_state)

        packet = {
            "task_mode": task_mode,
            "current_state": current_state,
            "user_text": self._sanitize_text(user_text),
            "normalized_text": self._sanitize_text(normalized_text),
            "previous_turns": self._build_previous_turns(context),
            "previous_assistant_prompt": self._get_previous_assistant_prompt(context),
            "local_intent": local_intent,
            "local_confidence": (
                round(float(local_confidence), 4) if local_confidence is not None else None
            ),
            "local_candidates": self._build_candidates(local_candidates),
            "local_slots": self._build_slots(local_slots),
            "allowed_intents": [
                {"name": ai.name, "description": ai.description}
                for ai in self._intents.get_allowed_intents_for_state(current_state, context)
            ],
            "allowed_response_keys": list(
                self._response_keys.get_allowed_response_keys_for_state(current_state, context)
            ),
            "pending_item": self._build_pending_item_summary(context),
            "pending_group": self._build_pending_group_summary(context, current_state),
            "allowed_options": list(allowed_options[:_MAX_ALLOWED_OPTIONS]),
            "cart": self._build_cart_summary(context, cart_item_names),
            "order_type": getattr(context, "order_type", None),
        }

        if menu_candidates:
            packet["menu_candidates"] = self._build_menu_candidates(menu_candidates)

        return packet

    def build_metadata(self, packet: dict) -> dict:
        """Return safe logging metadata about a built context packet.

        Contains only counts and boolean flags — no user text or content.
        """
        return {
            "gpt_context_built": True,
            "gpt_task_mode": packet.get("task_mode"),
            "gpt_context_previous_turn_count": len(packet.get("previous_turns") or []),
            "gpt_context_allowed_intents_count": len(packet.get("allowed_intents") or []),
            "gpt_context_allowed_response_keys_count": len(packet.get("allowed_response_keys") or []),
            "gpt_context_allowed_options_count": len(packet.get("allowed_options") or []),
            "gpt_context_menu_candidates_count": len(packet.get("menu_candidates") or []),
            "gpt_context_pending_item_present": packet.get("pending_item") is not None,
            "gpt_context_pending_group_present": packet.get("pending_group") is not None,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sanitize_text(self, text: str | None) -> str | None:
        if not text:
            return text
        text = text[:_MAX_TEXT_LEN]
        text = _PAYMENT_LINK_RE.sub("[REDACTED_URL]", text)
        text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        if not _ISO_TS_RE.match(text):
            text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
        return text

    def _build_previous_turns(self, context: "ConversationContext") -> list[dict]:
        """Return last N turns as [{"role": ..., "text": ...}] dicts."""
        getter = getattr(context, "get_turn_memory_entries", None)
        if callable(getter):
            entries: tuple["TurnMemoryEntry", ...] = getter(_MAX_PREV_TURNS)
            return [
                {
                    "role": e.role,
                    "text": self._sanitize_text(e.text) or "",
                    **({"state": e.state} if e.state else {}),
                    **({"intent": e.intent} if e.intent else {}),
                    **({"response_key": e.response_key} if e.response_key else {}),
                }
                for e in entries
                if e.text and e.text.strip()
            ]
        # Fallback: legacy (role, text) tuple API
        getter2 = getattr(context, "get_turn_memory", None)
        if callable(getter2):
            raw = getter2(_MAX_PREV_TURNS)
            return [
                {"role": str(r), "text": self._sanitize_text(str(t)) or ""}
                for r, t in raw
                if t and str(t).strip()
            ]
        return []

    def _get_previous_assistant_prompt(self, context: "ConversationContext") -> str | None:
        """Extract the most recent assistant turn's response_key as a prompt hint."""
        getter = getattr(context, "get_turn_memory_entries", None)
        if callable(getter):
            entries: tuple["TurnMemoryEntry", ...] = getter(_MAX_PREV_TURNS)
            for entry in reversed(entries):
                if entry.role == "assistant" and entry.response_key:
                    return entry.response_key
        return None

    def _build_pending_item_summary(self, context: "ConversationContext") -> dict | None:
        """Compact pending item summary: name, variant count, group counts."""
        item_id = getattr(context, "current_item_id", None)
        item_name = getattr(context, "current_item_name", None)
        if not item_name and not item_id:
            return None
        summary: dict[str, Any] = {
            "item_id": str(item_id or ""),
            "item_name": str(item_name or ""),
        }
        pending = getattr(context, "pending_add_item", None)
        if pending is not None:
            mod_groups = getattr(pending, "modifier_groups", None) or []
            side_groups = getattr(pending, "side_groups", None) or []
            variants = getattr(pending, "item_variants", None) or []
            summary["modifier_group_count"] = len(mod_groups)
            summary["side_group_count"] = len(side_groups)
            summary["variant_count"] = len(variants)
        qty = getattr(context, "quantity", None)
        if qty is not None:
            summary["quantity"] = qty
        return summary

    def _build_pending_group_summary(
        self,
        context: "ConversationContext",
        state_key: str,
    ) -> dict | None:
        """Compact summary of the active modifier/side group being resolved."""
        pending = getattr(context, "pending_add_item", None)
        if pending is None:
            return None
        if "modifier" in state_key:
            groups = getattr(pending, "modifier_groups", None) or []
            idx = int(getattr(context, "current_modifier_group_index", 0) or 0)
            if idx >= len(groups):
                return None
            g = groups[idx]
            return {
                "type": "modifier",
                "group_id": str(getattr(g, "group_id", "") or ""),
                "group_name": str(getattr(g, "name", "") or ""),
                "is_required": bool(getattr(g, "is_required", False)),
                "choice_count": len(getattr(g, "choices", None) or []),
            }
        if "side" in state_key:
            groups = getattr(pending, "side_groups", None) or []
            idx = int(getattr(context, "current_side_group_index", 0) or 0)
            if idx >= len(groups):
                return None
            g = groups[idx]
            return {
                "type": "side",
                "group_id": str(getattr(g, "group_id", "") or ""),
                "group_name": str(getattr(g, "name", "") or ""),
                "is_required": bool(getattr(g, "is_required", False)),
                "choice_count": len(getattr(g, "choices", None) or []),
            }
        return None

    def _build_cart_summary(
        self,
        context: "ConversationContext",
        cart_item_names: tuple[str, ...] | None,
    ) -> dict:
        """Compact cart: item count + names only. No prices, no payment data."""
        if cart_item_names is None:
            cart_item_names = ()
        capped = cart_item_names[:_MAX_CART_NAMES]
        return {
            "item_count": len(capped),
            "item_names": list(capped),
        }

    def _build_candidates(
        self,
        candidates: list | tuple | None,
    ) -> list[dict]:
        """Convert top-K intent candidates to compact dicts."""
        if not candidates:
            return []
        out = []
        for c in candidates[:8]:
            if isinstance(c, dict):
                out.append({"intent": str(c.get("intent", "")), "confidence": c.get("confidence")})
            else:
                intent = getattr(c, "canonical_intent", None) or getattr(c, "intent", None)
                confidence = getattr(c, "confidence", None)
                if intent:
                    out.append({
                        "intent": str(intent),
                        "confidence": round(float(confidence), 4) if confidence is not None else None,
                    })
        return out

    def _build_slots(self, slots: list | tuple | None) -> list[dict]:
        """Convert slot values to compact {"name": ..., "value": ...} dicts."""
        if not slots:
            return []
        out = []
        for s in slots:
            if isinstance(s, dict):
                out.append({"name": str(s.get("name", "")), "value": str(s.get("value", ""))})
            else:
                name = getattr(s, "name", None)
                value = getattr(s, "value", None)
                if name and value is not None:
                    out.append({"name": str(name), "value": str(value)})
        return out

    def _build_menu_candidates(self, candidates: list | tuple) -> list[dict]:
        """Convert menu candidates to compact name-only dicts. Capped at 12."""
        out = []
        for c in candidates[:_MAX_MENU_CANDIDATES]:
            if isinstance(c, str):
                out.append({"name": c})
            elif isinstance(c, dict):
                out.append({
                    "name": str(c.get("name", c.get("item_name", ""))),
                    **({"item_id": str(c["item_id"])} if "item_id" in c else {}),
                })
            else:
                name = getattr(c, "name", None) or getattr(c, "item_name", None)
                if name:
                    item_id = getattr(c, "item_id", None)
                    entry: dict = {"name": str(name)}
                    if item_id:
                        entry["item_id"] = str(item_id)
                    out.append(entry)
        return out
