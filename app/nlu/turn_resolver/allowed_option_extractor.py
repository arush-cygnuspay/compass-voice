# app/nlu/turn_resolver/allowed_option_extractor.py
"""Extracts allowed options from pending item context for waiting states.

Returns compact, JSON-safe dicts — no full menu dumps, no payment data.
Always returns an empty tuple if context is incomplete or an error occurs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.state_machine.models.conversation_context import ConversationContext

_MAX_OPTIONS = 12


class AllowedOptionExtractor:
    """Extracts the current group's options from ConversationContext for GPT context.

    Handles waiting_for_modifier, waiting_for_side, waiting_for_size,
    waiting_for_side_size, and waiting_for_order_type states.
    Never raises — returns empty tuple on any error or missing context.
    """

    def extract(
        self,
        context: "ConversationContext",
        state: str,
    ) -> tuple[dict, ...]:
        """Return allowed option dicts for the given *state*.

        Each dict contains only JSON-safe primitives:
        no UUIDs objects, no pricing, no payment data.
        """
        try:
            key = (state or "").strip().lower()
            if key == "waiting_for_modifier":
                return self._extract_modifier_options(context)
            if key == "waiting_for_side":
                return self._extract_side_options(context)
            if key in ("waiting_for_size", "waiting_for_side_size"):
                return self._extract_size_options(context, key)
            if key == "waiting_for_order_type":
                return self._extract_order_type_options()
            return ()
        except Exception:
            return ()

    # ── Modifier options ──────────────────────────────────────────────────────

    def _extract_modifier_options(
        self,
        context: "ConversationContext",
    ) -> tuple[dict, ...]:
        pending = getattr(context, "pending_add_item", None)
        if pending is None:
            return ()
        groups = getattr(pending, "modifier_groups", None) or []
        idx = int(getattr(context, "current_modifier_group_index", 0) or 0)
        if idx >= len(groups):
            return ()
        group = groups[idx]
        choices = getattr(group, "choices", None) or []
        out: list[dict] = []
        for i, choice in enumerate(choices[:_MAX_OPTIONS]):
            entry: dict = {
                "index": i,
                "modifier_id": str(getattr(choice, "modifier_id", "") or ""),
                "name": str(getattr(choice, "name", "") or ""),
                "group_id": str(getattr(group, "group_id", "") or ""),
                "group_name": str(getattr(group, "name", "") or ""),
            }
            aliases = getattr(choice, "match_texts", None)
            if aliases:
                entry["aliases"] = [str(a) for a in aliases[:6]]
            out.append(entry)
        return tuple(out)

    # ── Side options ──────────────────────────────────────────────────────────

    def _extract_side_options(
        self,
        context: "ConversationContext",
    ) -> tuple[dict, ...]:
        pending = getattr(context, "pending_add_item", None)
        if pending is None:
            return ()
        groups = getattr(pending, "side_groups", None) or []
        idx = int(getattr(context, "current_side_group_index", 0) or 0)
        if idx >= len(groups):
            return ()
        group = groups[idx]
        choices = getattr(group, "choices", None) or []
        out: list[dict] = []
        for i, choice in enumerate(choices[:_MAX_OPTIONS]):
            entry: dict = {
                "index": i,
                "item_id": str(getattr(choice, "item_id", "") or ""),
                "name": str(getattr(choice, "name", "") or ""),
                "group_id": str(getattr(group, "group_id", "") or ""),
                "group_name": str(getattr(group, "name", "") or ""),
            }
            aliases = getattr(choice, "match_texts", None)
            if aliases:
                entry["aliases"] = [str(a) for a in aliases[:6]]
            # Include top variant names if side has sizes
            top_variants = getattr(choice, "top_variant_names", None)
            if top_variants:
                entry["size_variants"] = list(top_variants[:4])
            out.append(entry)
        return tuple(out)

    # ── Size / variant options ────────────────────────────────────────────────

    def _extract_size_options(
        self,
        context: "ConversationContext",
        state_key: str,
    ) -> tuple[dict, ...]:
        pending = getattr(context, "pending_add_item", None)
        if pending is None:
            return ()

        if state_key == "waiting_for_side_size":
            # Variants of the pending side item
            pending_side_id = getattr(context, "pending_side_item_id", None)
            if not pending_side_id:
                return ()
            side_choice_map = getattr(pending, "side_choice_by_item_id", {}) or {}
            side_choice = side_choice_map.get(pending_side_id)
            if side_choice is None:
                return ()
            variants = getattr(side_choice, "variants", None) or []
        else:
            # Top-level item variants (waiting_for_size)
            variants = getattr(pending, "item_variants", None) or []

        out: list[dict] = []
        for i, variant in enumerate(variants[:_MAX_OPTIONS]):
            out.append({
                "index": i,
                "variant_id": str(getattr(variant, "variant_id", "") or ""),
                "name": str(getattr(variant, "name", "") or ""),
            })
        return tuple(out)

    # ── Order type options ────────────────────────────────────────────────────

    @staticmethod
    def _extract_order_type_options() -> tuple[dict, ...]:
        return (
            {"index": 0, "name": "pickup", "description": "Customer picks up the order."},
            {"index": 1, "name": "delivery", "description": "Order is delivered to the customer."},
        )
