# app/menu/store.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.menu.exceptions import MenuLoadError
from app.menu.models import (
    MenuItem,
    Pricing,
    PricingVariant,
    SideGroup,
    SideChoice,
    ModifierGroup,
    ModifierChoice,
)

# =========================================================
# Internal Helpers (LOW-LEVEL, DETERMINISTIC)
# =========================================================

def _norm(text: str) -> str:
    """
    Canonical normalization for menu indexing.
    NOTE:
    - Lowercase only
    - No punctuation removal
    - No stemming
    This function MUST remain cheap and deterministic.
    """
    return text.lower().strip()


def _token_score(query_tokens: set[str], item_tokens: set[str]) -> float:
    """
    Simple overlap ratio used ONLY for cheap fallback heuristics.

    IMPORTANT:
    - This is NOT a semantic score.
    - This MUST NOT be used for final decision-making.
    - Business-level resolution happens in MenuRepository.
    """
    if not item_tokens:
        return 0.0
    return len(query_tokens & item_tokens) / len(item_tokens)


# =========================================================
# MenuStore
# =========================================================

class MenuStore:
    """
    Immutable, in-memory menu store.

    Responsibilities:
    -----------------
    - Load menu.json and entity_index.json
    - Parse raw JSON into domain models
    - Build cheap runtime indexes
    - Provide deterministic, low-level lookup helpers

    NON-Responsibilities:
    ---------------------
    - Intent interpretation
    - Scoring strategies
    - Ambiguity resolution
    - Conversational logic

    All "which item should win?" decisions belong to MenuRepository.
    """

    def __init__(self, menu_path: Path, entity_index_path: Path):
        self.menu_path = menu_path
        self.entity_index_path = entity_index_path

        # Canonical parsed data
        self.items: Dict[str, MenuItem] = {}
        self.categories: Dict[str, dict] = {}
        self.entity_index: Dict[str, List[dict]] = {}

        # Runtime indexes (cheap, deterministic)
        self._item_by_name: Dict[str, MenuItem] = {}
        self._item_tokens: Dict[str, Set[str]] = {}

        self._load()

    # =================================================
    # Load & Parse
    # =================================================

    def _load(self) -> None:
        """
        Loads menu.json and entity_index.json and builds indexes.

        This method MUST:
        - Fail fast on malformed input
        - Leave MenuStore in a fully consistent state
        """
        try:
            with open(self.menu_path, "r", encoding="utf-8") as f:
                raw_menu = json.load(f)

            raw_items = raw_menu.get("items", {})
            self.categories = raw_menu.get("categories", {})

            if not raw_items:
                raise MenuLoadError("menu.json contains no items")

            with open(self.entity_index_path, "r", encoding="utf-8") as f:
                self.entity_index = json.load(f)

            self.items.clear()
            for item_id, raw_item in raw_items.items():
                self.items[item_id] = self._parse_menu_item(raw_item)

            self._build_indexes()

        except Exception as e:
            raise MenuLoadError(str(e)) from e

    # =================================================
    # Parsing Helpers
    # =================================================

    def _parse_menu_item(self, raw: dict) -> MenuItem:
        return MenuItem(
            item_id=raw["item_id"],
            name=raw["name"],
            aliases=raw.get("aliases", []),
            pricing=self._parse_pricing(raw["pricing"]),
            side_groups=self._parse_side_groups(raw.get("side_groups", [])),
            modifier_groups=self._parse_modifier_groups(raw.get("modifier_groups", [])),
            available=raw.get("available", True),
        )

    def _parse_pricing(self, raw: dict) -> Pricing:
        mode = raw["mode"]

        if mode == "fixed":
            return Pricing(
                mode="fixed",
                price_cents=raw["price_cents"],
                currency=raw.get("currency", "USD"),
            )

        if mode == "variant":
            variants = [
                PricingVariant(
                    variant_id=v["variant_id"],
                    label=v["label"],
                    price_cents=v["price_cents"],
                )
                for v in raw.get("variants", [])
            ]
            return Pricing(
                mode="variant",
                variants=variants,
                currency=raw.get("currency", "USD"),
            )

        if mode == "unit":
            return Pricing(
                mode="unit",
                price_cents=raw["price_cents"],
                currency=raw.get("currency", "USD"),
            )

        raise MenuLoadError(f"Unknown pricing mode: {mode}")

    def _parse_side_groups(self, groups: List[dict]) -> List[SideGroup]:
        parsed: List[SideGroup] = []

        for g in groups:
            choices = [
                SideChoice(
                    item_id=c["item_id"],
                    name=c["name"],
                    pricing=self._parse_pricing(c["pricing"]),
                )
                for c in g.get("choices", [])
            ]

            parsed.append(
                SideGroup(
                    group_id=g["group_id"],
                    name=g["name"],
                    is_required=g["is_required"],
                    min_selector=g["min_selector"],
                    max_selector=g["max_selector"],
                    choices=choices,
                )
            )

        return parsed

    def _parse_modifier_groups(self, groups: List[dict]) -> List[ModifierGroup]:
        parsed: List[ModifierGroup] = []

        for g in groups:
            choices = [
                ModifierChoice(
                    modifier_id=c["modifier_id"],
                    name=c["name"],
                    price_cents=c["price_cents"],
                )
                for c in g.get("choices", [])
            ]

            parsed.append(
                ModifierGroup(
                    group_id=g["group_id"],
                    name=g["name"],
                    is_required=g["is_required"],
                    min_selector=g["min_selector"],
                    max_selector=g["max_selector"],
                    choices=choices,
                )
            )

        return parsed

    # =================================================
    # Indexes
    # =================================================

    def _build_indexes(self) -> None:
        """
        Builds cheap runtime indexes for deterministic lookup.

        Category normalization rules:
        - Singular / plural equivalence
        - Lowercase only
        - No stemming libraries (deterministic)
        """
        self._item_by_name.clear()
        self._item_tokens.clear()

        # -----------------------------
        # Item indexes (unchanged)
        # -----------------------------
        for item in self.items.values():
            key = _norm(item.name)
            self._item_by_name[key] = item
            self._item_tokens[key] = set(key.split())

        # -----------------------------
        # Category name index (NEW)
        # -----------------------------
        self._category_name_index: Dict[str, dict] = {}

        for cat in self.categories.values():
            name = cat["name"].lower().strip()

            # canonical
            self._category_name_index[name] = cat

            # plural ↔ singular normalization
            if name.endswith("s"):
                self._category_name_index[name[:-1]] = cat
            else:
                self._category_name_index[f"{name}s"] = cat

    # =================================================
    # Queries (LOW-LEVEL ONLY)
    # =================================================

    def get_item(self, item_id: str) -> MenuItem:
        """
        Fetch item by ID.
        This is a hard lookup and MUST raise if missing.
        """
        item = self.items.get(item_id)
        if not item:
            raise KeyError(f"Item not found: {item_id}")
        return item

    def find_entity(
        self,
        key: str,
        *,
        allowed_types: set[str] | None = None,
        parent_item_id: str | None = None,
    ) -> List[dict]:
        """
        Entity index lookup.

        Returns RAW entity index entries.
        DOES NOT rank, score, or select winners.
        """
        key = _norm(key)
        raw = self.entity_index.get(key)
        if not raw:
            return []

        entries = raw if isinstance(raw, list) else [raw]
        results: List[dict] = []

        for e in entries:
            etype = e.get("type")
            if not etype:
                continue

            if allowed_types and etype not in allowed_types:
                continue

            if parent_item_id:
                if (
                    e.get("item_id") != parent_item_id
                    and e.get("parent_item_id") != parent_item_id
                ):
                    continue

            results.append(e)

        return results

    def find_item_exact(self, name: str) -> Optional[MenuItem]:
        """
        Exact name match (after normalization).
        """
        return self._item_by_name.get(_norm(name))

    def find_item_by_tokens(self, text: str) -> Optional[MenuItem]:
        """
        Token-overlap fallback heuristic.

        WARNING:
        - This is intentionally weak.
        - Only used as a LAST RESORT.
        - Threshold prevents garbage matches.
        """
        query_tokens = set(_norm(text).split())

        best_item: Optional[MenuItem] = None
        best_score = 0.0

        for name, item_tokens in self._item_tokens.items():
            score = _token_score(query_tokens, item_tokens)
            if score > best_score:
                best_score = score
                best_item = self._item_by_name[name]

        return best_item if best_score >= 0.6 else None

    def find_category_by_name(self, text: str) -> Optional[dict]:
        """
        Resolve category by normalized name with plural tolerance.

        Deterministic:
        - No fuzzy guessing
        - No scoring
        """
        key = _norm(text)
        return self._category_name_index.get(key)