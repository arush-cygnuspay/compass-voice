# app/menu/store.py

from __future__ import annotations

import json
from pathlib import Path
from difflib import SequenceMatcher
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


def _norm(text: str) -> str:
    return text.lower().strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class MenuStore:
    """
    Immutable, in-memory menu store.
    Loads menu.json + entity_index.json and parses them into models.
    """

    def __init__(self, menu_path: Path, entity_index_path: Path):
        self.menu_path = menu_path
        self.entity_index_path = entity_index_path

        # Canonical data (MODELS)
        self.items: Dict[str, MenuItem] = {}
        self.categories: Dict[str, dict] = {}
        self.entity_index: Dict[str, List[dict]] = {}

        # Runtime indexes
        self._item_by_name: Dict[str, MenuItem] = {}
        self._item_tokens: Dict[str, Set[str]] = {}
        self._items_by_category: Dict[str, List[MenuItem]] = {}

        self._load()

    # =================================================
    # Load & Parse
    # =================================================

    def _load(self) -> None:
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
        parsed = []

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
        parsed = []

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
        self._item_by_name.clear()
        self._item_tokens.clear()
        self._items_by_category.clear()

        for item in self.items.values():
            key = _norm(item.name)
            self._item_by_name[key] = item
            self._item_tokens[key] = set(key.split())

        for item in self.items.values():
            for cat_id in self.categories:
                if cat_id in self.categories:
                    self._items_by_category.setdefault(cat_id, []).append(item)

    # =================================================
    # Queries
    # =================================================

    def get_item(self, item_id: str) -> MenuItem:
        item = self.items.get(item_id)
        if not item:
            raise KeyError(f"Item not found: {item_id}")
        return item

    def find_entity(self,key: str,*,allowed_types: set[str] | None = None,parent_item_id: str | None = None,) -> List[dict]:

        key = _norm(key)
        raw = self.entity_index.get(key)
        if not raw:
            return []

        entries = raw if isinstance(raw, list) else [raw]
        results = []

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


    def find_item_by_tokens(self, text: str) -> Optional[MenuItem]:
        """
        Token overlap heuristic.
        Deterministic, cheap fallback after entity index.
        """
        tokens = set(_norm(text).split())
        best_item: Optional[MenuItem] = None
        best_score = 0

        for name, name_tokens in self._item_tokens.items():
            overlap = len(tokens & name_tokens)
            if overlap > best_score:
                best_score = overlap
                best_item = self._item_by_name[name]

        return best_item if best_score > 0 else None

    def find_item_exact(self, name: str) -> Optional[MenuItem]:
        return self._item_by_name.get(_norm(name))

