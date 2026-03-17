# app/menu/store.py
from __future__ import annotations

import json
from pathlib import Path

from app.menu.exceptions import MenuLoadError
from app.menu.models import (
    MenuItem,
    ModifierChoice,
    ModifierGroup,
    Pricing,
    PricingVariant,
    SideChoice,
    SideGroup,
)
from app.nlu.query_normalization.text_preprocessor import normalize_text


class MenuStore:
    """
    Immutable, in-memory menu store.

    Responsibilities:
    - Load menu.json and entity_index.json
    - Parse raw JSON into domain models
    - Build cheap runtime indexes
    - Provide deterministic low-level lookup helpers

    Non-responsibilities:
    - Intent interpretation
    - Ambiguity resolution
    - Conversational routing
    - Winner selection logic
    """

    def __init__(self, menu_path: Path, entity_index_path: Path):
        self.menu_path = menu_path
        self.entity_index_path = entity_index_path

        self.items: dict[str, MenuItem] = {}
        self.categories: dict[str, dict] = {}
        self.entity_index: dict[str, list[dict]] = {}

        self._item_by_name: dict[str, MenuItem] = {}
        self._item_ids_by_alias: dict[str, list[str]] = {}
        self._category_name_index: dict[str, dict] = {}

        self._load()

    def _load(self) -> None:
        """
        Load menu and entity index, then build runtime indexes.

        This method must fail fast on malformed input and leave the store
        in a fully consistent state.
        """
        try:
            with open(self.menu_path, "r", encoding="utf-8") as f:
                raw_menu = json.load(f)

            raw_items = raw_menu.get("items", {})
            raw_categories = raw_menu.get("categories", {})

            if not raw_items:
                raise MenuLoadError("menu.json contains no items")

            with open(self.entity_index_path, "r", encoding="utf-8") as f:
                raw_entity_index = json.load(f)

            self.items = {
                item_id: self._parse_menu_item(raw_item)
                for item_id, raw_item in raw_items.items()
            }
            self.categories = dict(raw_categories)
            self.entity_index = self._normalize_entity_index(raw_entity_index)

            self._build_indexes()

        except Exception as e:
            raise MenuLoadError(str(e)) from e

    def _normalize_entity_index(self, raw_entity_index: dict) -> dict[str, list[dict]]:
        normalized: dict[str, list[dict]] = {}

        for raw_key, raw_value in raw_entity_index.items():
            norm_key = normalize_text(str(raw_key))
            if not norm_key:
                continue

            entries = raw_value if isinstance(raw_value, list) else [raw_value]
            bucket = normalized.setdefault(norm_key, [])
            for entry in entries:
                if isinstance(entry, dict):
                    bucket.append(entry)

        return normalized

    def _parse_menu_item(self, raw: dict) -> MenuItem:
        aliases = tuple(str(alias) for alias in raw.get("aliases", []))
        normalized_aliases = tuple(
            norm_alias
            for alias in aliases
            if (norm_alias := normalize_text(alias))
        )

        return MenuItem(
            item_id=raw["item_id"],
            name=raw["name"],
            normalized_name=normalize_text(raw["name"]),
            aliases=aliases,
            normalized_aliases=normalized_aliases,
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
                    variant_id=variant["variant_id"],
                    label=variant["label"],
                    normalized_label=normalize_text(variant["label"]),
                    price_cents=variant["price_cents"],
                )
                for variant in raw.get("variants", [])
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

    def _parse_side_groups(self, groups: list[dict]) -> list[SideGroup]:
        parsed: list[SideGroup] = []

        for group in groups:
            choices = [
                SideChoice(
                    item_id=choice["item_id"],
                    name=choice["name"],
                    normalized_name=normalize_text(choice["name"]),
                    pricing=self._parse_pricing(choice["pricing"]),
                )
                for choice in group.get("choices", [])
            ]

            parsed.append(
                SideGroup(
                    group_id=group["group_id"],
                    name=group["name"],
                    normalized_name=normalize_text(group["name"]),
                    is_required=group["is_required"],
                    min_selector=group["min_selector"],
                    max_selector=group["max_selector"],
                    choices=choices,
                )
            )

        return parsed

    def _parse_modifier_groups(self, groups: list[dict]) -> list[ModifierGroup]:
        parsed: list[ModifierGroup] = []

        for group in groups:
            choices = [
                ModifierChoice(
                    modifier_id=choice["modifier_id"],
                    name=choice["name"],
                    normalized_name=normalize_text(choice["name"]),
                    price_cents=choice["price_cents"],
                )
                for choice in group.get("choices", [])
            ]

            parsed.append(
                ModifierGroup(
                    group_id=group["group_id"],
                    name=group["name"],
                    normalized_name=normalize_text(group["name"]),
                    is_required=group["is_required"],
                    min_selector=group["min_selector"],
                    max_selector=group["max_selector"],
                    choices=choices,
                )
            )

        return parsed

    def _build_indexes(self) -> None:
        """
        Build deterministic runtime indexes once at startup.

        Category index also supports singular/plural tolerance.
        """
        self._item_by_name.clear()
        self._item_ids_by_alias.clear()
        self._category_name_index.clear()

        for item in self.items.values():
            if item.normalized_name:
                self._item_by_name[item.normalized_name] = item

            for alias in item.normalized_aliases:
                self._item_ids_by_alias.setdefault(alias, []).append(item.item_id)

        for category in self.categories.values():
            category_name = str(category.get("name", ""))
            norm_name = normalize_text(category_name)
            if not norm_name:
                continue

            self._category_name_index[norm_name] = category

            if norm_name.endswith("s"):
                singular = norm_name[:-1]
                if singular:
                    self._category_name_index[singular] = category
            else:
                self._category_name_index[f"{norm_name}s"] = category

    def get_item(self, item_id: str) -> MenuItem:
        item = self.items.get(item_id)
        if item is None:
            raise KeyError(f"Item not found: {item_id}")
        return item

    def find_entity(
        self,
        key: str,
        *,
        allowed_types: set[str] | None = None,
        parent_item_id: str | None = None,
        group_id: str | None = None,
    ) -> list[dict]:
        raw_entries = self.entity_index.get(key)
        if not raw_entries:
            return []

        results: list[dict] = []

        for entry in raw_entries:
            entity_type = entry.get("type")
            if not entity_type:
                continue

            if allowed_types and entity_type not in allowed_types:
                continue

            if parent_item_id is not None and entry.get("parent_item_id") != parent_item_id:
                continue

            if group_id is not None and entry.get("group_id") != group_id:
                continue

            results.append(entry)

        return results

    def find_item_exact(self, normalized_name: str) -> MenuItem | None:
        return self._item_by_name.get(normalized_name)

    def find_item_ids_by_alias(self, normalized_alias: str) -> list[str]:
        return self._item_ids_by_alias.get(normalized_alias, [])

    def find_category_by_name(self, normalized_text: str) -> dict | None:
        return self._category_name_index.get(normalized_text)