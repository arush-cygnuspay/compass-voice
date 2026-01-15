# app/menu/store.py

from __future__ import annotations

import json
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional, Set

from app.menu.exceptions import MenuLoadError


def _norm(text: str) -> str:
    return text.lower().strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class MenuStore:
    """
    Immutable, in-memory menu store.

    Loads:
    - menu.json
    - entity_index.json

    After load:
    - NO IO
    - NO mutation
    """

    def __init__(self, menu_path: Path, entity_index_path: Path):
        self.menu_path = menu_path
        self.entity_index_path = entity_index_path

        # Canonical data
        self.items: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, Dict[str, Any]] = {}
        self.entity_index: Dict[str, List[Dict[str, Any]]] = {}

        # Runtime indexes
        self._item_by_name: Dict[str, Dict[str, Any]] = {}
        self._item_tokens: Dict[str, Set[str]] = {}
        self._items_by_category: Dict[str, List[Dict[str, Any]]] = {}

        self._load()

    # -------------------------
    # Load
    # -------------------------

    def _load(self) -> None:
        try:
            with open(self.menu_path, "r", encoding="utf-8") as f:
                menu = json.load(f)

            self.items = menu.get("items", {})
            self.categories = menu.get("categories", {})

            if not self.items:
                raise MenuLoadError("menu.json contains no items")

            with open(self.entity_index_path, "r", encoding="utf-8") as f:
                self.entity_index = json.load(f)

            self._build_indexes()

        except Exception as e:
            raise MenuLoadError(str(e)) from e

    def _build_indexes(self) -> None:
        self._item_by_name.clear()
        self._item_tokens.clear()
        self._items_by_category.clear()

        for item_id, item in self.items.items():
            name = item.get("name")
            if not name:
                continue

            key = _norm(name)
            self._item_by_name[key] = item
            self._item_tokens[key] = set(key.split())

            for cat_id in item.get("categories", []):
                self._items_by_category.setdefault(cat_id, []).append(item)

    # -------------------------
    # Direct Access
    # -------------------------

    def get_item(self, item_id: str) -> Dict[str, Any]:
        item = self.items.get(item_id)
        if not item:
            raise KeyError(f"Item not found: {item_id}")
        return item

    # -------------------------
    # Entity Index (Tier 1)
    # -------------------------

    def find_entity(
        self,
        key: str,
        *,
        allowed_types: set[str] | None = None,
        parent_item_id: str | None = None,
    ) -> List[Dict[str, Any]]:

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

    # -------------------------
    # Item Matching (Fallback)
    # -------------------------

    def find_item_exact(self, name: str) -> Optional[Dict[str, Any]]:
        return self._item_by_name.get(_norm(name))

    def find_item_by_tokens(self, text: str) -> Optional[Dict[str, Any]]:
        tokens = set(_norm(text).split())
        best_item = None
        best_score = 0

        for name, name_tokens in self._item_tokens.items():
            overlap = len(tokens & name_tokens)
            if overlap > best_score:
                best_score = overlap
                best_item = self._item_by_name[name]

        return best_item if best_score > 0 else None

    def find_item_fuzzy(
        self,
        text: str,
        *,
        min_score: float = 0.6,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        q = _norm(text)
        tokens = set(q.split())
        matches = []

        for name, name_tokens in self._item_tokens.items():
            if not (tokens & name_tokens):
                continue

            item = self._item_by_name[name]
            score = _similarity(q, _norm(item["name"]))

            if score >= min_score:
                matches.append({"item": item, "score": score})

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:max_results]
