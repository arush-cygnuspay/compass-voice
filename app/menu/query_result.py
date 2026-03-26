# app/menu/query_result.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.menu.models import MenuItem


class MenuQueryType(str, Enum):
    ITEM = "item"
    CATEGORY = "category"
    CATEGORY_SINGLE_ITEM = "category_single_item"
    ITEM_AMBIGUOUS = "item_ambiguous"
    CATEGORY_AMBIGUOUS = "category_ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class MenuQueryResult:
    """
    Final deterministic resolution of a menu query.
    Describes what matched in the menu, not what the assistant should say.
    """

    type: MenuQueryType
    item: MenuItem | None = None
    category_id: str | None = None
    category_name: str | None = None
    items: list[MenuItem] | None = None
    matched_items: list[MenuItem] | None = None
    matched_categories: list[dict[str, Any]] | None = None

    # new
    suggested_items: list[MenuItem] | None = None
    suggested_categories: list[dict[str, Any]] | None = None


