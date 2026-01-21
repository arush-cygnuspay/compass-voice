# app/menu/query_result.py

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional
from app.menu.models import MenuItem


class MenuQueryType(Enum):
    ITEM = auto()                  # One clear item dominates
    CATEGORY = auto()              # One clear category dominates
    CATEGORY_SINGLE_ITEM = auto()  # Category with exactly 1 item
    ITEM_AMBIGUOUS = auto()        # Multiple item candidates
    CATEGORY_AMBIGUOUS = auto()    # Multiple category candidates
    NOT_FOUND = auto()


@dataclass(frozen=True)
class MenuQueryResult:
    """
    Final, deterministic resolution of a menu query.

    This object describes WHAT matched in the menu,
    not WHAT the assistant should say.
    """

    type: MenuQueryType

    # ---------- ITEM ----------
    item: Optional[MenuItem] = None

    # ---------- CATEGORY ----------
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    items: Optional[List[MenuItem]] = None  # items in category

    # ---------- AMBIGUITY ----------
    matched_items: Optional[List[MenuItem]] = None
    matched_categories: Optional[List[dict]] = None


