# app/menu/menu_repository.py

from typing import Optional
from app.menu.models import MenuItem


class MenuRepository:
    """
    Read-only interface to query menu items.
    """

    def __init__(self, items: dict[str, MenuItem]) -> None:
        self.items = items

    def find_item_by_name(self, name: str) -> Optional[MenuItem]:
        name = name.lower()
        for item in self.items.values():
            if item.name.lower() == name or name in item.aliases:
                return item
        return None
