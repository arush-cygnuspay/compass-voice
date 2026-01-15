# app/menu/repository.py

from typing import Optional
from app.menu.store import MenuStore
from app.menu.models import MenuItem


class MenuRepository:
    """
    Public menu query API for NLU & handlers.
    """

    def __init__(self, store: MenuStore):
        self.store = store

    def resolve_item_by_entity(self, text: str) -> Optional[MenuItem]:
        matches = self.store.find_entity(text, allowed_types={"item"})
        if not matches:
            return None

        return self.store.get_item(matches[0]["item_id"])
