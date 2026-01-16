# app/menu/repository.py

from typing import Optional
from app.menu.store import MenuStore
from app.menu.models import MenuItem


class MenuRepository:
    """
    Public menu query API for NLU & handlers.
    Provides a single, canonical item resolution pipeline.
    """

    def __init__(self, store: MenuStore):
        self.store = store

    def resolve_item(self, text: str) -> Optional[MenuItem]:
        """
        Resolve a single menu item from free-form user text.

        Resolution order (strong → weak):
        1. Entity index
        2. Exact name / alias
        3. Token overlap

        Returns:
            MenuItem | None
        """

        # ---------------------------
        # Tier 1: Entity index
        # ---------------------------
        entities = self.store.find_entity(text, allowed_types={"item"})
        if entities:
            return self.store.get_item(entities[0]["item_id"])

        # ---------------------------
        # Tier 2: Exact name / alias
        # ---------------------------
        exact = self.store.find_item_exact(text)
        if exact:
            return exact

        # ---------------------------
        # Tier 3: Token overlap
        # ---------------------------
        token_match = self.store.find_item_by_tokens(text)
        if token_match:
            return token_match

        return None
