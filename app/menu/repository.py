# app/menu/repository.py

from typing import Optional, Dict

from app.menu.store import MenuStore
from app.menu.models import MenuItem
from app.utils.item_matching import score_item


class MenuRepository:
    """
    Public menu query API for NLU and handlers.

    Responsibilities:
    -----------------
    - Resolve user text into a MenuItem
    - Apply deterministic scoring and confidence thresholds
    - Act as the ONLY authority for item resolution

    Non-responsibilities:
    ---------------------
    - JSON parsing
    - Index construction
    - Conversational flow
    """

    def __init__(self, store: MenuStore):
        self.store = store

    # =================================================
    # Item Resolution
    # =================================================

    def resolve_item(self, text: str) -> Optional[MenuItem]:
        """
        Resolve a single menu item from free-form user text.

        Strategy:
        ---------
        1. Collect candidates from multiple weak/strong signals
        2. Score all candidates deterministically
        3. Select the highest-scoring item above confidence threshold

        This method NEVER short-circuits early.
        """

        candidates: Dict[str, MenuItem] = {}

        # -------------------------------------------------
        # 1️⃣ Entity index candidates (high precision)
        # -------------------------------------------------
        entities = self.store.find_entity(text, allowed_types={"item"})
        for e in entities:
            try:
                item = self.store.get_item(e["item_id"])
                candidates[item.item_id] = item
            except KeyError:
                # Defensive: entity index may reference stale IDs
                continue

        # -------------------------------------------------
        # 2️⃣ Exact name / alias candidates
        # -------------------------------------------------
        exact = self.store.find_item_exact(text)
        if exact:
            candidates[exact.item_id] = exact

        # -------------------------------------------------
        # 3️⃣ Global fallback (ensures recall)
        # -------------------------------------------------
        for item in self.store.items.values():
            candidates.setdefault(item.item_id, item)

        # -------------------------------------------------
        # 4️⃣ Score & select best candidate
        # -------------------------------------------------
        best_item: Optional[MenuItem] = None
        best_score: float = 0.0

        for item in candidates.values():
            score = score_item(text, item.name)
            if score > best_score:
                best_score = score
                best_item = item

        # -------------------------------------------------
        # 5️⃣ Confidence threshold
        # -------------------------------------------------
        return best_item if best_score >= 2.5 else None

    # =================================================
    # Direct Access
    # =================================================

    def get_item(self, item_id: str) -> MenuItem:
        """
        Fetch item by ID.

        This is a hard lookup and will raise if the item does not exist.
        """
        return self.store.get_item(item_id)
