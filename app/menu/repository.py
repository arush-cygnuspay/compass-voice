# app/menu/repository.py

from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.store import MenuStore
from app.menu.models import *
from app.utils.item_matching import score_item


class MenuRepository:
    """
    Public menu query API for NLU and handlers.

    Authority:
    ------------------
    This layer is the ONLY authority for:
    - Category vs Item dominance
    - Ambiguity classification
    - Deterministic menu resolution

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

    It MUST NOT:
    - Interpret intent
    - Perform conversational logic
    """

    def __init__(self, store: MenuStore):
        self.store = store

    # =================================================
    # Item Resolution
    # =================================================

    def resolve_item(self, text: str) -> Optional[ItemResolution]:
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
        return ItemResolution(best_item, best_score) if best_score >= 6.5 else None

    # =====================================================
    # MENU QUERY RESOLUTION (ITEM | CATEGORY | AMBIGUOUS)
    # =====================================================

    def resolve_menu_query(self, text: str, *, limit: int = 5) -> MenuQueryResult:
        """
        Resolve a menu-related query into a structured MenuQueryResult.

        Dominance order:
        -----------------
        1. Exact category dominance
        2. Strong item dominance
        3. Category ambiguity
        4. Item ambiguity
        5. Weak fallback
        """

        norm_text = text.strip().lower()
        tokens = set(norm_text.split())

        # ---------------------------------------------
        # 1️⃣ Entity index lookup (ground truth)
        # ---------------------------------------------
        entities = self.store.find_entity(norm_text)

        item_entities = []
        category_entities = []

        for e in entities:
            if e.get("type") == "item":
                try:
                    item_entities.append(self.store.get_item(e["item_id"]))
                except KeyError:
                    pass
            elif e.get("type") == "category":
                cat = self.store.categories.get(e["category_id"])
                if cat:
                    category_entities.append(cat)

        # ---------------------------------------------
        # 2️⃣ CATEGORY DOMINANCE CHECK (ROBUST)
        # ---------------------------------------------
        category = self.store.find_category_by_name(norm_text)

        if category:
            items = [
                self.store.get_item(i)
                for i in category.get("item_ids", [])
                if i in self.store.items
            ]

            if len(items) == 1:
                return MenuQueryResult(
                    type=MenuQueryType.CATEGORY_SINGLE_ITEM,
                    category_id=category["category_id"],
                    category_name=category["name"],
                    items=items,
                )

            return MenuQueryResult(
                type=MenuQueryType.CATEGORY,
                category_id=category["category_id"],
                category_name=category["name"],
                items=items[:limit],
            )

        # ---------------------------------------------
        # 3️⃣ ITEM DOMINANCE CHECK
        # ---------------------------------------------
        scored_items = []

        for item in self.store.items.values():
            score = score_item(norm_text, item.name)
            if score > 0:
                scored_items.append((score, item))

        if scored_items:
            scored_items.sort(key=lambda x: x[0], reverse=True)

            best_score, best_item = scored_items[0]

            # Strong dominance threshold
            if best_score >= 6.0:
                return MenuQueryResult(
                    type=MenuQueryType.ITEM,
                    item=best_item,
                )

            # Multiple close candidates → ambiguity
            strong_items = [
                item for score, item in scored_items
                if score >= best_score * 0.85
            ]

            if len(strong_items) > 1:
                return MenuQueryResult(
                    type=MenuQueryType.ITEM_AMBIGUOUS,
                    matched_items=strong_items[:limit],
                )

        # ---------------------------------------------
        # 4️⃣ Nothing matched
        # ---------------------------------------------
        return MenuQueryResult(type=MenuQueryType.NOT_FOUND)

    # =================================================
    # Direct Access
    # =================================================

    def get_item(self, item_id: str) -> MenuItem:
        """
        Fetch item by ID.

        This is a hard lookup and will raise if the item does not exist.
        """
        return self.store.get_item(item_id)

