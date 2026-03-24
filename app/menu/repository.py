# app/menu/repository.py
from __future__ import annotations

from app.menu.models import ItemResolution, MenuItem
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.slot_helpers import first_slot_value
from app.menu.store import MenuStore
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.utils.item_matching import score_item, score_item_normalized


class MenuRepository:
    """
    Public menu query API for NLU and handlers.

    Hot-path contract:
    - normalized_* methods expect already-normalized text
    - wrapper methods exist only for compatibility callers that still pass raw text
    """

    def __init__(self, store: MenuStore):
        self.store = store

    # ======================================================
    # CATEGORY RESOLUTION
    # ======================================================

    def resolve_category_query(self, text: str, *, limit: int = 5) -> MenuQueryResult | None:
        return self.resolve_category_query_normalized(normalize_text(text or ""), limit=limit)

    def resolve_category_query_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult | None:
        if not normalized_text:
            return None

        entries = self.store.find_entity(normalized_text, allowed_types={"category"})
        for entry in entries:
            category_id = entry.get("category_id")
            if not category_id:
                continue

            category = self.store.categories.get(category_id)
            if category is not None:
                return self._build_category_result(category, limit=limit)

        category = self.store.find_category_by_name(normalized_text)
        if category is not None:
            return self._build_category_result(category, limit=limit)

        fuzzy = self._find_category_fuzzy(normalized_text)
        if fuzzy is not None:
            return self._build_category_result(fuzzy, limit=limit)

        return None

    def _build_category_result(self, category: dict, *, limit: int = 5) -> MenuQueryResult:
        items = [
            self.store.get_item(item_id)
            for item_id in category.get("item_ids", [])
            if item_id in self.store.items
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

    def _find_category_fuzzy(self, normalized_text: str) -> dict | None:
        best_category = None
        best_score = 0

        for category in self.store.categories.values():
            category_name = normalize_text(str(category.get("name", "")))
            if not category_name:
                continue

            if normalized_text == category_name:
                return category

            if len(normalized_text) >= 3 and normalized_text in category_name:
                score = 3
            elif len(category_name) >= 3 and category_name in normalized_text:
                score = 2
            else:
                score = len(set(normalized_text.split()) & set(category_name.split()))

            if score > best_score:
                best_score = score
                best_category = category

        return best_category if best_score > 0 else None

    # ======================================================
    # SLOT-FIRST RESOLUTION
    # ======================================================

    def resolve_menu_query_from_slots(
        self,
        *,
        user_text: str,
        slots: list[SlotValue] | tuple[SlotValue, ...],
        fallback_to_text: bool = True,
        limit: int = 5,
    ) -> MenuQueryResult:
        return self.resolve_menu_query_from_slots_normalized(
            normalized_user_text=normalize_text(user_text or ""),
            slots=slots,
            fallback_to_text=fallback_to_text,
            limit=limit,
        )

    def resolve_menu_query_from_slots_normalized(
        self,
        *,
        normalized_user_text: str,
        slots: list[SlotValue] | tuple[SlotValue, ...],
        fallback_to_text: bool = True,
        limit: int = 5,
    ) -> MenuQueryResult:
        item_query = first_slot_value(slots, "ITEM", "MENU_ITEM")
        category_query = first_slot_value(slots, "CATEGORY", "MENU_CATEGORY")

        normalized_item_query = normalize_text(item_query) if item_query else ""
        normalized_category_query = normalize_text(category_query) if category_query else ""

        if normalized_category_query and not normalized_item_query:
            category_result = self.resolve_category_query_normalized(
                normalized_category_query,
                limit=limit,
            )
            if category_result is not None:
                return category_result

        if normalized_item_query:
            slot_item_result = self._resolve_item_query_from_explicit_slot_normalized(
                normalized_item_query,
                limit=limit,
            )

            if normalized_category_query:
                category_result = self.resolve_category_query_normalized(
                    normalized_category_query,
                    limit=limit,
                )

                if category_result is not None and slot_item_result is not None:
                    if slot_item_result.type in {
                        MenuQueryType.ITEM_AMBIGUOUS,
                        MenuQueryType.NOT_FOUND,
                    }:
                        return category_result

                if category_result is not None and slot_item_result is None:
                    return category_result

            if slot_item_result is not None:
                return slot_item_result

        if normalized_category_query:
            category_result = self.resolve_category_query_normalized(
                normalized_category_query,
                limit=limit,
            )
            if category_result is not None:
                return category_result

        if not fallback_to_text or not normalized_user_text:
            similar_items, categories = self.build_not_found_recovery_normalized(
                normalized_user_text,
                item_limit=3,
                category_limit=4,
            )
            return MenuQueryResult(
                type=MenuQueryType.NOT_FOUND,
                suggested_items=similar_items,
                suggested_categories=categories,
            )

        return self.resolve_menu_query_normalized(normalized_user_text, limit=limit)

    def _resolve_item_query_from_explicit_slot(
        self,
        text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult | None:
        return self._resolve_item_query_from_explicit_slot_normalized(
            normalize_text(text),
            limit=limit,
        )

    def _resolve_item_query_from_explicit_slot_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult | None:
        if not normalized_text:
            return None

        entity_candidates = self.store.find_entity(normalized_text, allowed_types={"item"})
        unique_item_ids: list[str] = []
        seen_ids: set[str] = set()

        for entry in entity_candidates:
            item_id = entry.get("item_id")
            if not item_id or item_id in seen_ids or item_id not in self.store.items:
                continue
            seen_ids.add(item_id)
            unique_item_ids.append(item_id)

        for item_id in self.store.find_item_ids_by_alias(normalized_text):
            if item_id in seen_ids or item_id not in self.store.items:
                continue
            seen_ids.add(item_id)
            unique_item_ids.append(item_id)

        exact_item = self.store.find_item_exact(normalized_text)
        if exact_item is not None and exact_item.item_id not in seen_ids:
            unique_item_ids.insert(0, exact_item.item_id)
            seen_ids.add(exact_item.item_id)

        if len(unique_item_ids) == 1:
            return MenuQueryResult(
                type=MenuQueryType.ITEM,
                item=self.store.get_item(unique_item_ids[0]),
            )

        candidates = (
            [self.store.get_item(item_id) for item_id in unique_item_ids]
            if unique_item_ids
            else list(self.store.items.values())
        )

        scored_items: list[tuple[float, MenuItem]] = []
        for item in candidates:
            if not item.available:
                continue

            score = max(
                score_item_normalized(normalized_text, item.normalized_name),
                max((score_item(normalized_text, alias) for alias in item.normalized_aliases), default=0.0),
            )
            if score > 0:
                scored_items.append((score, item))

        if not scored_items:
            similar_items, categories = self.build_not_found_recovery_normalized(
                normalized_text,
                item_limit=3,
                category_limit=4,
            )
            return MenuQueryResult(
                type=MenuQueryType.NOT_FOUND,
                suggested_items=similar_items,
                suggested_categories=categories,
            )

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_item = scored_items[0]
        second_score = scored_items[1][0] if len(scored_items) > 1 else 0.0

        clear_winner = (
            best_score >= 5.8 and (
                second_score == 0.0
                or best_score - second_score >= 0.9
                or best_score >= second_score * 1.18
            )
        )

        if clear_winner:
            return MenuQueryResult(type=MenuQueryType.ITEM, item=best_item)

        close_items: list[MenuItem] = []
        seen_item_ids: set[str] = set()

        for score, item in scored_items:
            if score < best_score * 0.92:
                break
            if item.item_id in seen_item_ids:
                continue
            seen_item_ids.add(item.item_id)
            close_items.append(item)

        if len(close_items) == 1 and best_score >= 5.4:
            return MenuQueryResult(type=MenuQueryType.ITEM, item=close_items[0])

        if len(close_items) > 1:
            return MenuQueryResult(
                type=MenuQueryType.ITEM_AMBIGUOUS,
                matched_items=close_items[:limit],
            )

        similar_items, categories = self.build_not_found_recovery_normalized(
            normalized_text,
            item_limit=3,
            category_limit=4,
        )
        return MenuQueryResult(
            type=MenuQueryType.NOT_FOUND,
            suggested_items=similar_items,
            suggested_categories=categories,
        )

    # ======================================================
    # GENERAL FREE-TEXT RESOLUTION
    # ======================================================

    def resolve_menu_query(self, text: str, *, limit: int = 5) -> MenuQueryResult:
        return self.resolve_menu_query_normalized(normalize_text(text), limit=limit)

    def resolve_menu_query_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult:
        if not normalized_text:
            similar_items, categories = self.build_not_found_recovery_normalized(
                normalized_text,
                item_limit=3,
                category_limit=4,
            )
            return MenuQueryResult(
                type=MenuQueryType.NOT_FOUND,
                suggested_items=similar_items,
                suggested_categories=categories,
            )

        category_result = self.resolve_category_query_normalized(normalized_text, limit=limit)
        if category_result is not None:
            return category_result

        entity_candidates = self.store.find_entity(normalized_text, allowed_types={"item"})
        candidate_ids = {
            entry.get("item_id")
            for entry in entity_candidates
            if entry.get("item_id")
        }

        exact_item = self.store.find_item_exact(normalized_text)
        if exact_item is not None:
            candidate_ids.add(exact_item.item_id)

        for item_id in self.store.find_item_ids_by_alias(normalized_text):
            candidate_ids.add(item_id)

        candidates = (
            [
                self.store.get_item(item_id)
                for item_id in candidate_ids
                if item_id in self.store.items
            ]
            if candidate_ids
            else list(self.store.items.values())
        )

        scored_items: list[tuple[float, MenuItem]] = []
        for item in candidates:
            if not item.available:
                continue

            score = max(
                score_item_normalized(normalized_text, item.normalized_name),
                max((score_item(normalized_text, alias) for alias in item.normalized_aliases), default=0.0),
            )
            if score > 0:
                scored_items.append((score, item))

        if not scored_items:
            similar_items, categories = self.build_not_found_recovery_normalized(
                normalized_text,
                item_limit=3,
                category_limit=4,
            )
            return MenuQueryResult(
                type=MenuQueryType.NOT_FOUND,
                suggested_items=similar_items,
                suggested_categories=categories,
            )

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_item = scored_items[0]

        strong_items: list[MenuItem] = []
        seen_item_ids: set[str] = set()

        for score, item in scored_items:
            if score < best_score * 0.85:
                break
            if item.item_id in seen_item_ids:
                continue
            seen_item_ids.add(item.item_id)
            strong_items.append(item)

        if best_score >= 6.0 and len(strong_items) == 1:
            return MenuQueryResult(type=MenuQueryType.ITEM, item=best_item)

        if len(strong_items) > 1:
            return MenuQueryResult(
                type=MenuQueryType.ITEM_AMBIGUOUS,
                matched_items=strong_items[:limit],
            )

        similar_items, categories = self.build_not_found_recovery_normalized(
            normalized_text,
            item_limit=3,
            category_limit=4,
        )
        return MenuQueryResult(
            type=MenuQueryType.NOT_FOUND,
            suggested_items=similar_items,
            suggested_categories=categories,
        )

    # ======================================================
    # EXISTING API
    # ======================================================

    def get_item(self, item_id: str) -> MenuItem:
        return self.store.get_item(item_id)

    def resolve_item(self, text: str) -> ItemResolution | None:
        return self.resolve_item_normalized(normalize_text(text))

    def resolve_item_normalized(self, normalized_text: str) -> ItemResolution | None:
        if not normalized_text:
            return None

        candidates: dict[str, MenuItem] = {}

        entity_candidates = self.store.find_entity(normalized_text, allowed_types={"item"})
        for entry in entity_candidates:
            item_id = entry.get("item_id")
            if not item_id:
                continue

            try:
                item = self.store.get_item(item_id)
            except KeyError:
                continue

            candidates[item.item_id] = item

        exact = self.store.find_item_exact(normalized_text)
        if exact is not None:
            candidates[exact.item_id] = exact

        for item_id in self.store.find_item_ids_by_alias(normalized_text):
            if item_id in self.store.items:
                candidates[item_id] = self.store.get_item(item_id)

        if not candidates:
            candidates = self.store.items.copy()

        best_item: MenuItem | None = None
        best_score = 0.0

        for item in candidates.values():
            score = max(
                score_item_normalized(normalized_text, item.normalized_name),
                max((score_item(normalized_text, alias) for alias in item.normalized_aliases), default=0.0),
            )
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None or best_score < 6.5:
            return None

        return ItemResolution(item=best_item, score=best_score)

    def build_not_found_recovery(
        self,
        text: str,
        *,
        item_limit: int = 3,
        category_limit: int = 4,
    ) -> tuple[list[MenuItem], list[dict]]:
        return self.build_not_found_recovery_normalized(
            normalize_text(text),
            item_limit=item_limit,
            category_limit=category_limit,
        )

    def build_not_found_recovery_normalized(
        self,
        normalized_text: str,
        *,
        item_limit: int = 3,
        category_limit: int = 4,
    ) -> tuple[list[MenuItem], list[dict]]:
        similar_items = self._find_similar_items_normalized(normalized_text, limit=item_limit)
        categories = self._top_browse_categories()
        return similar_items, categories[:category_limit]

    def _find_similar_items(self, text: str, *, limit: int = 3) -> list[MenuItem]:
        return self._find_similar_items_normalized(normalize_text(text), limit=limit)

    def _find_similar_items_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 3,
    ) -> list[MenuItem]:
        if not normalized_text:
            return []

        entity_candidates = self.store.find_entity(normalized_text, allowed_types={"item"})
        candidate_ids = {
            entry.get("item_id")
            for entry in entity_candidates
            if entry.get("item_id")
        }

        exact_item = self.store.find_item_exact(normalized_text)
        if exact_item is not None:
            candidate_ids.add(exact_item.item_id)

        for item_id in self.store.find_item_ids_by_alias(normalized_text):
            candidate_ids.add(item_id)

        candidates = (
            [
                self.store.get_item(item_id)
                for item_id in candidate_ids
                if item_id in self.store.items
            ]
            if candidate_ids
            else list(self.store.items.values())
        )

        scored: list[tuple[float, MenuItem]] = []
        seen_ids: set[str] = set()

        for item in candidates:
            if not item.available:
                continue

            score = max(
                score_item_normalized(normalized_text, item.normalized_name),
                max((score_item(normalized_text, alias) for alias in item.normalized_aliases), default=0.0),
            )

            if score < 4.8 or item.item_id in seen_ids:
                continue

            seen_ids.add(item.item_id)
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def _top_browse_categories(self) -> list[dict]:
        blocked = {
            "ebt",
            "gift card",
        }

        preferred = {
            "tacos",
            "drinks",
            "desserts",
            "pizza",
            "soups",
            "kids menu",
            "specials",
            "sandwiches",
            "party platters",
            "fried seafood platters",
            "broiled seafood platters",
            "fried combo platters",
            "broiled combo platters",
            "fried party wings",
        }

        ranked: list[tuple[int, int, str, dict]] = []

        for category in self.store.categories.values():
            name = str(category.get("name", "")).strip()
            if not name:
                continue

            norm = normalize_text(name)
            if not norm or norm in blocked:
                continue

            item_ids = category.get("item_ids", []) or []
            item_count = sum(1 for item_id in item_ids if item_id in self.store.items)
            if item_count <= 0:
                continue

            preferred_rank = 0 if norm in preferred else 1
            ranked.append((preferred_rank, -item_count, name, category))

        ranked.sort(key=lambda row: (row[0], row[1], row[2]))
        return [category for _, _, _, category in ranked]