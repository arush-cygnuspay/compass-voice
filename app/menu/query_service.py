# app/menu/query_service.py
"""High-level menu query orchestration.

Owns category resolution, not-found recovery, and slot-orchestrated queries.
Delegates all item matching/scoring to ``MenuMatcher``.

``MenuQueryService`` wires together every component:

    store → MenuIndexer → MenuMatcher
                ↑                ↑
           MenuScorer    (callbacks passed as constructor args to MenuMatcher
                          to break the circular dependency)
"""
from __future__ import annotations

from app.menu.indexer import MenuIndexer
from app.menu.matcher import MenuMatcher
from app.menu.models import ItemResolution, MenuItem
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.scorer import MenuScorer
from app.menu.slot_helpers import slot_values
from app.menu.store import MenuStore
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text


class MenuQueryService:
    """Orchestrates menu queries across all resolution strategies.

    Parameters
    ----------
    store:
        Live ``MenuStore`` — source of all menu data and indexes.
    scorer:
        Optional ``MenuScorer`` override (defaults to ``MenuScorer()``).
    indexer:
        Optional ``MenuIndexer`` override (defaults to one built from *store*).
    """

    def __init__(
        self,
        store: MenuStore,
        *,
        scorer: MenuScorer | None = None,
        indexer: MenuIndexer | None = None,
    ) -> None:
        self._store = store
        self._scorer = scorer or MenuScorer()
        self._indexer = indexer or MenuIndexer(store)
        self._matcher = MenuMatcher(
            store=self._store,
            indexer=self._indexer,
            scorer=self._scorer,
            build_not_found_recovery_fn=self.build_not_found_recovery_normalized,
            resolve_category_fn=self._resolve_category_for_matcher,
        )

    def _resolve_category_for_matcher(
        self, normalized_text: str, limit: int
    ) -> MenuQueryResult | None:
        return self.resolve_category_query_normalized(normalized_text, limit=limit)

    # ------------------------------------------------------------------
    # DATA ACCESS
    # ------------------------------------------------------------------

    def get_item(self, item_id: str) -> MenuItem:
        return self._store.get_item(item_id)

    # ------------------------------------------------------------------
    # NOT-FOUND RECOVERY
    # ------------------------------------------------------------------

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
        item_limit: int = 3,
        category_limit: int = 4,
    ) -> tuple[list[MenuItem], list[dict]]:
        similar_items = self._find_similar_items_normalized(normalized_text, limit=item_limit)
        categories = self._top_browse_categories()
        return similar_items, categories[:category_limit]

    def _find_similar_items_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 3,
    ) -> list[MenuItem]:
        if not normalized_text:
            return []

        candidates = self._indexer.candidate_items(normalized_text)

        scored: list[tuple[float, MenuItem]] = []
        seen_ids: set[str] = set()

        for item in candidates:
            if not item.available:
                continue

            score = self._scorer.score_item_labels(normalized_text, item)

            if score < self._scorer.similarity_minimum or item.item_id in seen_ids:
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

        for category in self._store.categories.values():
            name = str(category.get("name", "")).strip()
            if not name:
                continue

            norm = normalize_text(name)
            if not norm or norm in blocked:
                continue

            item_ids = category.get("item_ids", []) or []
            item_count = sum(1 for item_id in item_ids if item_id in self._store.items)
            if item_count <= 0:
                continue

            preferred_rank = 0 if norm in preferred else 1
            ranked.append((preferred_rank, -item_count, name, category))

        ranked.sort(key=lambda row: (row[0], row[1], row[2]))
        return [category for _, _, _, category in ranked]

    # ------------------------------------------------------------------
    # CATEGORY RESOLUTION
    # ------------------------------------------------------------------

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

        entries = self._store.find_entity(normalized_text, allowed_types={"category"})
        for entry in entries:
            category_id = entry.get("category_id")
            if not category_id:
                continue

            category = self._store.categories.get(category_id)
            if category is not None:
                return self._build_category_result(category, limit=limit)

        category = self._store.find_category_by_name(normalized_text)
        if category is not None:
            return self._build_category_result(category, limit=limit)

        fuzzy = self._find_category_fuzzy(normalized_text)
        if fuzzy is not None:
            return self._build_category_result(fuzzy, limit=limit)

        return None

    def _build_category_result(self, category: dict, *, limit: int = 5) -> MenuQueryResult:
        items = [
            self._store.get_item(item_id)
            for item_id in category.get("item_ids", [])
            if item_id in self._store.items
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

        for category in self._store.categories.values():
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

    # ------------------------------------------------------------------
    # SLOT-FIRST RESOLUTION
    # ------------------------------------------------------------------

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
        item_queries = [
            normalize_text(value)
            for value in slot_values(slots, "ITEM", "MENU_ITEM")
            if normalize_text(value)
        ]
        category_queries = [
            normalize_text(value)
            for value in slot_values(slots, "CATEGORY", "MENU_CATEGORY")
            if normalize_text(value)
        ]

        category_result = None
        for normalized_category_query in category_queries:
            category_result = self.resolve_category_query_normalized(
                normalized_category_query,
                limit=limit,
            )
            if category_result is not None:
                break

        if category_result is not None and not item_queries:
            return category_result

        prioritized_item_queries = [
            query for query in item_queries if self._indexer.has_item_evidence(query)
        ]
        if not prioritized_item_queries:
            prioritized_item_queries = item_queries

        fallback_slot_item_result: MenuQueryResult | None = None
        for normalized_item_query in prioritized_item_queries:
            slot_item_result = self._matcher.resolve_from_slot(
                normalized_item_query,
                limit=limit,
            )
            if slot_item_result is None:
                continue

            if slot_item_result.type == MenuQueryType.ITEM:
                return slot_item_result

            if fallback_slot_item_result is None:
                fallback_slot_item_result = slot_item_result

        if category_result is not None:
            return category_result

        if fallback_slot_item_result is not None:
            return fallback_slot_item_result

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

    # ------------------------------------------------------------------
    # GENERAL FREE-TEXT RESOLUTION
    # ------------------------------------------------------------------

    def resolve_menu_query(self, text: str, *, limit: int = 5) -> MenuQueryResult:
        return self.resolve_menu_query_normalized(normalize_text(text), limit=limit)

    def resolve_menu_query_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult:
        return self._matcher.resolve_free_text(normalized_text, limit=limit)

    # ------------------------------------------------------------------
    # IDLE AVAILABILITY RESOLUTION
    # ------------------------------------------------------------------

    def resolve_idle_availability_query_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult:
        return self._matcher.resolve_idle_availability(normalized_text, limit=limit)

    # ------------------------------------------------------------------
    # CANDIDATE-LOCAL / SIDE / MODIFIER RESOLUTION (delegation)
    # ------------------------------------------------------------------

    def resolve_item_within_candidates_normalized(
        self,
        *,
        normalized_text: str,
        candidate_item_ids: list[str] | tuple[str, ...],
    ) -> MenuItem | None:
        return self._matcher.resolve_within_candidates(
            normalized_text=normalized_text,
            candidate_item_ids=candidate_item_ids,
        )

    def resolve_side_choice_within_group_normalized(
        self,
        *,
        normalized_text: str,
        group_id: str,
        candidate_names_by_id: dict[str, tuple[str, ...]],
    ) -> list[str]:
        return self._matcher.resolve_side_group(
            normalized_text=normalized_text,
            group_id=group_id,
            candidate_names_by_id=candidate_names_by_id,
        )

    def resolve_modifier_choice_within_group_normalized(
        self,
        *,
        normalized_text: str,
        group_id: str,
        candidate_names_by_id: dict[str, tuple[str, ...]],
    ) -> list[str]:
        return self._matcher.resolve_modifier_group(
            normalized_text=normalized_text,
            group_id=group_id,
            candidate_names_by_id=candidate_names_by_id,
        )

    def resolve_modifier_availability_for_item_normalized(
        self,
        *,
        normalized_text: str,
        item_id: str,
    ) -> dict | None:
        return self._matcher.resolve_modifier_availability(
            normalized_text=normalized_text,
            item_id=item_id,
        )

    def resolve_item(self, text: str) -> ItemResolution | None:
        return self.resolve_item_normalized(normalize_text(text))

    def resolve_item_normalized(self, normalized_text: str) -> ItemResolution | None:
        return self._matcher.resolve_item(normalized_text)
