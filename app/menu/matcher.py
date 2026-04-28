# app/menu/matcher.py
"""Menu matching, candidate selection, and ambiguity resolution.

All matching/scoring decision logic moved verbatim from MenuRepository.
No normalization happens here — all inputs are already normalized.

Dependency injection pattern
----------------------------
``MenuMatcher`` receives two callables at construction time to avoid
a circular import with ``MenuQueryService``:

* ``build_not_found_recovery_fn(normalized_text, item_limit, category_limit)``
  → ``tuple[list[MenuItem], list[dict]]``
  Provided by ``MenuQueryService.build_not_found_recovery_normalized``.

* ``resolve_category_fn(normalized_text, limit)``
  → ``MenuQueryResult | None``
  Provided by ``MenuQueryService.resolve_category_query_normalized``.

Both are called lazily (at query time, not at construction) so the
``MenuQueryService`` instance is fully initialised before any call.
"""
from __future__ import annotations

from typing import Callable, Sequence

from app.menu.indexer import MenuIndexer
from app.menu.models import ItemResolution, MenuItem
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.scorer import MenuScorer
from app.menu.store import MenuStore
from app.utils.item_matching import score_item


class MenuMatcher:
    """Owns all matching, candidate selection, and ambiguity resolution.

    Parameters
    ----------
    store:
        Live ``MenuStore`` — used for direct item/group lookups not covered
        by ``MenuIndexer``.
    indexer:
        Pre-built candidate index adapter.
    scorer:
        Stateless label scorer.
    build_not_found_recovery_fn:
        Callable that produces ``(similar_items, browse_categories)`` for
        not-found results.  Supplied by ``MenuQueryService`` to avoid circular
        imports.
    resolve_category_fn:
        Callable that resolves a normalized text to a category
        ``MenuQueryResult``.  Supplied by ``MenuQueryService``.
    """

    def __init__(
        self,
        store: MenuStore,
        indexer: MenuIndexer,
        scorer: MenuScorer,
        build_not_found_recovery_fn: Callable[
            [str, int, int], tuple[list[MenuItem], list[dict]]
        ],
        resolve_category_fn: Callable[[str, int], "MenuQueryResult | None"],
    ) -> None:
        self._store = store
        self._indexer = indexer
        self._scorer = scorer
        self._build_not_found_recovery = build_not_found_recovery_fn
        self._resolve_category = resolve_category_fn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _not_found_result(
        self, normalized_text: str, item_limit: int = 3, category_limit: int = 4
    ) -> MenuQueryResult:
        similar_items, categories = self._build_not_found_recovery(
            normalized_text, item_limit, category_limit
        )
        return MenuQueryResult(
            type=MenuQueryType.NOT_FOUND,
            suggested_items=similar_items,
            suggested_categories=categories,
        )

    # ------------------------------------------------------------------
    # CANDIDATE-LOCAL RESOLUTION
    # (moved verbatim from MenuRepository.resolve_item_within_candidates_normalized)
    # ------------------------------------------------------------------

    def resolve_within_candidates(
        self,
        *,
        normalized_text: str,
        candidate_item_ids: list[str] | tuple[str, ...],
    ) -> MenuItem | None:
        if not normalized_text or not candidate_item_ids:
            return None

        candidates: list[MenuItem] = []
        for item_id in candidate_item_ids:
            if item_id in self._store.items:
                candidates.append(self._store.get_item(item_id))

        if not candidates:
            return None

        # 1. Exact / alias / voice-label deterministic hit
        exact_hits: list[MenuItem] = []
        for item in candidates:
            if normalized_text == item.normalized_name:
                exact_hits.append(item)
                continue
            if normalized_text in item.normalized_aliases:
                exact_hits.append(item)
                continue
            if normalized_text in item.voice_labels:
                exact_hits.append(item)

        if len(exact_hits) == 1:
            return exact_hits[0]

        # 2. Score within shortlist
        scored: list[tuple[float, MenuItem]] = []
        for item in candidates:
            score = self._scorer.score_item_labels(normalized_text, item)
            if score > 0:
                scored.append((score, item))

        if not scored:
            return None

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_item = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0

        if best_score >= self._scorer.candidate_clear_winner and (
            second_score == 0.0
            or best_score - second_score >= self._scorer.candidate_gap
            or best_score >= second_score * self._scorer.candidate_ratio
        ):
            return best_item

        return None

    # ------------------------------------------------------------------
    # SIDE / MODIFIER GROUP RESOLUTION
    # (moved verbatim from MenuRepository.resolve_side/modifier_choice_within_group_normalized)
    # ------------------------------------------------------------------

    def resolve_side_group(
        self,
        *,
        normalized_text: str,
        group_id: str,
        candidate_names_by_id: dict[str, tuple[str, ...]],
    ) -> list[str]:
        if not normalized_text or not group_id:
            return []

        exact_ids = self._store.find_side_ids_for_group_by_label(group_id, normalized_text)
        if exact_ids:
            return exact_ids

        scored: list[tuple[float, str]] = []
        for item_id, labels in candidate_names_by_id.items():
            best = max((score_item(normalized_text, label) for label in labels), default=0.0)
            if best > 0:
                scored.append((best, item_id))

        if not scored:
            return []

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score = scored[0][0]
        return [
            item_id
            for score, item_id in scored
            if score >= best_score * self._scorer.group_match_band
        ]

    def resolve_modifier_group(
        self,
        *,
        normalized_text: str,
        group_id: str,
        candidate_names_by_id: dict[str, tuple[str, ...]],
    ) -> list[str]:
        if not normalized_text or not group_id:
            return []

        exact_ids = self._store.find_modifier_ids_for_group_by_label(
            group_id, normalized_text
        )
        if exact_ids:
            return exact_ids

        scored: list[tuple[float, str]] = []
        for modifier_id, labels in candidate_names_by_id.items():
            best = max((score_item(normalized_text, label) for label in labels), default=0.0)
            if best > 0:
                scored.append((best, modifier_id))

        if not scored:
            return []

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score = scored[0][0]
        return [
            modifier_id
            for score, modifier_id in scored
            if score >= best_score * self._scorer.group_match_band
        ]

    # ------------------------------------------------------------------
    # SLOT-FIRST ITEM RESOLUTION
    # (moved verbatim from MenuRepository._resolve_item_query_from_explicit_slot_normalized)
    # ------------------------------------------------------------------

    def resolve_from_slot(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult | None:
        if not normalized_text:
            return None

        candidates = self._indexer.candidate_items(normalized_text)

        scored_items: list[tuple[float, MenuItem]] = []
        for item in candidates:
            if not item.available:
                continue
            score = self._scorer.score_item_labels(normalized_text, item)
            if score > 0:
                scored_items.append((score, item))

        if not scored_items:
            return self._not_found_result(normalized_text)

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_item = scored_items[0]
        second_score = scored_items[1][0] if len(scored_items) > 1 else 0.0

        clear_winner = best_score >= self._scorer.slot_clear_winner and (
            second_score == 0.0
            or best_score - second_score >= self._scorer.slot_gap
            or best_score >= second_score * self._scorer.slot_ratio
        )

        if clear_winner:
            return MenuQueryResult(type=MenuQueryType.ITEM, item=best_item)

        close_items: list[MenuItem] = []
        seen_item_ids: set[str] = set()

        for score, item in scored_items:
            if score < best_score * self._scorer.slot_close_band:
                break
            if item.item_id in seen_item_ids:
                continue
            seen_item_ids.add(item.item_id)
            close_items.append(item)

        if len(close_items) == 1 and best_score >= self._scorer.slot_fallback_single:
            return MenuQueryResult(type=MenuQueryType.ITEM, item=close_items[0])

        if len(close_items) > 1:
            return MenuQueryResult(
                type=MenuQueryType.ITEM_AMBIGUOUS,
                matched_items=close_items[:limit],
            )

        return self._not_found_result(normalized_text)

    # ------------------------------------------------------------------
    # GENERAL FREE-TEXT RESOLUTION
    # (moved verbatim from MenuRepository.resolve_menu_query_normalized)
    # ------------------------------------------------------------------

    def resolve_free_text(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult:
        if not normalized_text:
            return self._not_found_result(normalized_text)

        category_result = self._resolve_category(normalized_text, limit)
        if category_result is not None:
            return category_result

        candidates = self._indexer.candidate_items(normalized_text)

        scored_items: list[tuple[float, MenuItem]] = []
        for item in candidates:
            if not item.available:
                continue
            score = self._scorer.score_item_labels(normalized_text, item)
            if score > 0:
                scored_items.append((score, item))

        if not scored_items:
            return self._not_found_result(normalized_text)

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_item = scored_items[0]

        strong_items: list[MenuItem] = []
        seen_item_ids: set[str] = set()

        for score, item in scored_items:
            if score < best_score * self._scorer.free_strong_band:
                break
            if item.item_id in seen_item_ids:
                continue
            seen_item_ids.add(item.item_id)
            strong_items.append(item)

        if best_score >= self._scorer.free_clear_winner and len(strong_items) == 1:
            return MenuQueryResult(type=MenuQueryType.ITEM, item=best_item)

        if len(strong_items) > 1:
            return MenuQueryResult(
                type=MenuQueryType.ITEM_AMBIGUOUS,
                matched_items=strong_items[:limit],
            )

        return self._not_found_result(normalized_text)

    # ------------------------------------------------------------------
    # IDLE AVAILABILITY RESOLUTION
    # (moved verbatim from MenuRepository.resolve_idle_availability_query_normalized)
    # ------------------------------------------------------------------

    def resolve_idle_availability(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult:
        if not normalized_text:
            return self._not_found_result(normalized_text)

        category_result = self._resolve_category(normalized_text, limit)
        if category_result is not None:
            return category_result

        candidate_ids: set[str] = set()

        for entry in self._store.find_entity(normalized_text, allowed_types={"item"}):
            item_id = entry.get("item_id")
            if item_id and self._store.is_discoverable_item(item_id):
                candidate_ids.add(item_id)

        exact_item = self._store.find_item_exact(normalized_text)
        if exact_item is not None and self._store.is_discoverable_item(exact_item.item_id):
            candidate_ids.add(exact_item.item_id)

        for item_id in self._store.find_item_ids_by_alias(normalized_text):
            if self._store.is_discoverable_item(item_id):
                candidate_ids.add(item_id)

        for item_id in self._store.find_item_ids_by_voice_label(normalized_text):
            if self._store.is_discoverable_item(item_id):
                candidate_ids.add(item_id)

        candidates = (
            [
                self._store.get_item(item_id)
                for item_id in candidate_ids
                if item_id in self._store.items
            ]
            if candidate_ids
            else self._store.iter_discoverable_items()
        )

        scored_items: list[tuple[float, MenuItem]] = []
        for item in candidates:
            if not item.available:
                continue
            score = self._scorer.score_item_labels(normalized_text, item)
            if score > 0:
                scored_items.append((score, item))

        if not scored_items:
            modifier_hits = self._store.find_modifier_entities(normalized_text)
            if modifier_hits:
                return MenuQueryResult(
                    type=MenuQueryType.NOT_FOUND,
                    suggested_items=[],
                    suggested_categories=[],
                )
            return self._not_found_result(normalized_text)

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_item = scored_items[0]

        strong_items: list[MenuItem] = []
        seen_item_ids: set[str] = set()

        for score, item in scored_items:
            if score < best_score * self._scorer.free_strong_band:
                break
            if item.item_id in seen_item_ids:
                continue
            seen_item_ids.add(item.item_id)
            strong_items.append(item)

        if best_score >= self._scorer.free_clear_winner and len(strong_items) == 1:
            return MenuQueryResult(type=MenuQueryType.ITEM, item=best_item)

        if len(strong_items) > 1:
            return MenuQueryResult(
                type=MenuQueryType.ITEM_AMBIGUOUS,
                matched_items=strong_items[:limit],
            )

        return self._not_found_result(normalized_text)

    # ------------------------------------------------------------------
    # EXISTING API — resolve_item (ItemResolution path)
    # (moved verbatim from MenuRepository.resolve_item_normalized)
    # ------------------------------------------------------------------

    def resolve_item(self, normalized_text: str) -> ItemResolution | None:
        if not normalized_text:
            return None

        candidates = self._indexer.candidate_items(normalized_text)

        best_item: MenuItem | None = None
        best_score = 0.0

        for item in candidates:
            score = self._scorer.score_item_labels(normalized_text, item)
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None or best_score < self._scorer.legacy_resolve:
            return None

        return ItemResolution(item=best_item, score=best_score)

    # ------------------------------------------------------------------
    # MODIFIER AVAILABILITY RESOLUTION
    # (moved verbatim from MenuRepository.resolve_modifier_availability_for_item_normalized)
    # ------------------------------------------------------------------

    def resolve_modifier_availability(
        self,
        *,
        normalized_text: str,
        item_id: str,
    ) -> dict | None:
        if not normalized_text or not item_id or item_id not in self._store.items:
            return None

        item = self._store.get_item(item_id)

        for group in item.modifier_groups:
            label_map = {
                choice.modifier_id: choice.voice_labels
                for choice in group.choices
            }
            matched_ids = self.resolve_modifier_group(
                normalized_text=normalized_text,
                group_id=group.group_id,
                candidate_names_by_id=label_map,
            )
            if len(matched_ids) == 1:
                choice = next(
                    (c for c in group.choices if c.modifier_id == matched_ids[0]),
                    None,
                )
                if choice is not None:
                    return {
                        "match_type": "modifier",
                        "group_name": group.name,
                        "modifier_name": choice.name,
                        "price_cents": choice.price_cents,
                    }

        for group in item.side_groups:
            label_map = {
                choice.item_id: choice.voice_labels
                for choice in group.choices
            }
            matched_ids = self.resolve_side_group(
                normalized_text=normalized_text,
                group_id=group.group_id,
                candidate_names_by_id=label_map,
            )
            if len(matched_ids) == 1:
                choice = next(
                    (c for c in group.choices if c.item_id == matched_ids[0]),
                    None,
                )
                if choice is not None:
                    return {
                        "match_type": "side",
                        "group_name": group.name,
                        "item_name": choice.name,
                    }

        return None
