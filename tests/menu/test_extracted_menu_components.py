# tests/menu/test_extracted_menu_components.py
"""Tests for the four extracted menu components: MenuScorer, MenuIndexer, MenuMatcher, MenuQueryService."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.menu.indexer import MenuIndexer
from app.menu.matcher import MenuMatcher
from app.menu.models import ItemResolution, MenuItem
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.query_service import MenuQueryService
from app.menu.scorer import (
    CANDIDATE_CLEAR_WINNER_THRESHOLD,
    CANDIDATE_GAP_REQUIRED,
    CANDIDATE_RATIO_REQUIRED,
    FREE_CLEAR_WINNER_THRESHOLD,
    FREE_STRONG_BAND_RATIO,
    GROUP_MATCH_BAND_RATIO,
    LEGACY_RESOLVE_THRESHOLD,
    SIMILARITY_MINIMUM_THRESHOLD,
    SLOT_CLEAR_WINNER_THRESHOLD,
    SLOT_CLOSE_BAND_RATIO,
    SLOT_FALLBACK_SINGLE_THRESHOLD,
    SLOT_GAP_REQUIRED,
    SLOT_RATIO_REQUIRED,
    MenuScorer,
)
from app.menu.store import MenuStore
from app.nlu.nlu_result import SlotValue

_DATA_ROOT = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "steves_grill"


def _build_store() -> MenuStore:
    return MenuStore(
        menu_path=_DATA_ROOT / "menu.json",
        entity_index_path=_DATA_ROOT / "entity_index.json",
    )


def _build_service() -> MenuQueryService:
    return MenuQueryService(_build_store())


# ===========================================================================
# MenuScorer
# ===========================================================================


class TestMenuScorer:
    def test_threshold_constants_are_module_level(self):
        assert CANDIDATE_CLEAR_WINNER_THRESHOLD == 6.0
        assert CANDIDATE_GAP_REQUIRED == 0.9
        assert CANDIDATE_RATIO_REQUIRED == 1.15
        assert SLOT_CLEAR_WINNER_THRESHOLD == 5.8
        assert SLOT_GAP_REQUIRED == 0.9
        assert SLOT_RATIO_REQUIRED == 1.18
        assert SLOT_CLOSE_BAND_RATIO == 0.92
        assert SLOT_FALLBACK_SINGLE_THRESHOLD == 5.4
        assert FREE_STRONG_BAND_RATIO == 0.85
        assert FREE_CLEAR_WINNER_THRESHOLD == 6.0
        assert GROUP_MATCH_BAND_RATIO == 0.90
        assert SIMILARITY_MINIMUM_THRESHOLD == 4.8
        assert LEGACY_RESOLVE_THRESHOLD == 6.5

    def test_class_attributes_match_constants(self):
        s = MenuScorer()
        assert s.candidate_clear_winner == CANDIDATE_CLEAR_WINNER_THRESHOLD
        assert s.slot_clear_winner == SLOT_CLEAR_WINNER_THRESHOLD
        assert s.free_clear_winner == FREE_CLEAR_WINNER_THRESHOLD
        assert s.legacy_resolve == LEGACY_RESOLVE_THRESHOLD

    def test_score_item_labels_exact_name_scores_high(self):
        scorer = MenuScorer()
        item = MagicMock(spec=MenuItem)
        item.normalized_name = "chicken burger"
        item.normalized_aliases = ()
        item.voice_labels = ()
        score = scorer.score_item_labels("chicken burger", item)
        assert score >= scorer.free_clear_winner

    def test_score_item_labels_empty_returns_zero(self):
        scorer = MenuScorer()
        item = MagicMock(spec=MenuItem)
        item.normalized_name = "chicken burger"
        item.normalized_aliases = ()
        item.voice_labels = ()
        score = scorer.score_item_labels("", item)
        assert score == 0.0

    def test_score_labels_picks_best_from_tuple(self):
        scorer = MenuScorer()
        score_multi = scorer.score_labels("burger", ("burger", "sandwich", "wrap"))
        score_single = scorer.score_labels("burger", ("burger",))
        assert score_multi == score_single

    def test_score_labels_empty_labels_returns_zero(self):
        scorer = MenuScorer()
        assert scorer.score_labels("burger", ()) == 0.0

    def test_thresholds_overridable_per_instance(self):
        scorer = MenuScorer()
        scorer.free_clear_winner = 99.0
        assert scorer.free_clear_winner == 99.0
        default = MenuScorer()
        assert default.free_clear_winner == FREE_CLEAR_WINNER_THRESHOLD


# ===========================================================================
# MenuIndexer
# ===========================================================================


class TestMenuIndexer:
    def test_candidate_items_known_item_returns_nonempty(self):
        store = _build_store()
        indexer = MenuIndexer(store)
        results = indexer.candidate_items("chicken burger")
        assert len(results) > 0

    def test_candidate_items_unknown_text_falls_back_to_all_items(self):
        store = _build_store()
        indexer = MenuIndexer(store)
        all_count = len(store.items)
        results = indexer.candidate_items("xyzzy not a real item at all")
        assert len(results) == all_count

    def test_candidate_items_returns_menuitem_objects(self):
        store = _build_store()
        indexer = MenuIndexer(store)
        results = indexer.candidate_items("chicken")
        assert all(isinstance(r, MenuItem) for r in results)

    def test_has_item_evidence_known_item_true(self):
        store = _build_store()
        indexer = MenuIndexer(store)
        assert indexer.has_item_evidence("chicken burger") is True

    def test_has_item_evidence_unknown_item_false(self):
        store = _build_store()
        indexer = MenuIndexer(store)
        assert indexer.has_item_evidence("xyzzy completely unknown") is False

    def test_has_item_evidence_empty_string_false(self):
        store = _build_store()
        indexer = MenuIndexer(store)
        assert indexer.has_item_evidence("") is False


# ===========================================================================
# MenuMatcher
# ===========================================================================


class TestMenuMatcher:
    def _make_matcher(self) -> MenuMatcher:
        store = _build_store()
        from app.menu.indexer import MenuIndexer
        from app.menu.scorer import MenuScorer
        service = MenuQueryService(store)
        return service._matcher

    def test_resolve_item_known_item_returns_item_resolution(self):
        matcher = self._make_matcher()
        result = matcher.resolve_item("chicken burger")
        assert result is not None
        assert isinstance(result, ItemResolution)
        assert result.score >= MenuScorer.legacy_resolve

    def test_resolve_item_unknown_item_returns_none(self):
        matcher = self._make_matcher()
        result = matcher.resolve_item("xyzzy not a food at all")
        assert result is None

    def test_resolve_item_empty_returns_none(self):
        matcher = self._make_matcher()
        assert matcher.resolve_item("") is None

    def test_resolve_free_text_known_returns_item_type(self):
        matcher = self._make_matcher()
        result = matcher.resolve_free_text("chicken burger")
        assert result.type in (MenuQueryType.ITEM, MenuQueryType.ITEM_AMBIGUOUS)

    def test_resolve_free_text_empty_returns_not_found(self):
        matcher = self._make_matcher()
        result = matcher.resolve_free_text("")
        assert result.type == MenuQueryType.NOT_FOUND

    def test_resolve_from_slot_known_item_returns_result(self):
        matcher = self._make_matcher()
        result = matcher.resolve_from_slot("chicken burger")
        assert result is not None
        assert result.type in (MenuQueryType.ITEM, MenuQueryType.ITEM_AMBIGUOUS, MenuQueryType.NOT_FOUND)

    def test_resolve_from_slot_empty_returns_none(self):
        matcher = self._make_matcher()
        assert matcher.resolve_from_slot("") is None

    def test_resolve_within_candidates_no_candidates_returns_none(self):
        matcher = self._make_matcher()
        result = matcher.resolve_within_candidates(
            normalized_text="chicken burger",
            candidate_item_ids=[],
        )
        assert result is None

    def test_resolve_within_candidates_empty_text_returns_none(self):
        matcher = self._make_matcher()
        result = matcher.resolve_within_candidates(
            normalized_text="",
            candidate_item_ids=["some-id"],
        )
        assert result is None

    def test_resolve_idle_availability_known_returns_result(self):
        matcher = self._make_matcher()
        result = matcher.resolve_idle_availability("chicken burger")
        assert result.type in (MenuQueryType.ITEM, MenuQueryType.ITEM_AMBIGUOUS, MenuQueryType.NOT_FOUND, MenuQueryType.CATEGORY, MenuQueryType.CATEGORY_SINGLE_ITEM)


# ===========================================================================
# MenuQueryService
# ===========================================================================


class TestMenuQueryService:
    def test_get_item_returns_menu_item(self):
        service = _build_service()
        store = _build_store()
        any_id = next(iter(store.items))
        item = service.get_item(any_id)
        assert isinstance(item, MenuItem)

    def test_resolve_menu_query_exact_name_returns_item(self):
        service = _build_service()
        result = service.resolve_menu_query("chicken burger")
        assert result.type in (MenuQueryType.ITEM, MenuQueryType.ITEM_AMBIGUOUS)

    def test_resolve_menu_query_empty_returns_not_found(self):
        service = _build_service()
        result = service.resolve_menu_query("")
        assert result.type == MenuQueryType.NOT_FOUND

    def test_resolve_menu_query_normalized_delegates_correctly(self):
        service = _build_service()
        raw = service.resolve_menu_query("Chicken Burger")
        normalized = service.resolve_menu_query_normalized("chicken burger")
        assert raw.type == normalized.type

    def test_resolve_category_query_known_category_returns_category_type(self):
        service = _build_service()
        result = service.resolve_category_query("drinks")
        if result is not None:
            assert result.type in (MenuQueryType.CATEGORY, MenuQueryType.CATEGORY_SINGLE_ITEM)

    def test_resolve_category_query_normalized_unknown_returns_none(self):
        service = _build_service()
        result = service.resolve_category_query_normalized("xyzzy not a category at all")
        assert result is None

    def test_build_not_found_recovery_normalized_returns_tuple(self):
        service = _build_service()
        items, categories = service.build_not_found_recovery_normalized("zorp")
        assert isinstance(items, list)
        assert isinstance(categories, list)

    def test_build_not_found_recovery_normalized_respects_limits(self):
        service = _build_service()
        items, categories = service.build_not_found_recovery_normalized(
            "chicken", item_limit=2, category_limit=3
        )
        assert len(items) <= 2
        assert len(categories) <= 3

    def test_resolve_menu_query_from_slots_uses_item_slot(self):
        service = _build_service()
        result = service.resolve_menu_query_from_slots_normalized(
            normalized_user_text="",
            slots=(SlotValue(name="ITEM", value="Chicken Burger"),),
            fallback_to_text=False,
        )
        assert result.type in (MenuQueryType.ITEM, MenuQueryType.ITEM_AMBIGUOUS, MenuQueryType.NOT_FOUND)

    def test_resolve_menu_query_from_slots_no_slots_not_found(self):
        service = _build_service()
        result = service.resolve_menu_query_from_slots_normalized(
            normalized_user_text="",
            slots=(),
            fallback_to_text=False,
        )
        assert result.type == MenuQueryType.NOT_FOUND

    def test_resolve_menu_query_from_slots_prefers_evidenced_slot(self):
        service = _build_service()
        result = service.resolve_menu_query_from_slots_normalized(
            normalized_user_text="not a thing chicken burger coke",
            slots=(
                SlotValue(name="ITEM", value="not a thing"),
                SlotValue(name="ITEM", value="Chicken Burger"),
                SlotValue(name="ITEM", value="Coke"),
            ),
            fallback_to_text=False,
        )
        assert result.type == MenuQueryType.ITEM
        assert result.item is not None
        assert result.item.name == "Chicken Burger"

    def test_resolve_item_normalized_empty_returns_none(self):
        service = _build_service()
        result = service.resolve_item_normalized("")
        assert result is None

    def test_resolve_item_known_item_returns_item_resolution(self):
        service = _build_service()
        result = service.resolve_item("chicken burger")
        assert result is not None
        assert isinstance(result, ItemResolution)

    def test_resolve_idle_availability_query_normalized_returns_query_result(self):
        service = _build_service()
        result = service.resolve_idle_availability_query_normalized("chicken burger")
        assert isinstance(result, MenuQueryResult)

    def test_resolve_modifier_availability_unknown_item_returns_none(self):
        service = _build_service()
        result = service.resolve_modifier_availability_for_item_normalized(
            normalized_text="grilled",
            item_id="nonexistent-item-id",
        )
        assert result is None

    def test_resolve_item_within_candidates_empty_returns_none(self):
        service = _build_service()
        result = service.resolve_item_within_candidates_normalized(
            normalized_text="burger",
            candidate_item_ids=[],
        )
        assert result is None

    def test_resolve_side_choice_within_group_unknown_group_returns_empty(self):
        service = _build_service()
        result = service.resolve_side_choice_within_group_normalized(
            normalized_text="fries",
            group_id="nonexistent-group-id",
            candidate_names_by_id={},
        )
        assert result == []

    def test_resolve_modifier_choice_within_group_unknown_returns_empty(self):
        service = _build_service()
        result = service.resolve_modifier_choice_within_group_normalized(
            normalized_text="extra cheese",
            group_id="nonexistent-group-id",
            candidate_names_by_id={},
        )
        assert result == []
