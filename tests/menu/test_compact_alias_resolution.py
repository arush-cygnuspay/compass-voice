# tests/menu/test_compact_alias_resolution.py
"""Phase D regression tests — joined-token forms must resolve to their menu items."""
from __future__ import annotations

import pytest

from app.menu.store import MenuStore
from app.nlu.query_normalization.text_preprocessor import normalize_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _voice_labels(store: MenuStore, item_name: str) -> list[str]:
    """Return voice_labels for the item whose normalized_name matches item_name."""
    norm = normalize_text(item_name)
    item = store.find_item_exact(norm)
    if item is None:
        # Fall back to searching by name substring so tests work against any menu.
        for it in store.items.values():
            if normalize_text(it.name) == norm:
                return list(it.voice_labels)
    if item is None:
        return []
    return list(item.voice_labels)


# ---------------------------------------------------------------------------
# _voice_label_variants unit tests (using MenuStore directly)
# ---------------------------------------------------------------------------

class TestVoiceLabelVariants:
    """_voice_label_variants must emit the joined-token form for multi-word labels."""

    def test_two_word_label_produces_joined(self):
        from app.menu.store import MenuStore
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        store_mock = MenuStore.__new__(MenuStore)
        store_mock.menu_path = None
        store_mock.entity_index_path = None
        store_mock.items = {}
        store_mock.categories = {}
        store_mock.entity_index = {}
        store_mock._item_by_name = {}
        store_mock._item_ids_by_alias = {}
        store_mock._item_ids_by_voice_label = {}
        store_mock._category_name_index = {}
        store_mock._discoverable_item_ids = set()
        store_mock._modifier_entries_by_name = {}
        store_mock._side_ids_by_group_and_label = {}
        store_mock._modifier_ids_by_group_and_label = {}

        variants = store_mock._voice_label_variants("cheese burger")
        assert "cheeseburger" in variants, f"Expected 'cheeseburger' in {variants}"

    def test_three_word_label_produces_joined(self):
        from app.menu.store import MenuStore

        store_mock = MenuStore.__new__(MenuStore)
        for attr in (
            "menu_path", "entity_index_path", "items", "categories",
            "entity_index", "_item_by_name", "_item_ids_by_alias",
            "_item_ids_by_voice_label", "_category_name_index",
            "_discoverable_item_ids", "_modifier_entries_by_name",
            "_side_ids_by_group_and_label", "_modifier_ids_by_group_and_label",
        ):
            setattr(store_mock, attr, {} if attr not in ("_discoverable_item_ids",) else set())

        variants = store_mock._voice_label_variants("double bacon burger")
        assert "doublebaconburger" in variants, f"Expected 'doublebaconburger' in {variants}"

    def test_four_word_label_produces_joined(self):
        from app.menu.store import MenuStore

        store_mock = MenuStore.__new__(MenuStore)
        for attr in (
            "menu_path", "entity_index_path", "items", "categories",
            "entity_index", "_item_by_name", "_item_ids_by_alias",
            "_item_ids_by_voice_label", "_category_name_index",
            "_modifier_entries_by_name", "_side_ids_by_group_and_label",
            "_modifier_ids_by_group_and_label",
        ):
            setattr(store_mock, attr, {})
        store_mock._discoverable_item_ids = set()

        variants = store_mock._voice_label_variants("build your own pizza")
        assert "buildyourownpizza" in variants, f"Expected 'buildyourownpizza' in {variants}"

    def test_short_two_word_label_not_joined(self):
        """Labels whose joined form is < 7 chars should NOT produce a joined variant."""
        from app.menu.store import MenuStore

        store_mock = MenuStore.__new__(MenuStore)
        for attr in (
            "menu_path", "entity_index_path", "items", "categories",
            "entity_index", "_item_by_name", "_item_ids_by_alias",
            "_item_ids_by_voice_label", "_category_name_index",
            "_modifier_entries_by_name", "_side_ids_by_group_and_label",
            "_modifier_ids_by_group_and_label",
        ):
            setattr(store_mock, attr, {})
        store_mock._discoverable_item_ids = set()

        # "hot dog" → "hotdog" is only 6 chars — not emitted.
        variants = store_mock._voice_label_variants("hot dog")
        assert "hotdog" not in variants


# ---------------------------------------------------------------------------
# Index-level lookup tests (require demo menu)
# ---------------------------------------------------------------------------

def _make_store():
    from pathlib import Path
    data_root = Path(__file__).resolve().parents[2] / "app" / "data" / "restaurants" / "demo"
    return MenuStore(
        menu_path=data_root / "menu.json",
        entity_index_path=data_root / "entity_index.json",
    )


class TestCompactAliasLookup:
    """find_item_ids_by_voice_label must find items via their joined-token voice labels."""

    @pytest.fixture(scope="class")
    def store(self):
        return _make_store()

    def _find_by_joined(self, store, joined_label):
        """Return item ids found via the voice_label index for a joined form."""
        return store.find_item_ids_by_voice_label(joined_label)

    def test_cheese_burger_spaced(self, store):
        """'cheese burger' (spaced) must resolve through voice_label index."""
        ids = store.find_item_ids_by_voice_label("cheese burger")
        # This checks the direct voice_label; item must exist in the demo menu.
        # If menu has no Cheese Burger we skip gracefully.
        if not ids:
            # Try normalized_name lookup as fallback assertion.
            item = store.find_item_exact("cheese burger")
            if item is None:
                pytest.skip("Cheese Burger not in demo menu — test is schema-dependent")
        assert ids, "Expected at least one item id for 'cheese burger'"

    def test_cheese_burger_joined(self, store):
        """'cheeseburger' (joined) must resolve via voice_label index."""
        ids = store.find_item_ids_by_voice_label("cheeseburger")
        item = store.find_item_exact("cheese burger")
        if item is None:
            pytest.skip("Cheese Burger not in demo menu — test is schema-dependent")
        assert item.item_id in ids, (
            f"'cheeseburger' should resolve to item_id={item.item_id}. "
            f"Voice labels: {list(item.voice_labels)}"
        )

    def test_double_bacon_burger_joined(self, store):
        ids = store.find_item_ids_by_voice_label("doublebaconburger")
        item = store.find_item_exact("double bacon burger")
        if item is None:
            pytest.skip("Double Bacon Burger not in demo menu")
        assert item.item_id in ids, (
            f"'doublebaconburger' should resolve to {item.item_id}. "
            f"Voice labels: {list(item.voice_labels)}"
        )

    def test_unknown_joined_does_not_overmatch(self, store):
        """A made-up joined token must not match real items."""
        ids = store.find_item_ids_by_voice_label("xyzgarbageitem123")
        assert ids == [], f"Expected no match for nonsense token, got {ids}"
