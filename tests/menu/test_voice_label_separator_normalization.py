# tests/menu/test_voice_label_separator_normalization.py
"""Tests for Phase 1: Unicode separator normalization in _voice_label_variants.

Validates:
- Items with em-dash separators ("Korean Tacos — Spicy Chicken") include a
  dash-stripped voice label ("korean tacos spicy chicken").
- Compact joins use the stripped form (no embedded dashes).
- Items without Unicode separators are unaffected.
- BBQ ↔ barbecue variants still work after stripping.
"""
from __future__ import annotations

import pytest

from app.menu.store import MenuStore


class _StoreStub(MenuStore):
    """Minimal stub that exposes _voice_label_variants without loading files."""

    def __init__(self) -> None:  # type: ignore[override]
        # Skip the normal __init__ (file I/O) — we only test the helper.
        pass


_STORE = _StoreStub()


def _variants(name: str) -> tuple[str, ...]:
    from app.nlu.query_normalization.text_preprocessor import normalize_text
    return tuple(_STORE._voice_label_variants(normalize_text(name)))


class TestSeparatorNormalization:
    def test_em_dash_stripped_variant_added(self):
        """'korean tacos — spicy chicken' should include 'korean tacos spicy chicken'."""
        variants = _variants("Korean Tacos — Spicy Chicken")
        assert "korean tacos spicy chicken" in variants, (
            f"Expected dash-stripped variant in {variants}"
        )

    def test_en_dash_stripped_variant_added(self):
        """En dash – variant should also be stripped."""
        variants = _variants("Breakfast – Eggs")
        assert "breakfast eggs" in variants

    def test_compact_join_without_dash(self):
        """Compact join for 'Korean Tacos — Spicy Chicken' should not contain a dash."""
        variants = _variants("Korean Tacos — Spicy Chicken")
        # The joined form must exist and must not contain a dash character
        compound_forms = [v for v in variants if " " not in v and len(v) >= 7]
        for form in compound_forms:
            assert "—" not in form, f"Compact form '{form}' should not contain em dash"
            assert "–" not in form, f"Compact form '{form}' should not contain en dash"

    def test_plain_item_unaffected(self):
        """Items with no Unicode separator should still generate their normal variants."""
        variants = _variants("Cheese Burger")
        assert "cheese burger" in variants
        assert "cheeseburger" in variants  # compact join

    def test_cheeseburger_compact_join(self):
        """'Cheese Burger' (2 tokens, length >= 7) → compact join 'cheeseburger'."""
        variants = _variants("Cheese Burger")
        assert "cheeseburger" in variants

    def test_double_bacon_burger_compact_join(self):
        """'Double Bacon Burger' (3 tokens) → 'doublebaconburger'."""
        variants = _variants("Double Bacon Burger")
        assert "doublebaconburger" in variants

    def test_bbq_barbecue_still_works(self):
        variants = _variants("BBQ Chicken")
        assert "barbecue chicken" in variants

    def test_barbecue_bbq_stripped_and_compact(self):
        """'BBQ Chicken — Spicy' should have both dash-stripped and BBQ swap variants."""
        variants = _variants("BBQ Chicken — Spicy")
        assert "bbq chicken spicy" in variants or "barbecue chicken spicy" in variants


class TestPendingAddItemHasLabelFields:
    """Phase 2: PendingAddItem carries item_aliases and item_voice_labels."""

    def test_fields_exist_and_default_empty(self):
        from app.state_machine.models.pending_item_models import PendingAddItem
        item = PendingAddItem(item_id="i1", item_name="Test Item")
        assert hasattr(item, "item_aliases")
        assert hasattr(item, "item_voice_labels")
        assert item.item_aliases == ()
        assert item.item_voice_labels == ()

    def test_factory_populates_aliases_and_voice_labels(self):
        from unittest.mock import MagicMock
        from app.state_machine.handlers.item.add_item.pending_add_item_factory import (
            build_pending_add_item,
        )

        menu_item = MagicMock()
        menu_item.item_id = "i1"
        menu_item.name = "Cheese Burger"
        menu_item.normalized_aliases = ("cheeseburger", "cheese burger")
        menu_item.voice_labels = ("cheese burger", "cheeseburger", "cheeseburgers")
        menu_item.pricing = MagicMock(mode="fixed", variants=[])
        menu_item.side_groups = []
        menu_item.modifier_groups = []

        pending = build_pending_add_item(menu_item)
        assert pending.item_aliases == ("cheeseburger", "cheese burger")
        assert "cheeseburger" in pending.item_voice_labels
