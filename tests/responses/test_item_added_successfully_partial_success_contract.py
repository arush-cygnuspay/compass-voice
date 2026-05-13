# tests/responses/test_item_added_successfully_partial_success_contract.py
"""Contract tests for the partial_success / unresolved_entities feedback path
in item_added_successfully().

The function has two feedback paths:
  Primary:  partial_success=True AND unresolved_entities non-empty
            → "I couldn't find {entities}." built from structured list
  Legacy:   everything else
            → unmatched_names with _filter_item_labels alias suppression

These tests lock in the selection logic so that neither path fires
spuriously and the two paths do not bleed into each other.
"""
from __future__ import annotations

import pytest

from app.responses.item.success import item_added_successfully


def _base(
    item_name: str = "Test Burger",
    quantity: int = 1,
    *,
    partial_success: bool | None = None,
    unresolved_entities: list[str] | None = None,
    unmatched_names: list[str] | None = None,
    item_aliases: list[str] | None = None,
    item_voice_labels: list[str] | None = None,
) -> dict:
    payload: dict = {
        "item_name": item_name,
        "quantity": quantity,
        "item_aliases": item_aliases or [],
        "item_voice_labels": item_voice_labels or [],
    }
    if partial_success is not None:
        payload["partial_success"] = partial_success
    if unresolved_entities is not None:
        payload["unresolved_entities"] = unresolved_entities
    if unmatched_names is not None:
        payload["unmatched_names"] = unmatched_names
    return payload


class TestPrimaryPath:
    """partial_success=True + unresolved_entities → structured feedback."""

    def test_single_entity_included(self) -> None:
        text = item_added_successfully(_base(partial_success=True, unresolved_entities=["avocado"]))
        assert "I couldn't find avocado." in text

    def test_multiple_entities_formatted(self) -> None:
        text = item_added_successfully(
            _base(partial_success=True, unresolved_entities=["rice", "avocado"])
        )
        assert "I couldn't find rice and avocado." in text

    def test_still_ends_with_would_you_like(self) -> None:
        text = item_added_successfully(_base(partial_success=True, unresolved_entities=["rice"]))
        assert text.endswith("Would you like anything else?")

    def test_item_name_still_present(self) -> None:
        text = item_added_successfully(
            _base("Spicy Tuna Roll", partial_success=True, unresolved_entities=["avocado"])
        )
        assert "Spicy Tuna Roll" in text

    def test_unmatched_names_ignored_when_primary_path_active(self) -> None:
        """unmatched_names must NOT produce extra feedback when primary path fires."""
        text = item_added_successfully(
            _base(
                partial_success=True,
                unresolved_entities=["avocado"],
                unmatched_names=["legacy item"],
            )
        )
        assert "legacy item" not in text
        assert "I couldn't find avocado." in text


class TestPrimaryPathNotFired:
    """Conditions where primary path must NOT produce feedback."""

    def test_partial_success_false_suppresses_entities(self) -> None:
        """Even if unresolved_entities is populated, False gate must block it."""
        text = item_added_successfully(
            _base(partial_success=False, unresolved_entities=["avocado"])
        )
        assert "I couldn't find" not in text

    def test_partial_success_true_empty_entities_no_feedback(self) -> None:
        """partial_success=True with empty list falls through; no primary feedback."""
        text = item_added_successfully(
            _base(partial_success=True, unresolved_entities=[])
        )
        assert "I couldn't find" not in text

    def test_partial_success_missing_suppresses_entities(self) -> None:
        """Absence of partial_success key must not activate primary path."""
        text = item_added_successfully(
            _base(unresolved_entities=["avocado"])  # no partial_success key
        )
        assert "I couldn't find avocado" not in text


class TestLegacyPath:
    """Legacy unmatched_names path: fired when primary path is inactive."""

    def test_unmatched_name_included_when_no_partial_success(self) -> None:
        text = item_added_successfully(_base(unmatched_names=["rice"]))
        assert "I couldn't find rice." in text

    def test_alias_suppressed_from_unmatched(self) -> None:
        """Item alias form in unmatched_names must not echo as "couldn't find"."""
        text = item_added_successfully(
            _base(
                "Spicy Tuna Roll",
                unmatched_names=["spicy tuna"],
                item_aliases=["spicy tuna"],
            )
        )
        assert "I couldn't find" not in text

    def test_voice_label_suppressed_from_unmatched(self) -> None:
        text = item_added_successfully(
            _base(
                "Cheese Burger",
                unmatched_names=["cheeseburger"],
                item_voice_labels=["cheeseburger"],
            )
        )
        assert "I couldn't find" not in text

    def test_unrelated_unmatched_name_passes_through(self) -> None:
        text = item_added_successfully(
            _base(
                "Cheese Burger",
                unmatched_names=["avocado"],
                item_aliases=["cheeseburger"],
            )
        )
        assert "I couldn't find avocado." in text

    def test_empty_unmatched_names_no_feedback(self) -> None:
        text = item_added_successfully(_base(unmatched_names=[]))
        assert "I couldn't find" not in text


class TestCleanAdd:
    """Terminal success (no feedback of any kind) — the most common case."""

    def test_clean_single_item_no_feedback(self) -> None:
        text = item_added_successfully(_base("Coke"))
        assert "I couldn't find" not in text
        assert "Coke added." in text
        assert text.endswith("Would you like anything else?")

    def test_clean_quantity_item_no_feedback(self) -> None:
        text = item_added_successfully(_base("Coke", 2))
        assert "I couldn't find" not in text
        assert "2 Coke" in text

    @pytest.mark.parametrize("qty", [1, 2, 3])
    def test_no_payload_keys_no_feedback(self, qty: int) -> None:
        text = item_added_successfully({"item_name": "Burger", "quantity": qty})
        assert "I couldn't find" not in text

    def test_fuzzy_accepted_item_no_partial_success_no_feedback(self) -> None:
        """Fuzzy match accepted (e.g. 'port stickers' → 'Pot Stickers') with
        partial_success=False must never produce 'I couldn't find' feedback."""
        text = item_added_successfully(
            _base("Pot Stickers", partial_success=False, unresolved_entities=["port stickers"])
        )
        assert "I couldn't find" not in text
        assert "Pot Stickers added." in text

    def test_partial_success_true_with_unresolved_entities_emits_once(self) -> None:
        """Genuine partial success must emit 'I couldn't find' exactly once."""
        text = item_added_successfully(
            _base("Burger", partial_success=True, unresolved_entities=["avocado", "rice"])
        )
        assert text.count("I couldn't find") == 1
        assert "avocado" in text
        assert "rice" in text
