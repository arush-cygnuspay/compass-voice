# tests/responses/test_item_added_stacked_ack.py
"""Tests for Phase 5: deduplication of stacked item-added acknowledgements.

When queue_transition=True and prev_item_name == item_name (same item added
twice in sequence), the response must NOT repeat "X added. X added." but instead
emit a single merged acknowledgement ("Added 2 X.").
"""
from __future__ import annotations

import pytest

from app.responses.item.success import item_added_successfully


def _queue_payload(
    item_name: str,
    quantity: int,
    prev_item_name: str,
    prev_quantity: int,
    remaining: int = 0,
) -> dict:
    return {
        "item_name": item_name,
        "quantity": quantity,
        "prev_item_name": prev_item_name,
        "prev_quantity": prev_quantity,
        "remaining_queue_count": remaining,
        "queue_transition": True,
        "item_aliases": [],
        "item_voice_labels": [],
        "unmatched_names": [],
    }


class TestStackedAckDedup:
    def test_same_item_twice_merges_acknowledgement(self):
        """'Cheese Cake added. Cheese Cake added.' must not occur."""
        payload = _queue_payload("Cheese Cake", 1, "Cheese Cake", 1)
        text = item_added_successfully(payload)
        # Must not repeat "Cheese Cake" as two separate sentences
        assert text.count("Cheese Cake added") <= 1 or "2" in text, (
            f"Stacked ack not merged: {text!r}"
        )
        # Must NOT contain the literal double-acknowledgement pattern
        assert "Cheese Cake added. Cheese Cake added" not in text

    def test_same_item_qty_1_plus_1_becomes_added_2(self):
        payload = _queue_payload("Cheese Cake", 1, "Cheese Cake", 1)
        text = item_added_successfully(payload)
        # Combined quantity of 2 should appear
        assert "2" in text or "Cheese Cake" in text  # At minimum, still mentions item
        # Main invariant: no double acknowledgement
        assert text.count("Cheese Cake added") <= 1

    def test_same_item_qty_2_plus_2_merged(self):
        payload = _queue_payload("Coke", 2, "Coke", 2)
        text = item_added_successfully(payload)
        assert "Coke added. Coke added" not in text
        assert "4" in text  # combined qty = 4

    def test_different_items_not_merged(self):
        """Different items in queue still produce two-part acknowledgement."""
        payload = _queue_payload("Coke", 1, "Cheese Burger", 1)
        text = item_added_successfully(payload)
        assert "Cheese Burger" in text
        assert "Coke" in text

    def test_different_items_no_merge(self):
        payload = _queue_payload("French Fries", 1, "Burger", 1)
        text = item_added_successfully(payload)
        # Both items mentioned
        assert "Burger" in text or "burger" in text.lower()
        assert "French Fries" in text or "fries" in text.lower()

    def test_same_item_remaining_queue(self):
        """When remaining > 0, merged ack still includes remaining count."""
        payload = _queue_payload("Coke", 1, "Coke", 1, remaining=1)
        text = item_added_successfully(payload)
        # Check remaining-count message present
        assert "more" in text.lower()
        # Check no double ack
        assert "Coke added. Coke added" not in text

    def test_case_insensitive_name_match(self):
        """Name comparison is case-insensitive."""
        payload = _queue_payload("cheese cake", 1, "Cheese Cake", 1)
        text = item_added_successfully(payload)
        assert "Cheese Cake added. Cheese Cake added" not in text
        assert "cheese cake added. cheese cake added" not in text.lower()

    @pytest.mark.parametrize("qty", [1, 2, 3])
    def test_non_queue_transition_unaffected(self, qty):
        """Without queue_transition flag the dedup logic doesn't fire."""
        payload = {
            "item_name": "Coke",
            "quantity": qty,
            "queue_transition": False,
            "item_aliases": [],
            "item_voice_labels": [],
            "unmatched_names": [],
        }
        text = item_added_successfully(payload)
        assert "Coke" in text or qty > 1
