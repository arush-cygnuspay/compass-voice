# tests/responses/test_voice_response_compact_format.py
"""Tests for compact voice response formatting.

Voice responses for item-added and order confirmation must omit modifier/side
details — only item name + quantity are spoken.  Modifier/side data is preserved
in cart payloads and kitchen data; it is intentionally excluded from TTS text.
"""
from __future__ import annotations

import pytest

from app.responses.cart_responses import render_cart_line_voice_compact, render_checkout_review_summary
from app.responses.item.success import item_added_successfully


# ---------------------------------------------------------------------------
# item_added_successfully — modifiers and sides omitted
# ---------------------------------------------------------------------------

def _item_payload(item_name: str, quantity: int = 1, **extra) -> dict:
    return {
        "item_name": item_name,
        "quantity": quantity,
        "item_aliases": [],
        "item_voice_labels": [],
        "unmatched_names": [],
        **extra,
    }


def test_item_added_response_omits_modifiers():
    payload = _item_payload("Bourbon Chicken", spoken_modifiers=["extra sauce", "no onions"])
    text = item_added_successfully(payload)
    assert "extra sauce" not in text
    assert "no onions" not in text
    assert "with" not in text
    assert "Bourbon Chicken added" in text


def test_item_added_response_omits_sides():
    payload = _item_payload("Chicken Burger", spoken_modifiers=["with Coke"])
    text = item_added_successfully(payload)
    assert "Coke" not in text
    assert "Chicken Burger added" in text


def test_item_added_response_preserves_quantity():
    text = item_added_successfully(_item_payload("Chicken Burger", quantity=2))
    assert "2" in text
    assert "Chicken Burger" in text


def test_bourbon_chicken_with_fresh_mushroom_added_says_only_bourbon_chicken_added():
    """The exact scenario from the bug report: modifier must not appear in voice."""
    payload = _item_payload("Bourbon Chicken", spoken_modifiers=["Fresh Mushroom"])
    text = item_added_successfully(payload)
    assert text == "Bourbon Chicken added. Would you like anything else?"


def test_item_added_qty_1_no_modifier_clean_response():
    text = item_added_successfully(_item_payload("Zinger Burger"))
    assert text == "Zinger Burger added. Would you like anything else?"


def test_item_added_qty_2_no_modifier_clean_response():
    text = item_added_successfully(_item_payload("Zinger Burger", quantity=2))
    assert text == "Added 2 Zinger Burger. Would you like anything else?"


# ---------------------------------------------------------------------------
# render_cart_line_voice_compact — item name + quantity only
# ---------------------------------------------------------------------------

def test_render_cart_line_voice_compact_qty_1_name_only():
    item = {"quantity": 1, "name": "Chicken Burger", "sides": ["Coke"], "modifiers": ["extra cheese"]}
    assert render_cart_line_voice_compact(item) == "Chicken Burger"


def test_render_cart_line_voice_compact_qty_2_prefixed():
    item = {"quantity": 2, "name": "Chicken Taco", "sides": ["Fries"], "modifiers": []}
    assert render_cart_line_voice_compact(item) == "2 Chicken Taco"


def test_render_cart_line_voice_compact_omits_sides():
    item = {"quantity": 1, "name": "Burger", "sides": ["Coke", "Fries"]}
    result = render_cart_line_voice_compact(item)
    assert "Coke" not in result
    assert "Fries" not in result
    assert "Burger" in result


def test_render_cart_line_voice_compact_omits_modifiers():
    item = {"quantity": 1, "name": "Burger", "modifiers": ["no onions", "extra mayo"]}
    result = render_cart_line_voice_compact(item)
    assert "onions" not in result
    assert "mayo" not in result
    assert "Burger" in result


# ---------------------------------------------------------------------------
# render_checkout_review_summary — compact voice format
# ---------------------------------------------------------------------------

def _summary_payload(items, total="$20.17"):
    return {"items": items, "total": total}


def test_confirm_order_summary_omits_modifiers():
    payload = _summary_payload([
        {"quantity": 1, "name": "Chicken Burger", "sides": [], "modifiers": ["American Cheese", "Plain Bun", "add Mayo"]},
    ])
    text = render_checkout_review_summary(payload)
    assert "American Cheese" not in text
    assert "Plain Bun" not in text
    assert "Mayo" not in text
    assert "Chicken Burger" in text


def test_confirm_order_summary_omits_sides():
    payload = _summary_payload([
        {"quantity": 1, "name": "Chicken Taco", "sides": ["Coke (12 oz.)"], "modifiers": []},
    ])
    text = render_checkout_review_summary(payload)
    assert "Coke" not in text
    assert "12 oz" not in text
    assert "Chicken Taco" in text


def test_confirm_order_summary_omits_side_variants():
    payload = _summary_payload([
        {"quantity": 1, "name": "Burger", "sides": ["Large Fries", "2 Coke"], "modifiers": ["no onions"]},
    ])
    text = render_checkout_review_summary(payload)
    assert "Large Fries" not in text
    assert "2 Coke" not in text
    assert "no onions" not in text
    assert "Burger" in text


def test_confirm_order_summary_preserves_item_quantities():
    payload = _summary_payload([
        {"quantity": 2, "name": "Chicken Burger", "sides": [], "modifiers": []},
        {"quantity": 1, "name": "Chicken Taco", "sides": [], "modifiers": []},
    ])
    text = render_checkout_review_summary(payload)
    assert "2 Chicken Burger" in text
    assert "Chicken Taco" in text


def test_confirm_order_summary_includes_total():
    payload = _summary_payload(
        [{"quantity": 1, "name": "Chicken Burger", "sides": [], "modifiers": []}],
        total="$20.17",
    )
    text = render_checkout_review_summary(payload)
    assert "$20.17" in text


def test_confirm_order_summary_uses_single_confirmation_question():
    """Must not have both 'Should I place the order?' and 'Would you like to checkout?'."""
    payload = _summary_payload([
        {"quantity": 1, "name": "Chicken Burger", "sides": [], "modifiers": []},
        {"quantity": 1, "name": "Chicken Taco", "sides": [], "modifiers": []},
    ])
    text = render_checkout_review_summary(payload)
    assert text.lower().count("should i place the order") == 1
    assert "would you like to checkout" not in text.lower()
    assert text.endswith("Should I place the order?")


def test_confirm_order_summary_full_scenario():
    """The exact scenario from the bug report."""
    payload = _summary_payload([
        {"quantity": 1, "name": "Chicken Burger", "sides": [], "modifiers": ["American Cheese", "Plain Bun", "add Mayo"]},
        {"quantity": 1, "name": "Chicken Taco", "sides": ["Coke (12 oz.)"], "modifiers": []},
    ])
    text = render_checkout_review_summary(payload)
    assert "Please review your order:" in text
    assert "Chicken Burger" in text
    assert "Chicken Taco" in text
    assert "$20.17" in text
    assert "Should I place the order?" in text
    # No modifier/side detail in voice
    assert "American Cheese" not in text
    assert "Mayo" not in text
    assert "Coke" not in text
    # No double question
    assert "Would you like to checkout" not in text
