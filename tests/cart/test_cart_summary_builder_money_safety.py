# tests/cart/test_cart_summary_builder_money_safety.py
"""Architecture guard tests for Phase 7.

normalize_food_quantity is a food-quantity-only helper.  Price / money
modules must NOT import it (prices use cents integers, never 0.N encoding).

Also guards that cart_summary_builder price rendering is unchanged by the
quantity normalization changes.
"""
from __future__ import annotations

import importlib
import sys

import pytest


# ---------------------------------------------------------------------------
# Import isolation: money/price modules must not pull in normalize_food_quantity
# ---------------------------------------------------------------------------

_MONEY_MODULES = [
    "app.cart.read_models.cart_summary_builder",
    "app.responses.cart_responses",
]


class TestMoneyModulesDoNotImportNormalizeFoodQuantity:
    @pytest.mark.parametrize("module_path", _MONEY_MODULES)
    def test_module_does_not_import_normalize_food_quantity(self, module_path):
        """normalize_food_quantity must never appear in price/money module namespaces."""
        mod = importlib.import_module(module_path)
        assert not hasattr(mod, "normalize_food_quantity"), (
            f"{module_path} must not expose normalize_food_quantity — "
            "food-quantity decoding must not bleed into price/money paths."
        )

    def test_quantity_formatter_not_imported_by_cart_summary_builder(self):
        import app.cart.read_models.cart_summary_builder as csb
        # The module's globals should not reference normalize_food_quantity
        globals_dict = vars(csb)
        assert "normalize_food_quantity" not in globals_dict

    def test_quantity_formatter_not_imported_by_cart_responses(self):
        import app.responses.cart_responses as cr
        globals_dict = vars(cr)
        assert "normalize_food_quantity" not in globals_dict


# ---------------------------------------------------------------------------
# Price rendering is unaffected: totals / per-item prices unchanged
# ---------------------------------------------------------------------------

class TestPriceRenderingUnchanged:
    """Smoke-test that price strings survive the quantity normalisation patch."""

    def test_cart_responses_price_intact(self):
        from app.responses.cart_responses import render_cart_summary

        payload = {
            "items": [{"quantity": 2, "name": "Burger"}],
            "total": "$17.20",
            "item_count": None,
        }
        text = render_cart_summary(payload)
        assert "$17.20" in text, "Total must be preserved verbatim"

    def test_checkout_review_price_intact(self):
        from app.responses.cart_responses import render_checkout_review_summary

        payload = {
            "items": [{"quantity": 1, "name": "BBQ Chicken Pizza"}],
            "total": "$12.99",
        }
        text = render_checkout_review_summary(payload)
        assert "$12.99" in text

    def test_decimal_encoded_quantity_price_unchanged(self):
        """Quantity 0.2 (=2 items) must not corrupt the price field."""
        from app.responses.cart_responses import render_cart_summary

        payload = {
            "items": [{"quantity": 0.2, "name": "Coke"}],
            "total": "$5.00",
            "item_count": None,
        }
        text = render_cart_summary(payload)
        assert "$5.00" in text


# ---------------------------------------------------------------------------
# normalize_food_quantity is importable from core only
# ---------------------------------------------------------------------------

class TestNormalizeFoodQuantityImportPath:
    def test_importable_from_core_quantity_formatter(self):
        from app.core.quantity_formatter import normalize_food_quantity
        assert callable(normalize_food_quantity)

    def test_not_re_exported_from_quantity_parser(self):
        """quantity_parser must not re-export normalize_food_quantity."""
        import app.nlu.matching.quantity_parser as qp
        assert not hasattr(qp, "normalize_food_quantity"), (
            "quantity_parser must not expose normalize_food_quantity — "
            "keep the food-quantity helper isolated in app.core."
        )

    def test_not_re_exported_from_cart_summary_builder(self):
        """cart_summary_builder must not expose normalize_food_quantity."""
        import app.cart.read_models.cart_summary_builder as csb
        assert not hasattr(csb, "normalize_food_quantity")
