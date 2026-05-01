# tests/nlu/test_quantity_resolver.py
"""
Unit tests for QuantityResolver.

Spec:
  1. Explicit extracted quantity → use it.
  2. Leading vague expression ("some", "a few", "several") → ask clarification.
  3. No quantity info → default to 1.
  4. Leading vague must be at the START of the item-stripped text to avoid
     false positives ("burger with some modifications" → default 1).
"""
from __future__ import annotations

import unittest

from app.nlu.quantity_resolver import QuantityResolution, QuantityResolver


class QuantityResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = QuantityResolver()

    # ------------------------------------------------------------------
    # Explicit quantity
    # ------------------------------------------------------------------

    def test_explicit_quantity_wins(self):
        result = self.resolver.resolve(extracted=2, user_text="two burgers")
        self.assertEqual(result.quantity, 2)
        self.assertEqual(result.source, "explicit")
        self.assertFalse(result.needs_clarification)

    def test_explicit_quantity_1_wins(self):
        result = self.resolver.resolve(extracted=1, user_text="a burger")
        self.assertEqual(result.quantity, 1)
        self.assertEqual(result.source, "explicit")

    def test_explicit_quantity_overrides_vague_text(self):
        # If slot already extracted 3, "some" in text should be ignored.
        result = self.resolver.resolve(extracted=3, user_text="some burgers")
        self.assertEqual(result.quantity, 3)
        self.assertEqual(result.source, "explicit")
        self.assertFalse(result.needs_clarification)

    # ------------------------------------------------------------------
    # Vague quantity → clarification
    # ------------------------------------------------------------------

    def test_leading_some_asks_clarification(self):
        result = self.resolver.resolve(extracted=None, user_text="some burgers")
        self.assertIsNone(result.quantity)
        self.assertEqual(result.source, "ambiguous")
        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.clarification_reason, "vague_quantity")

    def test_leading_a_few_asks_clarification(self):
        result = self.resolver.resolve(extracted=None, user_text="a few tacos")
        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.source, "ambiguous")

    def test_leading_several_asks_clarification(self):
        result = self.resolver.resolve(extracted=None, user_text="several wings")
        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.source, "ambiguous")

    def test_vague_case_insensitive(self):
        result = self.resolver.resolve(extracted=None, user_text="Some Burgers")
        self.assertTrue(result.needs_clarification)

    # ------------------------------------------------------------------
    # Non-leading vague → default to 1 (false-positive guard)
    # ------------------------------------------------------------------

    def test_non_leading_some_defaults_to_1(self):
        # "some" inside a modifier phrase must not trigger vague detection
        result = self.resolver.resolve(
            extracted=None, user_text="with some modifications"
        )
        self.assertEqual(result.quantity, 1)
        self.assertEqual(result.source, "implicit_default")
        self.assertFalse(result.needs_clarification)

    def test_non_leading_a_few_defaults_to_1(self):
        result = self.resolver.resolve(
            extracted=None, user_text="burger with a few toppings"
        )
        self.assertEqual(result.quantity, 1)
        self.assertFalse(result.needs_clarification)

    # ------------------------------------------------------------------
    # No quantity info → default to 1
    # ------------------------------------------------------------------

    def test_no_quantity_defaults_to_1(self):
        result = self.resolver.resolve(extracted=None, user_text="hamburger")
        self.assertEqual(result.quantity, 1)
        self.assertEqual(result.source, "implicit_default")
        self.assertFalse(result.needs_clarification)

    def test_empty_text_defaults_to_1(self):
        result = self.resolver.resolve(extracted=None, user_text="")
        self.assertEqual(result.quantity, 1)
        self.assertFalse(result.needs_clarification)

    def test_none_text_defaults_to_1(self):
        result = self.resolver.resolve(extracted=None, user_text=None)  # type: ignore[arg-type]
        self.assertEqual(result.quantity, 1)
        self.assertFalse(result.needs_clarification)

    # ------------------------------------------------------------------
    # Dataclass is immutable
    # ------------------------------------------------------------------

    def test_resolution_is_frozen(self):
        result = self.resolver.resolve(extracted=None, user_text="hamburger")
        with self.assertRaises((AttributeError, TypeError)):
            result.quantity = 99  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
