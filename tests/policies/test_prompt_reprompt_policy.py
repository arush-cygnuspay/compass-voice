# tests/policies/test_prompt_reprompt_policy.py
"""Unit tests for PromptRepromptPolicy tier logic."""
import unittest

from app.policies.prompt_reprompt_policy import PromptRepromptPolicy, RepromptAction


class PromptRepromptPolicyTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Tier boundaries
    # ------------------------------------------------------------------

    def test_miss_count_zero_returns_full_options(self):
        self.assertEqual(
            PromptRepromptPolicy.next_action("size", 0),
            RepromptAction.FULL_OPTIONS,
        )

    def test_miss_count_one_returns_concise(self):
        self.assertEqual(
            PromptRepromptPolicy.next_action("size", 1),
            RepromptAction.CONCISE,
        )

    def test_miss_count_two_returns_list_options_hint(self):
        self.assertEqual(
            PromptRepromptPolicy.next_action("size", 2),
            RepromptAction.LIST_OPTIONS_HINT,
        )

    def test_miss_count_three_returns_escalate_or_skip(self):
        self.assertEqual(
            PromptRepromptPolicy.next_action("size", 3),
            RepromptAction.ESCALATE_OR_SKIP,
        )

    def test_high_miss_count_returns_escalate_or_skip(self):
        self.assertEqual(
            PromptRepromptPolicy.next_action("size", 99),
            RepromptAction.ESCALATE_OR_SKIP,
        )

    def test_negative_miss_count_returns_full_options(self):
        self.assertEqual(
            PromptRepromptPolicy.next_action("size", -1),
            RepromptAction.FULL_OPTIONS,
        )

    # ------------------------------------------------------------------
    # Field-agnostic — policy uses the same tiers regardless of field name
    # ------------------------------------------------------------------

    def test_modifier_field_same_tier_logic(self):
        self.assertEqual(
            PromptRepromptPolicy.next_action("modifier", 1),
            RepromptAction.CONCISE,
        )
        self.assertEqual(
            PromptRepromptPolicy.next_action("modifier", 2),
            RepromptAction.LIST_OPTIONS_HINT,
        )
        self.assertEqual(
            PromptRepromptPolicy.next_action("modifier", 3),
            RepromptAction.ESCALATE_OR_SKIP,
        )

    def test_unknown_field_same_tier_logic(self):
        self.assertEqual(
            PromptRepromptPolicy.next_action("unknown_field", 1),
            RepromptAction.CONCISE,
        )

    # ------------------------------------------------------------------
    # RepromptAction values are strings (for JSON serialisation safety)
    # ------------------------------------------------------------------

    def test_reprompt_action_values_are_strings(self):
        for action in RepromptAction:
            self.assertIsInstance(action.value, str)


if __name__ == "__main__":
    unittest.main()
