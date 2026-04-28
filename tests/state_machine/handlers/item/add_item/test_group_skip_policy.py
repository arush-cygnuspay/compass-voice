"""Pure-function tests for evaluate_group_skip."""
import unittest

from app.state_machine.handlers.item.add_item.group_skip_policy import (
    GroupSkipDecision,
    evaluate_group_skip,
)


class EvaluateGroupSkipTests(unittest.TestCase):
    def test_zero_min_zero_selected_is_skip_optional(self):
        result = evaluate_group_skip(0, 0)
        self.assertEqual(result.decision, GroupSkipDecision.SKIP_OPTIONAL)
        self.assertEqual(result.remaining_to_min, 0)
        self.assertEqual(result.selected_count, 0)
        self.assertEqual(result.min_required, 0)

    def test_zero_min_with_selections_is_still_skip_optional(self):
        result = evaluate_group_skip(0, 3)
        self.assertEqual(result.decision, GroupSkipDecision.SKIP_OPTIONAL)
        self.assertEqual(result.remaining_to_min, 0)
        self.assertEqual(result.selected_count, 3)
        self.assertEqual(result.min_required, 0)

    def test_two_min_zero_selected_blocks_with_remaining_two(self):
        result = evaluate_group_skip(2, 0)
        self.assertEqual(result.decision, GroupSkipDecision.BLOCK_UNDER_MIN)
        self.assertEqual(result.remaining_to_min, 2)
        self.assertEqual(result.selected_count, 0)
        self.assertEqual(result.min_required, 2)

    def test_two_min_one_selected_blocks_with_remaining_one(self):
        result = evaluate_group_skip(2, 1)
        self.assertEqual(result.decision, GroupSkipDecision.BLOCK_UNDER_MIN)
        self.assertEqual(result.remaining_to_min, 1)
        self.assertEqual(result.selected_count, 1)
        self.assertEqual(result.min_required, 2)

    def test_two_min_two_selected_advances(self):
        result = evaluate_group_skip(2, 2)
        self.assertEqual(result.decision, GroupSkipDecision.ADVANCE_MIN_MET)
        self.assertEqual(result.remaining_to_min, 0)
        self.assertEqual(result.selected_count, 2)
        self.assertEqual(result.min_required, 2)

    def test_two_min_five_selected_advances(self):
        result = evaluate_group_skip(2, 5)
        self.assertEqual(result.decision, GroupSkipDecision.ADVANCE_MIN_MET)
        self.assertEqual(result.remaining_to_min, 0)
        self.assertEqual(result.selected_count, 5)
        self.assertEqual(result.min_required, 2)

    def test_negative_inputs_are_clamped_to_zero(self):
        result = evaluate_group_skip(-1, -3)
        self.assertEqual(result.decision, GroupSkipDecision.SKIP_OPTIONAL)
        self.assertEqual(result.remaining_to_min, 0)
        self.assertEqual(result.selected_count, 0)
        self.assertEqual(result.min_required, 0)

    def test_none_inputs_default_to_zero(self):
        result = evaluate_group_skip(None, None)  # type: ignore[arg-type]
        self.assertEqual(result.decision, GroupSkipDecision.SKIP_OPTIONAL)
        self.assertEqual(result.remaining_to_min, 0)
        self.assertEqual(result.selected_count, 0)
        self.assertEqual(result.min_required, 0)


if __name__ == "__main__":
    unittest.main()
