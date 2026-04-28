import unittest

from app.state_machine.models.conversation_context import ConversationContext


class ConversationContextReprompTests(unittest.TestCase):
    def test_bump_reprompt_returns_incrementing_ints_starting_from_1(self):
        context = ConversationContext()
        self.assertEqual(context.bump_reprompt("delivery_house_number"), 1)
        self.assertEqual(context.bump_reprompt("delivery_house_number"), 2)
        self.assertEqual(context.bump_reprompt("delivery_house_number"), 3)

    def test_bump_reprompt_is_per_field(self):
        context = ConversationContext()
        context.bump_reprompt("delivery_house_number")
        context.bump_reprompt("delivery_house_number")
        self.assertEqual(context.bump_reprompt("delivery_street"), 1)
        self.assertEqual(context.reprompt_count("delivery_house_number"), 2)

    def test_reset_reprompt_removes_the_key(self):
        context = ConversationContext()
        context.bump_reprompt("delivery_house_number")
        context.reset_reprompt("delivery_house_number")
        self.assertNotIn("delivery_house_number", context.reprompt_attempts)
        self.assertEqual(context.reprompt_count("delivery_house_number"), 0)

    def test_reset_reprompt_unknown_field_is_safe(self):
        context = ConversationContext()
        # Should not raise.
        context.reset_reprompt("never_seen_field")

    def test_reprompt_count_returns_0_for_unknown_field(self):
        context = ConversationContext()
        self.assertEqual(context.reprompt_count("never_seen_field"), 0)

    def test_reprompt_count_returns_integer_for_seen_field(self):
        context = ConversationContext()
        context.bump_reprompt("delivery_street")
        context.bump_reprompt("delivery_street")
        self.assertEqual(context.reprompt_count("delivery_street"), 2)
        self.assertIsInstance(context.reprompt_count("delivery_street"), int)

    def test_reset_task_clears_reprompt_attempts(self):
        context = ConversationContext()
        context.bump_reprompt("delivery_house_number")
        context.bump_reprompt("delivery_street")
        context.reset_task()
        self.assertEqual(len(context.reprompt_attempts), 0)

    def test_reprompt_attempts_serialize_round_trip(self):
        context = ConversationContext()
        context.bump_reprompt("delivery_house_number")
        context.bump_reprompt("delivery_house_number")
        context.bump_reprompt("delivery_street")

        restored = ConversationContext.from_dict(context.to_dict())

        self.assertEqual(restored.reprompt_count("delivery_house_number"), 2)
        self.assertEqual(restored.reprompt_count("delivery_street"), 1)


if __name__ == "__main__":
    unittest.main()
