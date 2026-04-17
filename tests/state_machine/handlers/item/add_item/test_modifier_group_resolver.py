import unittest

from app.state_machine.handlers.item.add_item.modifier_group_resolver import (
    ModifierGroupResolver,
)
from app.state_machine.models.pending_item_models import (
    PendingModifierChoice,
    PendingModifierGroup,
)


class ModifierGroupResolverTests(unittest.TestCase):
    def test_parses_on_the_side_instruction(self):
        ranch = PendingModifierChoice(
            modifier_id="ranch",
            name="Ranch",
            group_id="mods",
            normalized_name="ranch",
        )
        group = PendingModifierGroup(
            group_id="mods",
            name="Extras",
            is_required=False,
            min_selector=0,
            max_selector=3,
            choices=[ranch],
            choices_by_modifier_id={"ranch": ranch},
            choices_by_normalized_name={"ranch": [ranch]},
            choice_names=("Ranch",),
            normalized_choice_names=("ranch",),
            top_choice_names=("Ranch",),
        )

        result = ModifierGroupResolver().resolve(
            group=group,
            normalized_user_text="ranch on the side",
            normalized_slot_values=[],
            already_selected_ids=[],
        )

        self.assertEqual(len(result.selections), 1)
        self.assertEqual(result.selections[0].modifier_id, "ranch")
        self.assertEqual(result.selections[0].instruction, "on_side")


if __name__ == "__main__":
    unittest.main()
