from types import SimpleNamespace

from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
    WaitingForModifierHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
    WaitingForSideHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.pending_item_models import ModifierSelection


def test_modifier_choice_payload_excludes_already_selected_options():
    handler = WaitingForModifierHandler()
    group = SimpleNamespace(
        name="Burger Modification",
        choices=[
            SimpleNamespace(modifier_id="m1", name="Fresh Mushroom"),
            SimpleNamespace(modifier_id="m2", name="Red Onions"),
            SimpleNamespace(modifier_id="m3", name="Bacon"),
            SimpleNamespace(modifier_id="m4", name="Banana Pepper"),
        ],
        top_choice_names=("Fresh Mushroom", "Red Onions", "Bacon", "Banana Pepper"),
        choice_names=("Fresh Mushroom", "Red Onions", "Bacon", "Banana Pepper"),
        min_selector=0,
        max_selector=4,
        is_required=False,
    )
    selections = [
        ModifierSelection(modifier_id="m1", name="Fresh Mushroom"),
        ModifierSelection(modifier_id="m2", name="Red Onions"),
    ]

    payload = handler._choice_payload(group, selections)

    assert payload["top_choices"] == ["Bacon", "Banana Pepper"]
    assert payload["all_choices"] == ["Bacon", "Banana Pepper"]


def test_side_choice_payload_excludes_already_selected_options():
    handler = WaitingForSideHandler()
    context = ConversationContext()
    context.selected_side_groups = {"drink": ["coke"]}
    group = SimpleNamespace(
        group_id="drink",
        name="Choose your drink",
        choices=[
            SimpleNamespace(item_id="coke", name="Coke"),
            SimpleNamespace(item_id="sprite", name="Sprite"),
            SimpleNamespace(item_id="fanta", name="Fanta"),
        ],
        choices_by_item_id={
            "coke": SimpleNamespace(name="Coke"),
            "sprite": SimpleNamespace(name="Sprite"),
            "fanta": SimpleNamespace(name="Fanta"),
        },
        top_choice_names=("Coke", "Sprite", "Fanta"),
        choice_names=("Coke", "Sprite", "Fanta"),
        min_selector=0,
        max_selector=3,
        is_required=False,
    )

    payload = handler._choice_payload(context, group)

    assert payload["top_choices"] == ["Sprite", "Fanta"]
    assert payload["all_choices"] == ["Sprite", "Fanta"]
