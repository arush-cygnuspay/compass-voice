from types import SimpleNamespace

from app.core.response_builder import ResponseBuilder
from app.state_machine.models.conversation_context import ConversationContext


class FakeStore:
    def __init__(self, item):
        self._item = item

    def get_item(self, item_id):
        return self._item


class FakeMenuRepo:
    def __init__(self, item):
        self.store = FakeStore(item)


def make_item():
    return SimpleNamespace(
        item_id="burger_1",
        name="Chicken Burger",
        pricing=SimpleNamespace(variants=[]),
        side_groups=[],
        modifier_groups=[],
    )


def test_response_builder_includes_prefill_feedback_and_confirmation():
    builder = ResponseBuilder(FakeMenuRepo(make_item()))
    context = ConversationContext()

    text = builder.build(
        "ask_for_quantity",
        context,
        {
            "item_name": "Chicken Burger",
            "prefilled_summary": "with Coke",
            "prefill_feedback": "I couldn't find rice.",
        },
    )

    assert "I couldn't find rice." in text
    assert "Chicken Burger with Coke" in text
    assert "How many Chicken Burger would you like?" in text
    assert text.index("Chicken Burger with Coke") < text.index("I couldn't find rice.")
    assert text.count("I couldn't find rice.") == 1


def test_response_builder_side_size_prompt_includes_entity_feedback():
    builder = ResponseBuilder(FakeMenuRepo(make_item()))
    context = ConversationContext()

    text = builder.build(
        "ask_for_side_size",
        context,
        {
            "side_item_name": "Coke",
            "available_sizes": ["Small", "Medium"],
            "matched_names": ["Coke"],
            "unmatched_names": ["fried rice"],
        },
    )

    assert "Got Coke." in text
    assert "I couldn't find fried rice." in text
    assert "Small or Medium" in text
