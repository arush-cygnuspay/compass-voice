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
        name="Zinger Burger",
        pricing=SimpleNamespace(
            variants=[
                SimpleNamespace(label="Small", price_cents=500),
                SimpleNamespace(label="Large", price_cents=700),
            ]
        ),
        side_groups=[
            SimpleNamespace(
                name="Choose your side",
                choices=[
                    SimpleNamespace(name="Fries"),
                    SimpleNamespace(name="Salad"),
                    SimpleNamespace(name="Coleslaw"),
                ],
                is_required=True,
                min_selector=1,
            )
        ],
        modifier_groups=[
            SimpleNamespace(
                name="Add-on",
                choices=[
                    SimpleNamespace(name="Cheese"),
                    SimpleNamespace(name="Jalapeno"),
                    SimpleNamespace(name="Sauce"),
                ],
                is_required=False,
                min_selector=0,
            )
        ],
    )


def test_response_builder_ask_for_size():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()
    context.current_item_id = "burger_1"

    text = builder.build("ask_for_size", context, {"item_name": "Zinger Burger"})

    assert "Which size" in text
    assert "Small" in text
    assert "Large" in text


def test_response_builder_item_added_successfully():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()

    text = builder.build(
        "item_added_successfully",
        context,
        {"item_name": "Zinger Burger", "quantity": 2},
    )

    assert "added" in text.lower()
    assert "2" in text


def test_response_builder_confirm_cancel_current_item():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()
    context.current_item_name = "Zinger Burger"

    text = builder.build("confirm_cancel_current_item", context, {})

    assert "cancel" in text.lower()
    assert "yes or no" in text.lower()


def test_response_builder_unknown_key_returns_fallback():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()

    text = builder.build("does_not_exist", context, {})

    assert text == "Sorry, I didn’t understand that."