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
                group_id="mod_1",
                name="Add-on",
                choices=[
                    SimpleNamespace(name="Cheese"),
                    SimpleNamespace(name="Jalapeno"),
                    SimpleNamespace(name="Sauce"),
                ],
                is_required=True,
                min_selector=3,
                max_selector=5,
            )
        ],
    )


def test_response_builder_ask_for_size():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()
    context.current_item_id = "burger_1"

    text = builder.build("ask_for_size", context, {"item_name": "Zinger Burger"})

    assert "size" in text.lower()
    assert "Small" not in text
    assert "Large" not in text


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
    assert "yes or no" not in text.lower()


def test_response_builder_confirm_item_uses_natural_confirmation_prompt():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()
    context.current_item_name = "Zinger Burger"

    text = builder.build("confirm_item", context, {})

    assert "zinger burger" in text.lower()
    assert "is that right" in text.lower()
    assert "yes or no" not in text.lower()


def test_response_builder_ask_for_modifier_includes_noun_and_examples():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()
    context.current_item_id = "burger_1"
    context.current_modifier_group_index = 0

    text = builder.build(
        "ask_for_modifier",
        context,
        {"min_selector": 3, "max_selector": 5},
    )

    lower = text.lower()
    assert "add-on" in lower
    assert "would you like" in lower
    assert "cheese" in lower
    assert "jalapeno" in lower
    assert "sauce" in lower


def test_response_builder_repeat_modifier_options_reports_remaining_count():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()
    context.current_item_id = "burger_1"
    context.current_modifier_group_index = 0
    context.selected_modifier_groups = {
        "mod_1": [SimpleNamespace(name="Cheese", action="add", instruction=None)]
    }

    text = builder.build(
        "repeat_modifier_options",
        context,
        {
            "repeat_reason": "need_more",
            "selected_names": ["Cheese"],
            "selected_count": 1,
            "min_selector": 3,
            "max_selector": 5,
            "remaining_to_min": 2,
            "remaining_to_max": 4,
            "top_choices": ["Cheese", "Jalapeno", "Sauce"],
        },
    )

    assert "pick 2 more" in text.lower()
    assert "cheese" in text.lower()


def test_response_builder_repeat_modifier_options_uses_remaining_choices_only():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()
    context.current_item_id = "burger_1"
    context.current_modifier_group_index = 0

    text = builder.build(
        "repeat_modifier_options",
        context,
        {
            "repeat_reason": "optional_more",
            "selected_names": ["Cheese"],
            "selected_count": 1,
            "min_selector": 0,
            "max_selector": 3,
            "remaining_to_min": 0,
            "remaining_to_max": 2,
            "top_choices": ["Jalapeno", "Sauce"],
            "all_choices": ["Jalapeno", "Sauce"],
        },
    )

    assert "up to 2 more" in text.lower()
    assert "jalapeno or sauce" in text.lower()


def test_response_builder_invalid_after_min_is_met_offers_done_instead_of_repeating_failure():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()
    context.current_item_id = "burger_1"
    context.current_modifier_group_index = 0

    text = builder.build(
        "repeat_modifier_options",
        context,
        {
            "repeat_reason": "invalid",
            "selected_names": ["Cheese", "Jalapeno", "Sauce"],
            "selected_count": 3,
            "min_selector": 3,
            "max_selector": 5,
            "remaining_to_min": 0,
            "remaining_to_max": 2,
            "top_choices": ["Bacon", "Banana Pepper"],
            "all_choices": ["Bacon", "Banana Pepper"],
        },
    )

    assert "didn't catch a valid option" not in text.lower()
    assert "or say done" in text.lower()
    assert "bacon or banana pepper" in text.lower()


def test_response_builder_unknown_key_returns_fallback():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()

    text = builder.build("does_not_exist", context, {})

    assert text == "Sorry, I didn’t understand that."


def test_confirm_order_summary_does_not_repeat_checkout_prompt():
    item = make_item()
    builder = ResponseBuilder(FakeMenuRepo(item))
    context = ConversationContext()
    context.order_type = "pickup"

    text = builder.build(
        "confirm_order_summary",
        context,
        {
            "items": [{"quantity": 1, "name": "Zinger Burger", "sides": [], "modifiers": []}],
            "total": "$5.00",
        },
    )

    # "checkout" no longer appears — the confirmation question is now "Should I place the order?"
    assert "checkout" not in text.lower()
    assert "should i place the order?" in text.lower()
    assert "please say yes to confirm" not in text.lower()
