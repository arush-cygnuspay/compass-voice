from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.item.add_item.waiting_for_side_size_handler import (
    WaitingForSideSizeHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_size_handler import (
    WaitingForSizeHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import (
    PendingAddItem,
    PendingSideChoice,
    PendingVariantChoice,
)


def test_waiting_for_size_accepts_semantic_affirm_for_pending_guess() -> None:
    small = PendingVariantChoice(variant_id="small", name="Small", normalized_name="small")
    large = PendingVariantChoice(variant_id="large", name="Large", normalized_name="large")

    context = ConversationContext()
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        item_variants=[small, large],
        item_variants_by_id={"small": small, "large": large},
        item_variants_by_normalized_name={"small": small, "large": large},
        item_variant_names=("Small", "Large"),
        top_item_variant_names=("Small", "Large"),
    )
    context.awaiting_confirmation_for = {
        "type": "size_choice_guess",
        "variant_id": "large",
        "choice_name": "Large",
        "confidence": 0.7,
    }

    result = WaitingForSizeHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="sounds good",
        session=None,
    )

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_added_successfully"
    assert context.selected_variant_id == "large"


def test_waiting_for_side_size_accepts_semantic_deny_for_pending_guess() -> None:
    small = PendingVariantChoice(variant_id="small", name="Small", normalized_name="small")
    large = PendingVariantChoice(variant_id="large", name="Large", normalized_name="large")
    coke = PendingSideChoice(
        item_id="coke",
        name="Coke",
        pricing_mode="variant",
        normalized_name="coke",
        variants=[small, large],
        variants_by_id={"small": small, "large": large},
        variants_by_normalized_name={"small": small, "large": large},
        variant_names=("Small", "Large"),
        top_variant_names=("Small", "Large"),
    )

    context = ConversationContext()
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        side_choice_by_item_id={"coke": coke},
    )
    context.pending_side_item_id = "coke"
    context.awaiting_confirmation_for = {
        "type": "side_size_choice_guess",
        "side_item_id": "coke",
        "variant_id": "large",
        "choice_name": "Large",
        "confidence": 0.7,
    }

    result = WaitingForSideSizeHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="nope",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_SIDE_SIZE
    assert result.response_key == "repeat_side_size_options"
    assert context.selected_side_variants == {}


def test_waiting_for_size_does_not_skip_required_size_on_skip_it() -> None:
    small = PendingVariantChoice(variant_id="small", name="Small", normalized_name="small")
    large = PendingVariantChoice(variant_id="large", name="Large", normalized_name="large")

    context = ConversationContext()
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        item_variants=[small, large],
        item_variants_by_id={"small": small, "large": large},
        item_variants_by_normalized_name={"small": small, "large": large},
        item_variant_names=("Small", "Large"),
        top_item_variant_names=("Small", "Large"),
    )

    result = WaitingForSizeHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="skip it",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_SIZE
    assert result.response_key == "required_size_cannot_skip"


def test_waiting_for_size_lists_available_sizes_for_options_request() -> None:
    small = PendingVariantChoice(variant_id="small", name="Small", normalized_name="small")
    large = PendingVariantChoice(variant_id="large", name="Large", normalized_name="large")

    context = ConversationContext()
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        item_variants=[small, large],
        item_variants_by_id={"small": small, "large": large},
        item_variants_by_normalized_name={"small": small, "large": large},
        item_variant_names=("Small", "Large"),
        top_item_variant_names=("Small", "Large"),
    )

    result = WaitingForSizeHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="what sizes do you have?",
        session=None,
    )

    assert result.next_state == ConversationState.WAITING_FOR_SIZE
    assert result.response_key == "repeat_size_options"
    assert result.response_payload["available_sizes"] == ["Small", "Large"]


def test_waiting_for_size_accepts_direct_size_choice() -> None:
    small = PendingVariantChoice(variant_id="small", name="Small", normalized_name="small")
    large = PendingVariantChoice(variant_id="large", name="Large", normalized_name="large")

    context = ConversationContext()
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        item_variants=[small, large],
        item_variants_by_id={"small": small, "large": large},
        item_variants_by_normalized_name={"small": small, "large": large},
        item_variant_names=("Small", "Large"),
        top_item_variant_names=("Small", "Large"),
    )

    result = WaitingForSizeHandler().handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="small",
        session=None,
    )

    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_added_successfully"
    assert context.selected_variant_id == "small"
