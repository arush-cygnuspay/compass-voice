from app.menu.models import MenuItem, Pricing
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


class _StubMenuRepo:
    def get_item(self, item_id: str) -> MenuItem:
        return MenuItem(
            item_id=item_id,
            name="Zinger Burger" if item_id == "burger_1" else "Chicken Burger",
            normalized_name=normalize_text("Zinger Burger" if item_id == "burger_1" else "Chicken Burger"),
            aliases=(),
            normalized_aliases=(),
            voice_labels=(),
            pricing=Pricing(mode="fixed", price_cents=1000),
            side_groups=[],
            modifier_groups=[],
            available=True,
        )

    def resolve_item_within_candidates_normalized(self, normalized_text: str, candidate_item_ids: list[str]):
        return None

    def resolve_menu_query(self, normalized_text: str, limit: int = 5):
        raise AssertionError("fresh menu resolution should not run for these control-intent tests")

    def resolve_menu_query_from_slots(self, **kwargs):
        raise AssertionError("slot-based menu resolution should not run for these control-intent tests")


def _make_previous_confirmation() -> dict:
    return {
        "type": "item",
        "reason": "multiple_matches",
        "query": "burger",
        "candidate_item_ids": ["burger_1", "burger_2"],
        "candidate_item_names": ["Zinger Burger", "Chicken Burger"],
    }


def test_not_that_rejects_current_item_candidate() -> None:
    context = ConversationContext()
    previous_confirmation = _make_previous_confirmation()
    context.awaiting_confirmation_for = {
        "type": "item",
        "reason": "candidate_selected",
        "value_id": "burger_1",
        "value_name": "Zinger Burger",
        "previous_confirmation": previous_confirmation,
    }

    class _Session:
        conversation_state = ConversationState.CONFIRMING_ITEM

    result = ConfirmingHandler(_StubMenuRepo()).handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="not that",
        session=_Session(),
    )

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert result.response_key == "confirm_item_ambiguous"
    assert context.awaiting_confirmation_for["reason"] == "multiple_matches"


def test_what_do_you_have_repeats_relevant_item_options() -> None:
    context = ConversationContext()
    context.awaiting_confirmation_for = _make_previous_confirmation()

    class _Session:
        conversation_state = ConversationState.CONFIRMING_ITEM

    result = ConfirmingHandler(_StubMenuRepo()).handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="what do you have?",
        session=_Session(),
    )

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert result.response_key == "confirm_item_ambiguous"
    assert result.response_payload["candidate_item_names"] == ["Zinger Burger", "Chicken Burger"]


def test_repeat_that_repeats_item_disambiguation_prompt() -> None:
    context = ConversationContext()
    context.awaiting_confirmation_for = _make_previous_confirmation()

    class _Session:
        conversation_state = ConversationState.CONFIRMING_ITEM

    result = ConfirmingHandler(_StubMenuRepo()).handle(
        intent=Intent.UNKNOWN,
        context=context,
        user_text="repeat that",
        session=_Session(),
    )

    assert result.next_state == ConversationState.CONFIRMING_ITEM
    assert result.response_key == "confirm_item_ambiguous"
