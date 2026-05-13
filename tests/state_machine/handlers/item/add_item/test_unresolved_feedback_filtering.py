# tests/state_machine/handlers/item/add_item/test_unresolved_feedback_filtering.py
"""Unit tests for unresolved-entity feedback filtering.

Covers:
- Quantity words ("one", "two", "3") never produce "I couldn't find X."
- Order filler tokens ("add", "please") never produce feedback
- Connector tokens ("with", "and", "no") never produce feedback
- Pure-fuzzy false-positive bindings (e.g. "rice" → "sprite") are rejected
- Genuine unresolved menu entities ("rice", "avocado") appear in feedback
- "no sauce" / sauce-in-_WEAK_TOKENS: the word "sauce" is not silently dropped
  from feedback when it is the meaningful content
- Consumed tokens (item name, resolved quantity) suppress surface forms
- ResponseBuilder terminal-success guard blocks feedback for clean adds
- partial_success=True flag passes feedback through on genuine partial adds
"""
from __future__ import annotations

import pytest

from app.menu.models import MenuItem, ModifierChoice, ModifierGroup, Pricing, SideChoice, SideGroup
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.nlu.intent_resolution.intent import Intent
from app.nlu.lexicons.non_entity_tokens import (
    NON_ENTITY_TOKENS,
    filter_unmatched_for_speech,
    is_non_entity_phrase,
)
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.multi_group_prefill import (
    CONFIRM_THRESHOLD,
    MultiGroupPrefillEngine,
)
from app.state_machine.handlers.item.add_item.option_matching import score_scoped_choice
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeMenuRepo:
    def __init__(self, item: MenuItem) -> None:
        self._result = MenuQueryResult(type=MenuQueryType.ITEM, item=item)

    def resolve_menu_query_from_slots(self, **kwargs):
        return self._result

    def resolve_menu_query_from_slots_normalized(self, **kwargs):
        return self._result

    def resolve_menu_query(self, text, limit=5):
        return self._result

    def resolve_menu_query_normalized(self, text, limit=5):
        return self._result

    class store:
        @staticmethod
        def find_entity(*a, **kw):
            return []

        @staticmethod
        def find_item_exact(*a, **kw):
            return None

        @staticmethod
        def find_item_ids_by_alias(*a, **kw):
            return []

        @staticmethod
        def find_item_ids_by_voice_label(*a, **kw):
            return []

        @staticmethod
        def find_discoverable_item_mentions(*a, **kw):
            return []


def _item_with_side(side_choices: list[tuple[str, str]], max_selector: int = 1) -> MenuItem:
    choices = [
        SideChoice(
            item_id=item_id,
            name=name,
            normalized_name=normalize_text(name),
            pricing=Pricing(mode="fixed", price_cents=0),
        )
        for item_id, name in side_choices
    ]
    return MenuItem(
        item_id="test_item",
        name="Test Burger",
        normalized_name=normalize_text("Test Burger"),
        aliases=("test burger",),
        normalized_aliases=(normalize_text("test burger"),),
        voice_labels=("test burger",),
        pricing=Pricing(mode="fixed", price_cents=600),
        side_groups=[
            SideGroup(
                group_id="drinks",
                name="Choose your drink",
                normalized_name=normalize_text("Choose your drink"),
                is_required=False,
                min_selector=0,
                max_selector=max_selector,
                choices=choices,
            )
        ],
        modifier_groups=[],
        available=True,
    )


def _item_with_modifier(mod_choices: list[tuple[str, str]], max_selector: int = 1) -> MenuItem:
    choices = [
        ModifierChoice(
            modifier_id=mod_id,
            name=name,
            normalized_name=normalize_text(name),
            price_cents=0,
        )
        for mod_id, name in mod_choices
    ]
    return MenuItem(
        item_id="test_item",
        name="Test Burger",
        normalized_name=normalize_text("Test Burger"),
        aliases=("test burger",),
        normalized_aliases=(normalize_text("test burger"),),
        voice_labels=("test burger",),
        pricing=Pricing(mode="fixed", price_cents=600),
        side_groups=[],
        modifier_groups=[
            ModifierGroup(
                group_id="mods",
                name="Add-ons",
                normalized_name=normalize_text("Add-ons"),
                is_required=False,
                min_selector=0,
                max_selector=max_selector,
                choices=choices,
            )
        ],
        available=True,
    )


# ---------------------------------------------------------------------------
# NON_ENTITY_TOKENS contract
# ---------------------------------------------------------------------------


class TestIsNonEntityPhrase:
    @pytest.mark.parametrize("phrase", [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "a", "an",
        "1", "2", "3", "10", "99",
        "dozen",
        "add", "please", "get", "i", "want", "me",
        "with", "and", "no", "extra",
    ])
    def test_quantity_and_filler_words_are_non_entity(self, phrase: str) -> None:
        assert is_non_entity_phrase(phrase), f"{phrase!r} should be non-entity"

    @pytest.mark.parametrize("phrase", [
        "rice", "avocado", "sprite", "coke",
        "mayo", "sausage", "jelly",
        "american cheese", "red onions",
        "unicorn sauce",
    ])
    def test_food_words_are_not_non_entity(self, phrase: str) -> None:
        assert not is_non_entity_phrase(phrase), f"{phrase!r} should not be non-entity"

    def test_empty_phrase_is_non_entity(self) -> None:
        assert is_non_entity_phrase("")

    def test_mixed_phrase_with_food_word_is_not_non_entity(self) -> None:
        assert not is_non_entity_phrase("one rice")


class TestFilterUnmatchedForSpeech:
    def test_quantity_words_are_dropped(self) -> None:
        assert filter_unmatched_for_speech(["one", "two", "3"]) == []

    def test_filler_words_are_dropped(self) -> None:
        assert filter_unmatched_for_speech(["add", "please", "i", "want"]) == []

    def test_food_words_are_kept(self) -> None:
        result = filter_unmatched_for_speech(["rice", "avocado"])
        assert result == ["rice", "avocado"]

    def test_consumed_tokens_suppress_phrase(self) -> None:
        result = filter_unmatched_for_speech(["spicy tuna"], consumed_tokens=["spicy", "tuna"])
        assert result == []

    def test_unconsumed_food_word_passes_through(self) -> None:
        result = filter_unmatched_for_speech(["rice"], consumed_tokens=["spicy", "tuna"])
        assert result == ["rice"]

    def test_deduplication(self) -> None:
        result = filter_unmatched_for_speech(["rice", "rice", "Rice"])
        assert result == ["rice"]


# ---------------------------------------------------------------------------
# score_scoped_choice: pure-fuzzy false-positive guard
# ---------------------------------------------------------------------------


class TestScoreScopedChoiceFuzzyGuard:
    def test_rice_does_not_score_against_sprite(self) -> None:
        """'rice' and 'sprite' share character sequences but are unrelated foods."""
        score = score_scoped_choice("rice", "sprite", reject_candidate_superset=True)
        assert score < CONFIRM_THRESHOLD, (
            f"'rice' → 'sprite' scored {score:.3f}, must be < {CONFIRM_THRESHOLD}"
        )

    def test_rice_does_not_score_against_coke(self) -> None:
        score = score_scoped_choice("rice", "coke", reject_candidate_superset=True)
        assert score < CONFIRM_THRESHOLD

    def test_onions_scores_against_grilled_onions(self) -> None:
        """'onions' has token overlap with 'grilled onions', legitimate partial match."""
        score = score_scoped_choice("onions", "grilled onions", reject_candidate_superset=True)
        assert score >= CONFIRM_THRESHOLD

    def test_cheese_scores_against_grilled_cheese(self) -> None:
        score = score_scoped_choice("cheese", "grilled cheese", reject_candidate_superset=True)
        assert score >= CONFIRM_THRESHOLD

    def test_exact_matches_score_1(self) -> None:
        for name in ["coke", "sprite", "lettuce", "tomato"]:
            assert score_scoped_choice(name, name) == 1.0


# ---------------------------------------------------------------------------
# Full pipeline: handler-level feedback tests
# ---------------------------------------------------------------------------


class TestHandlerFeedbackFiltering:
    """Integration tests that run AddItemHandler and check prefill_feedback."""

    def _handle(
        self,
        item: MenuItem,
        slots: tuple[SlotValue, ...],
        user_text: str,
    ):
        repo = _FakeMenuRepo(item)
        handler = AddItemHandler(repo)
        context = ConversationContext()
        context.last_slots = slots
        return handler.handle(
            intent=Intent.ADD_ITEM,
            context=context,
            user_text=user_text,
            session=None,
        )

    def test_quantity_word_one_does_not_appear_in_feedback(self) -> None:
        """'Add one spicy tuna roll' → no 'one' in feedback."""
        item = _item_with_side([("tuna", "Spicy Tuna Roll")])
        slots = (
            SlotValue(name="QUANTITY", value="1"),
            SlotValue(name="ITEM", value="Spicy Tuna Roll"),
        )
        result = self._handle(item, slots, "add one spicy tuna roll")
        feedback = result.response_payload.get("prefill_feedback", "")
        assert "one" not in feedback.lower()
        assert "I couldn't find" not in feedback

    def test_rice_not_found_appears_in_feedback_despite_fuzzy_match(self) -> None:
        """'rice' should not bind to 'Sprite' and must appear as unresolved."""
        item = _item_with_side([("coke", "Coke"), ("sprite", "Sprite")], max_selector=1)
        slots = (
            SlotValue(name="ITEM", value="Test Burger"),
            SlotValue(name="ITEM", value="Coke"),
            SlotValue(name="ITEM", value="Sprite"),
            SlotValue(name="ITEM", value="Rice"),
        )
        result = self._handle(
            item, slots, "test burger with coke sprite and rice"
        )
        assert result.next_state == ConversationState.WAITING_FOR_SIDE
        feedback = result.response_payload.get("prefill_feedback", "")
        assert "I couldn't find rice." in feedback, f"feedback was: {feedback!r}"

    def test_no_sauce_appears_in_feedback(self) -> None:
        """'no sauce' is a genuine unresolved request; 'sauce' is in _WEAK_TOKENS
        but must still survive the canonical-key computation."""
        item = _item_with_modifier([("cheese", "Cheese"), ("bacon", "Bacon")], max_selector=1)
        slots = (
            SlotValue(name="ITEM", value="Test Burger"),
            SlotValue(name="MODIFIER", value="Cheese"),
            SlotValue(name="MODIFIER", value="Bacon"),
            SlotValue(name="ITEM", value="No Sauce"),
        )
        result = self._handle(
            item, slots, "test burger with cheese bacon and no sauce"
        )
        feedback = result.response_payload.get("prefill_feedback", "")
        assert "no sauce" in feedback.lower(), f"feedback was: {feedback!r}"

    def test_avocado_not_found_appears_in_feedback(self) -> None:
        """Genuine unresolved food items appear in feedback."""
        item = _item_with_modifier([("cheese", "Cheese"), ("bacon", "Bacon")], max_selector=1)
        slots = (
            SlotValue(name="ITEM", value="Test Burger"),
            SlotValue(name="MODIFIER", value="Cheese"),
            SlotValue(name="MODIFIER", value="Bacon"),
            SlotValue(name="MODIFIER", value="Avocado"),
        )
        result = self._handle(
            item, slots, "test burger with cheese bacon and avocado"
        )
        feedback = result.response_payload.get("prefill_feedback", "")
        assert "I couldn't find avocado." in feedback, f"feedback was: {feedback!r}"

    def test_filler_words_do_not_appear_in_feedback(self) -> None:
        """Structural filler ('add', 'please', 'i want') must not appear in feedback."""
        item = _item_with_side([("coke", "Coke")])
        slots = (SlotValue(name="ITEM", value="Test Burger"),)
        result = self._handle(item, slots, "i want a test burger please")
        feedback = result.response_payload.get("prefill_feedback", "")
        assert "i want" not in feedback.lower()
        assert "please" not in feedback.lower()

    def test_item_name_tokens_do_not_appear_in_feedback(self) -> None:
        """The item's own name ('Test Burger') must not echo back as unresolved."""
        item = _item_with_side([("coke", "Coke")])
        slots = (
            SlotValue(name="ITEM", value="Test Burger"),
            SlotValue(name="ITEM", value="Unknown Thing"),
        )
        result = self._handle(
            item, slots, "test burger and unknown thing"
        )
        feedback = result.response_payload.get("prefill_feedback", "")
        assert "test burger" not in feedback.lower()
        assert "burger" not in feedback.lower()

    def test_partial_success_flag_set_when_genuine_unresolved_entity(self) -> None:
        """partial_success=True iff there are genuine unresolved menu entities."""
        item = _item_with_side([("coke", "Coke"), ("sprite", "Sprite")], max_selector=1)
        slots = (
            SlotValue(name="ITEM", value="Test Burger"),
            SlotValue(name="ITEM", value="Coke"),
            SlotValue(name="ITEM", value="Rice"),
        )
        result = self._handle(item, slots, "test burger with coke and rice")
        payload = result.response_payload
        # rice is unresolved → partial_success
        assert payload.get("partial_success") is True
        assert "rice" in [e.lower() for e in (payload.get("unresolved_entities") or [])]

    def test_partial_success_false_when_only_quantity_unmatched(self) -> None:
        """Quantity words must not count as unresolved entities."""
        item = _item_with_side([("coke", "Coke")])
        slots = (
            SlotValue(name="QUANTITY", value="1"),
            SlotValue(name="ITEM", value="Test Burger"),
        )
        result = self._handle(item, slots, "one test burger")
        payload = result.response_payload
        assert payload.get("partial_success") is not True
        assert not payload.get("unresolved_entities")
