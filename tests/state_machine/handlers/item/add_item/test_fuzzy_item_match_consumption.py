# tests/state_machine/handlers/item/add_item/test_fuzzy_item_match_consumption.py
"""Tests that accepted fuzzy item matches do not leak into unresolved feedback.

Production failure: "port stickers" accepted as "Pot Stickers" → response said
"I couldn't find port stickers. Pot Stickers added. I couldn't find port stickers."

Root causes fixed:
1. ConsumedTokenLedger.consume_match() records the raw query phrase ("port stickers")
   so its tokens do not survive the deduplication pass as unresolved entities.
2. MultiGroupPrefillEngine skips consumed ITEM slot values in _build_candidate_phrases.
3. ItemResolutionHandler propagates accepted_item_raw_query to PrefillOrchestrator.
"""
from __future__ import annotations

import pytest

from app.menu.models import MenuItem, Pricing, SideChoice, SideGroup
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.add_item_handler import AddItemHandler
from app.state_machine.handlers.item.add_item.consumed_token_ledger import ConsumedTokenLedger
from app.state_machine.handlers.item.add_item.multi_group_prefill import (
    MultiGroupPrefillEngine,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _no_group_item(item_id: str, name: str) -> MenuItem:
    return MenuItem(
        item_id=item_id,
        name=name,
        normalized_name=normalize_text(name),
        aliases=(name.lower(),),
        normalized_aliases=(normalize_text(name),),
        voice_labels=(name.lower(),),
        pricing=Pricing(mode="fixed", price_cents=800),
        side_groups=[],
        modifier_groups=[],
        available=True,
    )


def _item_with_optional_side(item_id: str, name: str) -> MenuItem:
    return MenuItem(
        item_id=item_id,
        name=name,
        normalized_name=normalize_text(name),
        aliases=(name.lower(),),
        normalized_aliases=(normalize_text(name),),
        voice_labels=(name.lower(),),
        pricing=Pricing(mode="fixed", price_cents=800),
        side_groups=[
            SideGroup(
                group_id="drinks",
                name="Choose a drink",
                normalized_name=normalize_text("Choose a drink"),
                is_required=False,
                min_selector=0,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="coke",
                        name="Coke",
                        normalized_name=normalize_text("Coke"),
                        pricing=Pricing(mode="fixed", price_cents=0),
                    ),
                ],
            )
        ],
        modifier_groups=[],
        available=True,
    )


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


# ---------------------------------------------------------------------------
# ConsumedTokenLedger unit tests
# ---------------------------------------------------------------------------


class TestConsumedTokenLedger:
    def test_add_phrase_registers_exact_phrase(self) -> None:
        ledger = ConsumedTokenLedger()
        ledger.add_phrase("port stickers")
        assert ledger.is_consumed_phrase("port stickers")

    def test_add_phrase_individual_tokens_consumed(self) -> None:
        ledger = ConsumedTokenLedger()
        ledger.add_phrase("port stickers")
        tokens = ledger.tokens()
        assert "port" in tokens
        # tokenizer may stem "stickers" to "sticker"
        assert "sticker" in tokens or "stickers" in tokens

    def test_consume_match_raw_query_is_consumed_phrase(self) -> None:
        ledger = ConsumedTokenLedger()
        ledger.consume_match(raw_query="port stickers", canonical_name="Pot Stickers")
        assert ledger.is_consumed_phrase("port stickers")

    def test_consume_match_canonical_tokens_consumed(self) -> None:
        ledger = ConsumedTokenLedger()
        ledger.consume_match(raw_query="port stickers", canonical_name="Pot Stickers")
        tokens = ledger.tokens()
        assert "pot" in tokens
        # tokenizer may stem "stickers" to "sticker"
        assert "sticker" in tokens or "stickers" in tokens

    def test_is_consumed_phrase_false_for_unknown(self) -> None:
        ledger = ConsumedTokenLedger()
        ledger.consume_match(raw_query="port stickers", canonical_name="Pot Stickers")
        assert not ledger.is_consumed_phrase("crab rangoon")

    def test_consumed_phrases_returns_frozenset(self) -> None:
        ledger = ConsumedTokenLedger()
        ledger.consume_match(raw_query="port stickers", canonical_name="Pot Stickers")
        phrases = ledger.consumed_phrases()
        assert isinstance(phrases, frozenset)
        assert "port stickers" in phrases


# ---------------------------------------------------------------------------
# MultiGroupPrefillEngine: consumed_phrases suppresses raw query candidates
# ---------------------------------------------------------------------------


class TestPrefillEngineConsumedPhrases:
    def test_raw_query_in_consumed_phrases_not_in_unresolved(self) -> None:
        """'port stickers' in consumed_phrases must not appear in unresolved_phrases."""
        item = _no_group_item("pot_stickers", "Pot Stickers")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="port stickers",
            slots=(SlotValue(name="ITEM", value="port stickers"),),
            consumed_tokens=frozenset({"port", "stickers", "pot"}),
            consumed_phrases=frozenset({"port stickers"}),
        )

        for phrase in result.unresolved_phrases:
            assert "port" not in phrase.lower(), (
                f"'port stickers' raw query leaked into unresolved: {result.unresolved_phrases}"
            )
            assert "stickers" not in phrase.lower() or phrase.lower() not in {"stickers"}, (
                f"token from raw query leaked into unresolved: {result.unresolved_phrases}"
            )

    def test_consumed_item_slot_value_not_in_unresolved(self) -> None:
        """ITEM slot value in consumed_phrases is skipped in step 2a."""
        item = _no_group_item("pot_stickers", "Pot Stickers")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="i want a port stickers",
            slots=(SlotValue(name="ITEM", value="port stickers"),),
            consumed_tokens=frozenset({"port", "stickers"}),
            consumed_phrases=frozenset({"port stickers"}),
        )

        assert "port stickers" not in result.unresolved_phrases

    def test_non_consumed_item_slot_value_can_bind_as_side(self) -> None:
        """ITEM slot value NOT in consumed_phrases is still eligible for side binding."""
        item = _item_with_optional_side("burger_1", "Test Burger")
        pending = build_pending_add_item(item)
        engine = MultiGroupPrefillEngine()

        result = engine.prefill(
            pending=pending,
            segment_text="test burger with coke",
            slots=(
                SlotValue(name="ITEM", value="Test Burger"),
                SlotValue(name="ITEM", value="Coke"),
            ),
            consumed_tokens=frozenset({"test", "burger"}),
            consumed_phrases=frozenset({"test burger"}),
        )

        assert "drinks" in result.side_selections, (
            f"'Coke' should be bound to the drink side group; "
            f"side_selections={result.side_selections!r}"
        )


# ---------------------------------------------------------------------------
# AddItemHandler end-to-end: fuzzy item match → no duplicate feedback
# ---------------------------------------------------------------------------


class TestAddItemHandlerFuzzyMatch:
    def _handle_with_asr_variant(
        self,
        canonical_item: MenuItem,
        raw_asr_text: str,
        *,
        extra_slots: tuple[SlotValue, ...] = (),
    ):
        repo = _FakeMenuRepo(canonical_item)
        handler = AddItemHandler(repo)
        context = ConversationContext()
        context.last_slots = (
            SlotValue(name="ITEM", value=raw_asr_text),
            *extra_slots,
        )
        return handler.handle(
            intent=Intent.ADD_ITEM,
            context=context,
            user_text=normalize_text(f"i want a {raw_asr_text}"),
            session=None,
        )

    def test_port_stickers_no_couldnt_find_in_feedback(self) -> None:
        """Core regression: 'port stickers' accepted as 'Pot Stickers' → no feedback."""
        item = _no_group_item("pot_stickers", "Pot Stickers")
        result = self._handle_with_asr_variant(item, "port stickers")

        assert result.next_state == ConversationState.IDLE
        assert result.response_key == "item_added_successfully"

        payload = result.response_payload
        feedback = payload.get("prefill_feedback", "")
        assert "couldn't find" not in feedback.lower(), (
            f"feedback should be empty for clean add; got: {feedback!r}"
        )
        unresolved = payload.get("unresolved_entities") or []
        assert not any("port" in e.lower() for e in unresolved), (
            f"'port stickers' raw query leaked as unresolved: {unresolved}"
        )

    def test_accepted_fuzzy_match_does_not_set_partial_success(self) -> None:
        """Clean fuzzy add must not set partial_success=True."""
        item = _no_group_item("pot_stickers", "Pot Stickers")
        result = self._handle_with_asr_variant(item, "port stickers")

        assert result.response_payload.get("partial_success") is not True

    def test_exact_match_still_clean(self) -> None:
        """Sanity: exact match produces item_added_successfully with no feedback."""
        item = _no_group_item("pot_stickers", "Pot Stickers")
        repo = _FakeMenuRepo(item)
        handler = AddItemHandler(repo)
        context = ConversationContext()
        context.last_slots = (SlotValue(name="ITEM", value="Pot Stickers"),)
        result = handler.handle(
            intent=Intent.ADD_ITEM,
            context=context,
            user_text="pot stickers",
            session=None,
        )

        assert result.next_state == ConversationState.IDLE
        assert result.response_key == "item_added_successfully"
        feedback = result.response_payload.get("prefill_feedback", "")
        assert "couldn't find" not in feedback.lower()
