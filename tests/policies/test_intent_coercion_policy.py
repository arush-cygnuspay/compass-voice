# tests/policies/test_intent_coercion_policy.py
"""Phase E — IntentCoercionPolicy unit tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_result import NLUResult, SlotValue
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.policy.intent_coercion import IntentCoercionPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nlu(intent=Intent.UNKNOWN, slots=(), confidence=0.9) -> NLUResult:
    return NLUResult(
        effective_intent=intent,
        intent_confidence=confidence,
        raw_text="test",
        normalized_text="test",
        slots=slots,
    )


def _make_slot(name: str, value: str) -> SlotValue:
    return SlotValue(name=name, value=value)


def _make_cart(empty: bool = True, item_names: list[str] | None = None):
    """Build a cart mock matching the real Cart API: is_empty() + get_items().

    Each cart item has only item_id (no name — names come from menu_repo.get_item).
    item_names are used to build matching item_ids so _make_menu_repo can resolve them.
    """
    cart = MagicMock()
    cart.is_empty.return_value = empty
    if not empty and item_names:
        cart_items = []
        for n in item_names:
            ci = MagicMock()
            ci.item_id = n.lower().replace(" ", "_")  # deterministic fake id
            cart_items.append(ci)
        cart.get_items.return_value = cart_items
    else:
        cart.get_items.return_value = []
    return cart


def _make_menu_repo(matched_item_name: str | None = None):
    """Return a menu_repo mock.

    Handles two access paths:
    - repo.store.find_* — used by _has_menu_evidence (Rule 1)
    - repo.get_item(item_id) — used by _has_cart_target_for_item (Rule 2)

    item_id convention (must match _make_cart): name.lower().replace(' ', '_').
    """
    repo = MagicMock()
    store = MagicMock()
    repo.store = store

    def _find_exact(norm):
        if matched_item_name and norm in {matched_item_name.lower(), matched_item_name}:
            return MagicMock()
        return None

    def _find_alias(norm):
        if matched_item_name and norm in {matched_item_name.lower(), matched_item_name}:
            return [MagicMock()]
        return []

    def _find_voice(norm):
        if matched_item_name and norm in {matched_item_name.lower(), matched_item_name}:
            return ["item_123"]
        return []

    def _find_entity(norm, **kwargs):
        if matched_item_name and norm in {matched_item_name.lower(), matched_item_name}:
            return [{"type": "item", "item_id": "item_123"}]
        return []

    def _get_item(item_id: str):
        """Reverse the _make_cart id convention to recover item name."""
        if matched_item_name and item_id == matched_item_name.lower().replace(" ", "_"):
            menu_item = MagicMock()
            menu_item.name = matched_item_name
            return menu_item
        raise KeyError(item_id)

    store.find_item_exact.side_effect = _find_exact
    store.find_item_ids_by_alias.side_effect = _find_alias
    store.find_item_ids_by_voice_label.side_effect = _find_voice
    store.find_entity.side_effect = _find_entity
    repo.get_item.side_effect = _get_item
    return repo


def _policy(matched_name: str | None = None) -> IntentCoercionPolicy:
    return IntentCoercionPolicy(menu_repo=_make_menu_repo(matched_name))


# ---------------------------------------------------------------------------
# Rule 1 — UNKNOWN + ITEM slot + menu evidence → ADD_ITEM
# ---------------------------------------------------------------------------

class TestRule1IdleUnknownWithMenuEvidence:
    def test_unknown_with_item_evidence_coerced(self):
        policy = _policy("double bacon burger")
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.UNKNOWN, raw_text="i will take double bacon burger"),
            nlu=_make_nlu(slots=(
                _make_slot("ITEM", "double bacon burger"),
            )),
            cart=_make_cart(empty=True),
        )
        assert result.intent_result.intent == Intent.ADD_ITEM
        assert result.coercion_reason == "idle_item_slot_with_menu_evidence"

    def test_unknown_no_item_slot_not_coerced(self):
        policy = _policy()
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.UNKNOWN, raw_text="hmm"),
            nlu=_make_nlu(slots=()),
            cart=_make_cart(),
        )
        assert result.intent_result.intent == Intent.UNKNOWN
        assert result.coercion_reason is None

    def test_unknown_item_slot_no_menu_evidence_not_coerced(self):
        policy = _policy(matched_name=None)  # nothing matches
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.UNKNOWN, raw_text="i will take it"),
            nlu=_make_nlu(slots=(
                _make_slot("ITEM", "it"),
            )),
            cart=_make_cart(),
        )
        assert result.intent_result.intent == Intent.UNKNOWN
        assert result.coercion_reason is None

    def test_add_item_with_evidence_no_rewrite(self):
        """ADD_ITEM already correct — no rewrite, coercion_reason should be None."""
        policy = _policy("sprite")
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.ADD_ITEM, raw_text="sprite"),
            nlu=_make_nlu(intent=Intent.ADD_ITEM, slots=(_make_slot("ITEM", "sprite"),)),
            cart=_make_cart(),
        )
        assert result.intent_result.intent == Intent.ADD_ITEM
        assert result.coercion_reason is None


# ---------------------------------------------------------------------------
# Rule 2 — MODIFY_ITEM/REPLACE_ITEM + ITEM + no cart target → ADD_ITEM
# ---------------------------------------------------------------------------

class TestRule2IdleModifyNoTarget:
    def test_modify_item_empty_cart_coerced(self):
        policy = _policy("sprite")
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.MODIFY_ITEM, raw_text="change sprite to large"),
            nlu=_make_nlu(slots=(_make_slot("ITEM", "sprite"),)),
            cart=_make_cart(empty=True),
        )
        assert result.intent_result.intent == Intent.ADD_ITEM
        assert "no_target" in (result.coercion_reason or "")

    def test_replace_item_no_cart_target_coerced(self):
        policy = _policy("coke")
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.REPLACE_ITEM, raw_text="swap sprite with coke"),
            nlu=_make_nlu(slots=(_make_slot("ITEM", "coke"),)),
            cart=_make_cart(empty=True),
        )
        assert result.intent_result.intent == Intent.ADD_ITEM

    def test_modify_item_has_cart_target_not_coerced(self):
        policy = _policy("sprite")
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.MODIFY_ITEM, raw_text="make my sprite large"),
            nlu=_make_nlu(slots=(_make_slot("ITEM", "sprite"),)),
            cart=_make_cart(empty=False, item_names=["Sprite"]),
        )
        # Cart has Sprite — coercion must NOT fire.
        assert result.intent_result.intent == Intent.MODIFY_ITEM
        assert result.coercion_reason is None


# ---------------------------------------------------------------------------
# Rule 3 — MODIFY_ITEM + ITEM+SIZE + empty cart → ADD_ITEM
# ---------------------------------------------------------------------------

class TestRule3IdleItemVariantNoCartTarget:
    def test_modify_item_size_slot_empty_cart_coerced(self):
        policy = _policy()
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.MODIFY_ITEM, raw_text="i want a sprite in medium"),
            nlu=_make_nlu(slots=(
                _make_slot("ITEM", "sprite"),
                _make_slot("SIZE", "medium"),
            )),
            cart=_make_cart(empty=True),
        )
        assert result.intent_result.intent == Intent.ADD_ITEM
        assert result.coercion_reason == "idle_item_variant_no_cart_target"

    def test_modify_item_variant_slot_empty_cart_coerced(self):
        policy = _policy()
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.MODIFY_ITEM, raw_text="medium sprite"),
            nlu=_make_nlu(slots=(
                _make_slot("ITEM", "sprite"),
                _make_slot("VARIANT", "medium"),
            )),
            cart=_make_cart(empty=True),
        )
        assert result.intent_result.intent == Intent.ADD_ITEM
        assert result.coercion_reason == "idle_item_variant_no_cart_target"


# ---------------------------------------------------------------------------
# Guard — non-idle states must NOT be coerced
# ---------------------------------------------------------------------------

class TestGuardNonIdleStates:
    @pytest.mark.parametrize("state", [
        ConversationState.WAITING_FOR_SIDE,
        ConversationState.WAITING_FOR_MODIFIER,
        ConversationState.WAITING_FOR_SIZE,
        ConversationState.CONFIRMING_ITEM,
        ConversationState.WAITING_FOR_PAYMENT,
    ])
    def test_non_idle_not_coerced(self, state):
        policy = _policy("sprite")
        result = policy.coerce(
            state=state,
            intent_result=IntentResult(intent=Intent.UNKNOWN, raw_text="sprite"),
            nlu=_make_nlu(slots=(_make_slot("ITEM", "sprite"),)),
            cart=_make_cart(),
        )
        assert result.intent_result.intent == Intent.UNKNOWN
        assert result.coercion_reason is None


# ---------------------------------------------------------------------------
# Guard — protected intents must never be coerced
# ---------------------------------------------------------------------------

class TestGuardProtectedIntents:
    @pytest.mark.parametrize("intent", [
        Intent.CHECKOUT,
        Intent.CANCEL_ORDER,
        Intent.REMOVE_ITEM,
        Intent.PAYMENT_STATUS,
    ])
    def test_protected_intent_not_coerced(self, intent):
        policy = _policy("sprite")
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=intent, raw_text="test"),
            nlu=_make_nlu(intent=intent, slots=(_make_slot("ITEM", "sprite"),)),
            cart=_make_cart(),
        )
        assert result.intent_result.intent == intent
        assert result.coercion_reason is None


# ---------------------------------------------------------------------------
# Fix 1 regression — AFFIRM/DENY/CONFIRM/CANCEL must never become ADD_ITEM
# ---------------------------------------------------------------------------

class TestGuardAffirmDenyControlIntents:
    """User control responses must never be coerced to ADD_ITEM even when
    an ITEM slot is present.  "yeah, coke" (AFFIRM) or "no, spicy chicken"
    (DENY) are replies to in-progress yes/no questions, not new order intents.
    """

    @pytest.mark.parametrize("intent", [
        Intent.AFFIRM,
        Intent.DENY,
        Intent.CONFIRM,
        Intent.CANCEL,
    ])
    def test_control_intent_with_item_slot_not_coerced(self, intent):
        """AFFIRM/DENY/CONFIRM/CANCEL are never rewritten to ADD_ITEM."""
        policy = _policy("sprite")
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=intent, raw_text="yeah sprite"),
            nlu=_make_nlu(intent=intent, slots=(_make_slot("ITEM", "sprite"),)),
            cart=_make_cart(empty=True),
        )
        assert result.intent_result.intent == intent
        assert result.coercion_reason is None

    def test_unknown_with_item_slot_still_coerced(self):
        """UNKNOWN + item evidence still coerces — regression guard for Rule 1."""
        policy = _policy("sprite")
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.UNKNOWN, raw_text="sprite"),
            nlu=_make_nlu(slots=(_make_slot("ITEM", "sprite"),)),
            cart=_make_cart(empty=True),
        )
        assert result.intent_result.intent == Intent.ADD_ITEM
        assert result.coercion_reason == "idle_item_slot_with_menu_evidence"

    def test_modify_with_item_variant_empty_cart_still_coerced(self):
        """MODIFY_ITEM + SIZE + empty cart still coerces — regression guard for Rule 3."""
        policy = _policy()
        result = policy.coerce(
            state=ConversationState.IDLE,
            intent_result=IntentResult(intent=Intent.MODIFY_ITEM, raw_text="large sprite"),
            nlu=_make_nlu(slots=(
                _make_slot("ITEM", "sprite"),
                _make_slot("SIZE", "large"),
            )),
            cart=_make_cart(empty=True),
        )
        assert result.intent_result.intent == Intent.ADD_ITEM
        assert result.coercion_reason == "idle_item_variant_no_cart_target"
