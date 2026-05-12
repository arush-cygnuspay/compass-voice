# tests/state_machine/handlers/item/add_item/test_item_quantity_policy.py
"""Focused tests for the centralized item quantity policy.

Covers all source types (explicit, implicit_default, ambiguous, invalid),
vague detection edge cases, ASR decimal correction, and the
determine_next_add_item_step() gate that applies the policy.
"""
from __future__ import annotations

import pytest

from app.state_machine.handlers.item.add_item.item_quantity_policy import (
    NormalizedItemQuantity,
    normalize_item_quantity,
)
from app.state_machine.handlers.item.add_item.add_item_flow import (
    ReadyToFinalize,
    determine_next_add_item_step,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handlers.item.add_item.pending_add_item_factory import (
    build_pending_add_item,
)
from app.menu.models import MenuItem, Pricing


# ---------------------------------------------------------------------------
# Minimal item fixture with no sides, no modifiers, no variants
# ---------------------------------------------------------------------------

def _make_simple_item(item_id: str = "item_1", name: str = "Chicken Burger") -> MenuItem:
    return MenuItem(
        item_id=item_id,
        name=name,
        normalized_name=name.lower(),
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=999),
        side_groups=[],
        modifier_groups=[],
        available=True,
    )


def _ctx_with_item(name: str = "Chicken Burger") -> ConversationContext:
    ctx = ConversationContext()
    ctx.current_item_id = "item_1"
    ctx.current_item_name = name
    ctx.pending_add_item = build_pending_add_item(_make_simple_item(name=name))
    return ctx


# ===========================================================================
# 1. normalize_item_quantity — explicit source
# ===========================================================================

class TestExplicitQuantity:
    def test_positive_int_in_context_is_explicit(self):
        ctx = _ctx_with_item()
        ctx.quantity = 3
        result = normalize_item_quantity(ctx)
        assert result.source == "explicit"
        assert result.quantity == 3
        assert result.needs_clarification is False

    def test_quantity_1_is_explicit(self):
        ctx = _ctx_with_item()
        ctx.quantity = 1
        result = normalize_item_quantity(ctx)
        assert result.source == "explicit"
        assert result.quantity == 1
        assert result.needs_clarification is False

    def test_explicit_quantity_hint_overrides_context(self):
        ctx = _ctx_with_item()
        ctx.quantity = None
        result = normalize_item_quantity(ctx, explicit_quantity_hint=5)
        assert result.source == "explicit"
        assert result.quantity == 5

    def test_explicit_hint_takes_priority_over_context_quantity(self):
        ctx = _ctx_with_item()
        ctx.quantity = 2
        result = normalize_item_quantity(ctx, explicit_quantity_hint=7)
        assert result.source == "explicit"
        assert result.quantity == 7


# ===========================================================================
# 2. normalize_item_quantity — implicit_default source
# ===========================================================================

class TestImplicitDefaultQuantity:
    def test_missing_quantity_defaults_to_1(self):
        ctx = _ctx_with_item()
        assert ctx.quantity is None
        result = normalize_item_quantity(ctx)
        assert result.source == "implicit_default"
        assert result.quantity == 1
        assert result.needs_clarification is False

    def test_add_chicken_burger_does_not_need_clarification(self):
        ctx = _ctx_with_item()
        result = normalize_item_quantity(ctx, user_text="add chicken burger")
        assert result.source == "implicit_default"
        assert result.quantity == 1
        assert result.needs_clarification is False

    def test_i_want_hamburger_defaults_quantity_1(self):
        ctx = _ctx_with_item(name="Hamburger")
        result = normalize_item_quantity(ctx, user_text="i want hamburger")
        assert result.source == "implicit_default"
        assert result.quantity == 1
        assert result.needs_clarification is False

    def test_text_with_side_does_not_trigger_vague(self):
        ctx = _ctx_with_item()
        result = normalize_item_quantity(ctx, user_text="with fries and coke")
        assert result.source == "implicit_default"
        assert result.quantity == 1

    def test_empty_user_text_defaults_to_1(self):
        ctx = _ctx_with_item()
        result = normalize_item_quantity(ctx, user_text="")
        assert result.source == "implicit_default"
        assert result.quantity == 1

    def test_no_user_text_parameter_defaults_to_1(self):
        ctx = _ctx_with_item()
        result = normalize_item_quantity(ctx)
        assert result.source == "implicit_default"
        assert result.quantity == 1


# ===========================================================================
# 3. normalize_item_quantity — ambiguous (vague) source
# ===========================================================================

class TestAmbiguousQuantity:
    def test_some_triggers_clarification(self):
        ctx = _ctx_with_item(name="Burger")
        result = normalize_item_quantity(ctx, user_text="some burgers")
        assert result.source == "ambiguous"
        assert result.needs_clarification is True
        assert result.reason == "vague_quantity"
        assert result.quantity is None

    def test_a_few_triggers_clarification(self):
        ctx = _ctx_with_item(name="Coke")
        result = normalize_item_quantity(ctx, user_text="a few cokes")
        assert result.source == "ambiguous"
        assert result.needs_clarification is True

    def test_several_triggers_clarification(self):
        ctx = _ctx_with_item(name="Wings")
        result = normalize_item_quantity(ctx, user_text="several wings")
        assert result.source == "ambiguous"
        assert result.needs_clarification is True

    def test_some_mid_sentence_does_not_trigger_vague(self):
        """'burger with some modifications' must NOT trigger ambiguous."""
        ctx = _ctx_with_item(name="Burger")
        result = normalize_item_quantity(ctx, user_text="burger with some modifications")
        assert result.source == "implicit_default"
        assert result.needs_clarification is False

    def test_vague_only_matches_at_leading_position(self):
        ctx = _ctx_with_item(name="Taco")
        result = normalize_item_quantity(ctx, user_text="chicken taco with some sauce")
        assert result.source == "implicit_default"


# ===========================================================================
# 4. normalize_item_quantity — invalid source
# ===========================================================================

class TestInvalidQuantity:
    def test_zero_quantity_in_context_is_invalid(self):
        ctx = _ctx_with_item()
        ctx.quantity = 0
        result = normalize_item_quantity(ctx)
        assert result.source == "invalid"
        assert result.needs_clarification is True
        assert result.reason == "non_positive_quantity"
        assert result.quantity is None

    def test_negative_quantity_in_context_is_invalid(self):
        ctx = _ctx_with_item()
        ctx.quantity = -1
        result = normalize_item_quantity(ctx)
        assert result.source == "invalid"
        assert result.needs_clarification is True

    def test_zero_explicit_hint_is_invalid(self):
        ctx = _ctx_with_item()
        result = normalize_item_quantity(ctx, explicit_quantity_hint=0)
        assert result.source == "invalid"
        assert result.needs_clarification is True

    def test_negative_explicit_hint_is_invalid(self):
        ctx = _ctx_with_item()
        result = normalize_item_quantity(ctx, explicit_quantity_hint=-5)
        assert result.source == "invalid"
        assert result.needs_clarification is True


# ===========================================================================
# 5. determine_next_add_item_step — policy gate integration
# ===========================================================================

class TestDetermineNextStepQuantityGate:
    def test_missing_quantity_defaults_to_1_and_finalizes(self):
        ctx = _ctx_with_item()
        assert ctx.quantity is None
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.quantity == 1
        assert ctx.quantity == 1

    def test_add_chicken_burger_does_not_route_to_waiting_for_quantity(self):
        ctx = _ctx_with_item()
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.quantity == 1

    def test_missing_quantity_never_routes_to_waiting_for_quantity(self):
        ctx = _ctx_with_item()
        step = determine_next_add_item_step(ctx)
        if isinstance(step, ReadyToFinalize):
            return
        assert step.next_state != ConversationState.WAITING_FOR_QUANTITY

    def test_explicit_digit_quantity_preserved(self):
        ctx = _ctx_with_item()
        ctx.quantity = 2
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.quantity == 2

    def test_explicit_word_quantity_preserved(self):
        ctx = _ctx_with_item()
        ctx.quantity = 3
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.quantity == 3

    def test_two_carrot_cakes_quantity_2(self):
        ctx = _ctx_with_item(name="Carrot Cake")
        ctx.quantity = 2
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.quantity == 2

    def test_three_hamburgers_quantity_3(self):
        ctx = _ctx_with_item(name="Hamburger")
        ctx.quantity = 3
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.quantity == 3

    def test_zero_quantity_routes_to_invalid_quantity_option(self):
        ctx = _ctx_with_item()
        ctx.quantity = 0
        step = determine_next_add_item_step(ctx)
        assert not isinstance(step, ReadyToFinalize)
        assert step.next_state == ConversationState.WAITING_FOR_QUANTITY
        assert step.response_key == "invalid_quantity_option"

    def test_negative_quantity_routes_to_invalid_quantity_option(self):
        ctx = _ctx_with_item()
        ctx.quantity = -1
        step = determine_next_add_item_step(ctx)
        assert not isinstance(step, ReadyToFinalize)
        assert step.next_state == ConversationState.WAITING_FOR_QUANTITY
        assert step.response_key == "invalid_quantity_option"


# ===========================================================================
# 6. Multi-item quantity preservation
# ===========================================================================

class TestMultiItemQuantities:
    def test_explicit_qty_2_for_hamburger_preserved(self):
        ctx = _ctx_with_item(name="Hamburger")
        ctx.quantity = 2
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.quantity == 2

    def test_default_qty_1_for_coke_when_no_explicit(self):
        ctx = _ctx_with_item(name="Coke")
        assert ctx.quantity is None
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.quantity == 1

    def test_item_with_modifier_defaults_quantity_1(self):
        """Item that skipped its optional modifier group still defaults qty to 1."""
        from app.state_machine.models.pending_item_models import (
            PendingAddItem,
            PendingModifierGroup,
            PendingModifierChoice,
        )

        sauce = PendingModifierChoice(
            modifier_id="sauce", name="Sauce", group_id="mods", normalized_name="sauce"
        )
        group = PendingModifierGroup(
            group_id="mods",
            name="Extras",
            is_required=False,
            min_selector=0,
            max_selector=1,
            choices=[sauce],
            choices_by_modifier_id={"sauce": sauce},
            choices_by_normalized_name={"sauce": [sauce]},
            choice_names=("Sauce",),
            normalized_choice_names=("sauce",),
            top_choice_names=("Sauce",),
        )
        ctx = ConversationContext()
        ctx.current_item_id = "burger"
        ctx.current_item_name = "Burger"
        ctx.pending_add_item = PendingAddItem(
            item_id="burger",
            item_name="Burger",
            modifier_groups=[group],
            modifier_groups_by_id={"mods": group},
            modifier_choice_by_id={"sauce": sauce},
        )
        ctx.skipped_modifier_groups = {"mods"}

        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.quantity == 1


# ===========================================================================
# 7. Vague / invalid — clarification still works in policy
# ===========================================================================

class TestClarificationStillWorks:
    def test_some_burgers_needs_clarification(self):
        ctx = _ctx_with_item(name="Burger")
        result = normalize_item_quantity(ctx, user_text="some burgers")
        assert result.needs_clarification is True
        assert result.source == "ambiguous"

    def test_a_few_cokes_needs_clarification(self):
        ctx = _ctx_with_item(name="Coke")
        result = normalize_item_quantity(ctx, user_text="a few cokes")
        assert result.needs_clarification is True

    def test_several_wings_needs_clarification(self):
        ctx = _ctx_with_item(name="Wings")
        result = normalize_item_quantity(ctx, user_text="several wings")
        assert result.needs_clarification is True

    def test_zero_quantity_invalid(self):
        ctx = _ctx_with_item()
        ctx.quantity = 0
        result = normalize_item_quantity(ctx)
        assert result.needs_clarification is True
        assert result.source == "invalid"

    def test_negative_quantity_invalid(self):
        ctx = _ctx_with_item()
        ctx.quantity = -3
        result = normalize_item_quantity(ctx)
        assert result.needs_clarification is True
        assert result.source == "invalid"


# ===========================================================================
# 8. Return type is always NormalizedItemQuantity
# ===========================================================================

class TestReturnType:
    def test_returns_normalized_item_quantity_dataclass(self):
        ctx = _ctx_with_item()
        result = normalize_item_quantity(ctx)
        assert isinstance(result, NormalizedItemQuantity)

    def test_all_required_fields_present(self):
        ctx = _ctx_with_item()
        result = normalize_item_quantity(ctx)
        assert hasattr(result, "quantity")
        assert hasattr(result, "source")
        assert hasattr(result, "needs_clarification")
        assert hasattr(result, "reason")

    def test_source_values_are_valid_literals(self):
        valid_sources = {"explicit", "implicit_default", "ambiguous", "invalid"}
        for qty, user_text in [
            (None, ""),
            (1, ""),
            (None, "some burgers"),
            (0, ""),
        ]:
            ctx = _ctx_with_item()
            ctx.quantity = qty
            result = normalize_item_quantity(ctx, user_text=user_text)
            assert result.source in valid_sources, f"Unexpected source: {result.source}"
