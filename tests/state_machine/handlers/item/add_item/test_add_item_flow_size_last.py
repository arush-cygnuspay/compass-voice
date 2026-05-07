# tests/state_machine/handlers/item/add_item/test_add_item_flow_size_last.py
"""Phase F — size/variant step must come after optional modifier and side groups."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.menu.models import (
    MenuItem,
    ModifierGroup,
    ModifierChoice,
    Pricing,
    PricingVariant,
    SideChoice,
    SideGroup,
)
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.add_item_flow import (
    AddItemNextStep,
    ReadyToFinalize,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _variant(variant_id: str, label: str) -> PricingVariant:
    return PricingVariant(
        variant_id=variant_id,
        label=label,
        normalized_label=normalize_text(label),
        price_cents=0,
    )


def _modifier_choice(modifier_id: str, name: str) -> ModifierChoice:
    return ModifierChoice(
        modifier_id=modifier_id,
        name=name,
        normalized_name=normalize_text(name),
        price_cents=0,
    )


def _side_choice(item_id: str, name: str) -> SideChoice:
    return SideChoice(
        item_id=item_id,
        name=name,
        normalized_name=normalize_text(name),
        pricing=Pricing(mode="fixed", price_cents=0),
    )


def _opt_modifier_group(group_id: str = "opt_mod") -> ModifierGroup:
    return ModifierGroup(
        group_id=group_id,
        name="Add extras",
        normalized_name=normalize_text("Add extras"),
        is_required=False,
        min_selector=0,
        max_selector=3,
        choices=[_modifier_choice("cheese", "Cheese"), _modifier_choice("jalapeno", "Jalapeno")],
    )


def _opt_side_group(group_id: str = "opt_side") -> SideGroup:
    return SideGroup(
        group_id=group_id,
        name="Add a drink",
        normalized_name=normalize_text("Add a drink"),
        is_required=False,
        min_selector=0,
        max_selector=1,
        choices=[_side_choice("coke", "Coke"), _side_choice("water", "Water")],
    )


def _req_modifier_group(group_id: str = "req_mod") -> ModifierGroup:
    return ModifierGroup(
        group_id=group_id,
        name="Choose protein",
        normalized_name=normalize_text("Choose protein"),
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[_modifier_choice("chicken", "Chicken"), _modifier_choice("beef", "Beef")],
    )


def _req_side_group(group_id: str = "req_side") -> SideGroup:
    return SideGroup(
        group_id=group_id,
        name="Choose your side",
        normalized_name=normalize_text("Choose your side"),
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[_side_choice("fries", "Fries"), _side_choice("salad", "Salad")],
    )


def _make_item(
    *,
    has_variants: bool = True,
    required_modifier_groups: list | None = None,
    optional_modifier_groups: list | None = None,
    required_side_groups: list | None = None,
    optional_side_groups: list | None = None,
) -> MenuItem:
    modifier_groups = list(required_modifier_groups or []) + list(optional_modifier_groups or [])
    side_groups = list(required_side_groups or []) + list(optional_side_groups or [])
    pricing = (
        Pricing(
            mode="variant",
            variants=[_variant("small", "Small"), _variant("large", "Large")],
        )
        if has_variants
        else Pricing(mode="fixed", price_cents=500)
    )
    return MenuItem(
        item_id="test_item",
        name="Test Item",
        normalized_name=normalize_text("Test Item"),
        aliases=(),
        normalized_aliases=(),
        voice_labels=("test item",),
        pricing=pricing,
        side_groups=side_groups,
        modifier_groups=modifier_groups,
        available=True,
    )


# ---------------------------------------------------------------------------
# Size comes after optional modifier groups
# ---------------------------------------------------------------------------

class TestSizeAfterOptionalModifiers:
    def test_optional_modifier_asked_before_size(self):
        """Item with optional modifier + size variants: optional modifier asked before size."""
        item = _make_item(
            has_variants=True,
            optional_modifier_groups=[_opt_modifier_group()],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_MODIFIER

    def test_size_asked_after_optional_modifier_skipped(self):
        item = _make_item(
            has_variants=True,
            optional_modifier_groups=[_opt_modifier_group("opt_mod")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        context.skipped_modifier_groups = {"opt_mod"}

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIZE

    def test_size_asked_after_optional_modifier_satisfied(self):
        item = _make_item(
            has_variants=True,
            optional_modifier_groups=[_opt_modifier_group("opt_mod")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        context.selected_modifier_groups = {
            "opt_mod": [MagicMock(modifier_id="cheese", name="Cheese", action="add", instruction=None)]
        }

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIZE


# ---------------------------------------------------------------------------
# Size comes after optional side groups
# ---------------------------------------------------------------------------

class TestSizeAfterOptionalSides:
    def test_optional_side_asked_before_size(self):
        """Item with optional side + size variants: optional side asked before size."""
        item = _make_item(
            has_variants=True,
            optional_side_groups=[_opt_side_group()],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIDE

    def test_size_asked_after_optional_side_skipped(self):
        item = _make_item(
            has_variants=True,
            optional_side_groups=[_opt_side_group("opt_side")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        context.skipped_side_groups = {"opt_side"}

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIZE

    def test_size_asked_after_optional_side_satisfied(self):
        item = _make_item(
            has_variants=True,
            optional_side_groups=[_opt_side_group("opt_side")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        context.selected_side_groups = {"opt_side": ["coke"]}

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIZE


# ---------------------------------------------------------------------------
# Full ordering: req_mod → req_side → opt_mod → opt_side → size
# ---------------------------------------------------------------------------

class TestSizeIsAlwaysLast:
    def test_size_is_last_step_all_groups_resolved(self):
        """When all groups are resolved, size is the next step."""
        item = _make_item(
            has_variants=True,
            required_modifier_groups=[_req_modifier_group("req_mod")],
            optional_modifier_groups=[_opt_modifier_group("opt_mod")],
            required_side_groups=[_req_side_group("req_side")],
            optional_side_groups=[_opt_side_group("opt_side")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        context.selected_modifier_groups = {
            "req_mod": [MagicMock(modifier_id="chicken", name="Chicken", action="add", instruction=None)]
        }
        context.skipped_modifier_groups = {"opt_mod"}
        context.selected_side_groups = {"req_side": ["fries"]}
        context.skipped_side_groups = {"opt_side"}

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIZE

    def test_size_not_asked_while_optional_modifier_pending(self):
        """Size must not be asked while an optional modifier group is unresolved."""
        item = _make_item(
            has_variants=True,
            required_modifier_groups=[_req_modifier_group("req_mod")],
            optional_modifier_groups=[_opt_modifier_group("opt_mod")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        # required modifier satisfied; optional modifier NOT resolved yet
        context.selected_modifier_groups = {
            "req_mod": [MagicMock(modifier_id="chicken", name="Chicken", action="add", instruction=None)]
        }

        step = determine_next_add_item_step(context)

        # Must be optional modifier, not size
        assert step.next_state == ConversationState.WAITING_FOR_MODIFIER
        assert step.next_state != ConversationState.WAITING_FOR_SIZE

    def test_size_not_asked_while_optional_side_pending(self):
        """Size must not be asked while an optional side group is unresolved."""
        item = _make_item(
            has_variants=True,
            optional_side_groups=[_opt_side_group("opt_side")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        # optional side NOT resolved yet

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIDE
        assert step.next_state != ConversationState.WAITING_FOR_SIZE

    def test_ready_to_finalize_after_size_set(self):
        """After size and quantity are set with all groups resolved, ReadyToFinalize."""
        item = _make_item(
            has_variants=True,
            optional_modifier_groups=[_opt_modifier_group("opt_mod")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        context.skipped_modifier_groups = {"opt_mod"}
        context.selected_variant_id = "small"
        context.quantity = 1

        step = determine_next_add_item_step(context)

        assert isinstance(step, ReadyToFinalize)
