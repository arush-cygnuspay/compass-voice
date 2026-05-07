# tests/state_machine/handlers/item/add_item/test_add_item_flow_phase_f.py
"""Phase F regression tests — required modifier/side groups must be asked before size."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.menu.models import (
    MenuItem,
    ModifierChoice,
    ModifierGroup,
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


def _side_choice(item_id: str, name: str) -> SideChoice:
    return SideChoice(
        item_id=item_id,
        name=name,
        normalized_name=normalize_text(name),
        pricing=Pricing(mode="fixed", price_cents=0),
    )


def _modifier_choice(modifier_id: str, name: str) -> ModifierChoice:
    return ModifierChoice(
        modifier_id=modifier_id,
        name=name,
        normalized_name=normalize_text(name),
        price_cents=0,
    )


def _make_item(
    *,
    has_variants: bool = True,
    required_modifier_groups: list[ModifierGroup] | None = None,
    optional_modifier_groups: list[ModifierGroup] | None = None,
    required_side_groups: list[SideGroup] | None = None,
    optional_side_groups: list[SideGroup] | None = None,
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


def _req_modifier_group(group_id: str = "req_mod_1") -> ModifierGroup:
    return ModifierGroup(
        group_id=group_id,
        name="Choose protein",
        normalized_name=normalize_text("Choose protein"),
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[
            _modifier_choice("chicken", "Chicken"),
            _modifier_choice("beef", "Beef"),
        ],
    )


def _opt_modifier_group(group_id: str = "opt_mod_1") -> ModifierGroup:
    return ModifierGroup(
        group_id=group_id,
        name="Add extras",
        normalized_name=normalize_text("Add extras"),
        is_required=False,
        min_selector=0,
        max_selector=3,
        choices=[
            _modifier_choice("cheese", "Cheese"),
            _modifier_choice("jalapeno", "Jalapeno"),
        ],
    )


def _req_side_group(group_id: str = "req_side_1") -> SideGroup:
    return SideGroup(
        group_id=group_id,
        name="Choose your side",
        normalized_name=normalize_text("Choose your side"),
        is_required=True,
        min_selector=1,
        max_selector=1,
        choices=[
            _side_choice("fries", "Fries"),
            _side_choice("salad", "Salad"),
        ],
    )


def _opt_side_group(group_id: str = "opt_side_1") -> SideGroup:
    return SideGroup(
        group_id=group_id,
        name="Add a drink",
        normalized_name=normalize_text("Add a drink"),
        is_required=False,
        min_selector=0,
        max_selector=1,
        choices=[
            _side_choice("coke", "Coke"),
            _side_choice("water", "Water"),
        ],
    )


# ---------------------------------------------------------------------------
# Rule: required modifiers before size
# ---------------------------------------------------------------------------

class TestRequiredModifierBeforeSize:
    def test_required_modifier_asked_before_size(self):
        """Item with required modifier + size variants: modifier must come first."""
        item = _make_item(
            has_variants=True,
            required_modifier_groups=[_req_modifier_group()],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_MODIFIER
        assert step.response_key == "ask_for_modifier"
        assert context.current_prompt_field == "modifier"

    def test_size_asked_after_required_modifier_satisfied(self):
        """Once required modifier is satisfied, size is asked next."""
        item = _make_item(
            has_variants=True,
            required_modifier_groups=[_req_modifier_group()],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        context.selected_modifier_groups = {"req_mod_1": [MagicMock(modifier_id="chicken", name="Chicken", action="add", instruction=None)]}

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIZE
        assert step.response_key == "ask_for_size"

    def test_required_modifier_before_size_then_optional_after(self):
        """Full sequence: req_modifier → size → optional_modifier → finalize."""
        item = _make_item(
            has_variants=True,
            required_modifier_groups=[_req_modifier_group("req_mod")],
            optional_modifier_groups=[_opt_modifier_group("opt_mod")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)

        # Step 1: required modifier
        step = determine_next_add_item_step(context)
        assert step.next_state == ConversationState.WAITING_FOR_MODIFIER
        assert step.response_payload["group_name"] == "Choose protein"

        # Satisfy required modifier
        context.selected_modifier_groups = {"req_mod": [MagicMock(modifier_id="chicken", name="Chicken", action="add", instruction=None)]}

        # Step 2: size
        step = determine_next_add_item_step(context)
        assert step.next_state == ConversationState.WAITING_FOR_SIZE

        # Select size
        context.selected_variant_id = "small"

        # Step 3: optional modifier
        step = determine_next_add_item_step(context)
        assert step.next_state == ConversationState.WAITING_FOR_MODIFIER
        assert step.response_payload["group_name"] == "Add extras"


# ---------------------------------------------------------------------------
# Rule: required sides before size
# ---------------------------------------------------------------------------

class TestRequiredSideBeforeSize:
    def test_required_side_asked_before_size(self):
        """Item with required side group + size variants: side must come first."""
        item = _make_item(
            has_variants=True,
            required_side_groups=[_req_side_group()],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIDE
        assert step.response_key == "ask_for_side"
        assert context.current_prompt_field == "side"

    def test_size_asked_after_required_side_satisfied(self):
        item = _make_item(
            has_variants=True,
            required_side_groups=[_req_side_group()],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        context.selected_side_groups = {"req_side_1": ["fries"]}

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIZE


# ---------------------------------------------------------------------------
# Rule: required modifier → required side → size → optional modifier → optional side
# ---------------------------------------------------------------------------

class TestFullOrderingSequence:
    def test_ordering_req_mod_req_side_size_opt_mod_opt_side(self):
        """Complete ordering: req_modifier → req_side → size → opt_modifier → opt_side → finalize."""
        item = _make_item(
            has_variants=True,
            required_modifier_groups=[_req_modifier_group("req_mod")],
            optional_modifier_groups=[_opt_modifier_group("opt_mod")],
            required_side_groups=[_req_side_group("req_side")],
            optional_side_groups=[_opt_side_group("opt_side")],
        )
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)

        # 1. Required modifier first
        step = determine_next_add_item_step(context)
        assert step.next_state == ConversationState.WAITING_FOR_MODIFIER
        assert step.response_payload["group_name"] == "Choose protein"

        context.selected_modifier_groups = {"req_mod": [MagicMock(modifier_id="chicken", name="Chicken", action="add", instruction=None)]}

        # 2. Required side
        step = determine_next_add_item_step(context)
        assert step.next_state == ConversationState.WAITING_FOR_SIDE
        assert step.response_payload["group_name"] == "Choose your side"

        context.selected_side_groups = {"req_side": ["fries"]}

        # 3. Item size/variant
        step = determine_next_add_item_step(context)
        assert step.next_state == ConversationState.WAITING_FOR_SIZE

        context.selected_variant_id = "small"

        # 4. Optional modifier
        step = determine_next_add_item_step(context)
        assert step.next_state == ConversationState.WAITING_FOR_MODIFIER
        assert step.response_payload["group_name"] == "Add extras"

        context.skipped_modifier_groups = {"opt_mod"}

        # 5. Optional side
        step = determine_next_add_item_step(context)
        assert step.next_state == ConversationState.WAITING_FOR_SIDE
        assert step.response_payload["group_name"] == "Add a drink"

        context.skipped_side_groups = {"opt_side"}
        context.quantity = 1

        # 6. ReadyToFinalize
        step = determine_next_add_item_step(context)
        assert isinstance(step, ReadyToFinalize)


# ---------------------------------------------------------------------------
# Guard: no regressions when there are no groups
# ---------------------------------------------------------------------------

class TestNoGroupsStillAsksSize:
    def test_item_with_variants_only_still_asks_size(self):
        item = _make_item(has_variants=True)
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)

        step = determine_next_add_item_step(context)

        assert step.next_state == ConversationState.WAITING_FOR_SIZE

    def test_item_no_variants_no_groups_ready_to_finalize(self):
        item = _make_item(has_variants=False)
        context = ConversationContext()
        context.pending_add_item = build_pending_add_item(item)
        context.quantity = 1

        step = determine_next_add_item_step(context)

        assert isinstance(step, ReadyToFinalize)
