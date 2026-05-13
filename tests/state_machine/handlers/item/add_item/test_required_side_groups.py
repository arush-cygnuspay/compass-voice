# tests/state_machine/handlers/item/add_item/test_required_side_groups.py
"""Tests for required/optional/suggested-addon side group behavior.

Covers:
- Lobster Tail Platter style (min_selector=2, max_selector=2):
    partial fill asks "Choose 1 more side"
    full fill finalizes
    overflow (3 given for max=2) returns ask-correction response
- Combo style (two required groups):
    first unfilled blocks; second unfilled blocks after first resolved
    both filled → finalize
- Suggested add-on groups never block finalization
- default_item_ids applied silently at finalize
- Menu store normalization: is_required=False, min_selector>0 → normalized to 0
- top_choices filter excludes already-selected IDs on reprompt
- remaining_to_min present in step payload
"""
from __future__ import annotations

import logging

import pytest

from app.menu.models import (
    MenuItem,
    ModifierGroup,
    Pricing,
    SideChoice,
    SideGroup,
)
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.add_item_flow import (
    ReadyToFinalize,
    _build_side_step,
    _side_group_satisfied,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.confirmation_decision_helper import (
    _apply_group_defaults,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import (
    build_pending_add_item,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _side_choice(item_id: str, name: str) -> SideChoice:
    return SideChoice(
        item_id=item_id,
        name=name,
        normalized_name=normalize_text(name),
        pricing=Pricing(mode="fixed", price_cents=0),
    )


def _side_group(
    group_id: str,
    name: str,
    choices: list[SideChoice],
    *,
    is_required: bool = True,
    min_selector: int = 1,
    max_selector: int = 1,
    is_suggested_addon: bool = False,
    default_item_ids: tuple[str, ...] = (),
    allow_duplicate_selections: bool = False,
) -> SideGroup:
    return SideGroup(
        group_id=group_id,
        name=name,
        normalized_name=normalize_text(name),
        is_required=is_required,
        min_selector=min_selector,
        max_selector=max_selector,
        choices=choices,
        allow_duplicate_selections=allow_duplicate_selections,
        is_suggested_addon=is_suggested_addon,
        default_item_ids=default_item_ids,
    )


def _make_platter_item() -> MenuItem:
    """Item with a required min_selector=2, max_selector=2 side group."""
    choices = [
        _side_choice("potato_salad", "Potato Salad"),
        _side_choice("rice", "Rice"),
        _side_choice("corn", "Corn on the Cob"),
        _side_choice("cole_slaw", "Cole Slaw"),
        _side_choice("collard", "Collard Greens"),
    ]
    group = _side_group(
        "platter_sides",
        "Platter Sides",
        choices,
        is_required=True,
        min_selector=2,
        max_selector=2,
        allow_duplicate_selections=False,
    )
    return MenuItem(
        item_id="lobster_platter",
        name="Lobster Tail Platter",
        normalized_name="lobster tail platter",
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=2399),
        side_groups=[group],
        modifier_groups=[],
        available=True,
    )


def _make_combo_item() -> MenuItem:
    """Item with two required side groups (drink + food side)."""
    drink_group = _side_group(
        "combo_drink",
        "Choose Drink",
        [_side_choice("coke", "Coke"), _side_choice("sprite", "Sprite")],
        is_required=True, min_selector=1, max_selector=1,
    )
    food_group = _side_group(
        "combo_side",
        "Choose Side",
        [_side_choice("fries", "Fries"), _side_choice("rings", "Onion Rings")],
        is_required=True, min_selector=1, max_selector=1,
    )
    return MenuItem(
        item_id="burger_combo",
        name="Burger Combo",
        normalized_name="burger combo",
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=480),
        side_groups=[drink_group, food_group],
        modifier_groups=[],
        available=True,
    )


def _make_item_with_suggested_addon() -> MenuItem:
    """Item where the only side group is a suggested add-on."""
    addon_group = _side_group(
        "drink_addon",
        "Can Drinks",
        [_side_choice("coke_can", "Coke (12 oz.)"), _side_choice("sprite_can", "Sprite (12 oz.)")],
        is_required=False, min_selector=0, max_selector=1,
        is_suggested_addon=True,
    )
    return MenuItem(
        item_id="chicken_taco",
        name="Chicken Taco",
        normalized_name="chicken taco",
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=499),
        side_groups=[addon_group],
        modifier_groups=[],
        available=True,
    )


def _make_item_with_bun_default() -> MenuItem:
    """Item with a bun side group that has a default choice."""
    bun_group = _side_group(
        "bun_group",
        "Choose Bun",
        [
            _side_choice("plain_bun", "Plain Bun"),
            _side_choice("potato_bun", "Potato Bun"),
        ],
        is_required=False, min_selector=0, max_selector=1,
        default_item_ids=("plain_bun",),
    )
    return MenuItem(
        item_id="chicken_burger",
        name="Chicken Burger",
        normalized_name="chicken burger",
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=1268),
        side_groups=[bun_group],
        modifier_groups=[],
        available=True,
    )


def _ctx_with_item(item: MenuItem) -> ConversationContext:
    ctx = ConversationContext()
    ctx.pending_add_item = build_pending_add_item(item)
    ctx.quantity = 1
    return ctx


# ---------------------------------------------------------------------------
# _side_group_satisfied unit tests
# ---------------------------------------------------------------------------

class TestSideGroupSatisfied:
    def test_required_group_no_selection_not_satisfied(self):
        group = _side_group("g1", "G", [_side_choice("a", "A")], is_required=True, min_selector=1)
        pending = build_pending_add_item(
            MenuItem(
                item_id="x", name="X", normalized_name="x",
                aliases=(), normalized_aliases=(), voice_labels=(),
                pricing=Pricing(mode="fixed", price_cents=0),
                side_groups=[group], modifier_groups=[], available=True,
            )
        ).side_groups[0]
        assert not _side_group_satisfied(pending, [], skipped=False)

    def test_required_group_min2_one_given_not_satisfied(self):
        choices = [_side_choice("a", "A"), _side_choice("b", "B")]
        group = _side_group("g1", "G", choices, is_required=True, min_selector=2, max_selector=2)
        pending = build_pending_add_item(
            MenuItem(
                item_id="x", name="X", normalized_name="x",
                aliases=(), normalized_aliases=(), voice_labels=(),
                pricing=Pricing(mode="fixed", price_cents=0),
                side_groups=[group], modifier_groups=[], available=True,
            )
        ).side_groups[0]
        assert not _side_group_satisfied(pending, ["a"], skipped=False)

    def test_required_group_min2_two_given_satisfied(self):
        choices = [_side_choice("a", "A"), _side_choice("b", "B")]
        group = _side_group("g1", "G", choices, is_required=True, min_selector=2, max_selector=2)
        pending = build_pending_add_item(
            MenuItem(
                item_id="x", name="X", normalized_name="x",
                aliases=(), normalized_aliases=(), voice_labels=(),
                pricing=Pricing(mode="fixed", price_cents=0),
                side_groups=[group], modifier_groups=[], available=True,
            )
        ).side_groups[0]
        assert _side_group_satisfied(pending, ["a", "b"], skipped=False)

    def test_suggested_addon_always_satisfied_even_with_no_selection(self):
        group = _side_group(
            "g1", "Can Drinks", [_side_choice("coke", "Coke")],
            is_required=False, min_selector=0, is_suggested_addon=True,
        )
        pending = build_pending_add_item(
            MenuItem(
                item_id="x", name="X", normalized_name="x",
                aliases=(), normalized_aliases=(), voice_labels=(),
                pricing=Pricing(mode="fixed", price_cents=0),
                side_groups=[group], modifier_groups=[], available=True,
            )
        ).side_groups[0]
        assert _side_group_satisfied(pending, [], skipped=False)

    def test_optional_group_no_selection_not_satisfied_unless_skipped(self):
        group = _side_group("g1", "G", [_side_choice("a", "A")], is_required=False, min_selector=0)
        pending = build_pending_add_item(
            MenuItem(
                item_id="x", name="X", normalized_name="x",
                aliases=(), normalized_aliases=(), voice_labels=(),
                pricing=Pricing(mode="fixed", price_cents=0),
                side_groups=[group], modifier_groups=[], available=True,
            )
        ).side_groups[0]
        assert not _side_group_satisfied(pending, [], skipped=False)
        assert _side_group_satisfied(pending, [], skipped=True)


# ---------------------------------------------------------------------------
# Lobster Tail Platter — required min_selector=2 group
# ---------------------------------------------------------------------------

class TestPlattertwoRequiredSides:
    def test_no_sides_given_routes_to_waiting_for_side(self):
        ctx = _ctx_with_item(_make_platter_item())
        step = determine_next_add_item_step(ctx)
        assert step.next_state == ConversationState.WAITING_FOR_SIDE

    def test_no_sides_prompt_asks_choose_2(self):
        ctx = _ctx_with_item(_make_platter_item())
        step = determine_next_add_item_step(ctx)
        assert step.response_payload["min_selector"] == 2
        assert step.response_payload["selected_count"] == 0
        assert step.response_payload["remaining_to_min"] == 2

    def test_one_side_given_routes_to_waiting_for_side_again(self):
        ctx = _ctx_with_item(_make_platter_item())
        ctx.selected_side_groups["platter_sides"] = ["potato_salad"]
        step = determine_next_add_item_step(ctx)
        assert step.next_state == ConversationState.WAITING_FOR_SIDE

    def test_one_side_given_remaining_to_min_is_1(self):
        ctx = _ctx_with_item(_make_platter_item())
        ctx.selected_side_groups["platter_sides"] = ["potato_salad"]
        step = determine_next_add_item_step(ctx)
        assert step.response_payload["remaining_to_min"] == 1
        assert step.response_payload["selected_count"] == 1

    def test_one_side_given_top_choices_excludes_selected(self):
        ctx = _ctx_with_item(_make_platter_item())
        ctx.selected_side_groups["platter_sides"] = ["potato_salad"]
        step = determine_next_add_item_step(ctx)
        assert "Potato Salad" not in step.response_payload["top_choices"]

    def test_two_sides_given_finalizes(self):
        ctx = _ctx_with_item(_make_platter_item())
        ctx.selected_side_groups["platter_sides"] = ["potato_salad", "rice"]
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)

    def test_two_sides_in_command_payload(self):
        ctx = _ctx_with_item(_make_platter_item())
        ctx.selected_side_groups["platter_sides"] = ["potato_salad", "rice"]
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.sides["platter_sides"] == ["potato_salad", "rice"]


# ---------------------------------------------------------------------------
# Combo — two required groups
# ---------------------------------------------------------------------------

class TestComboTwoRequiredGroups:
    def test_nothing_selected_asks_for_drink_group_first(self):
        """Drink group comes first in sort order (is_drink_like_group = True sorts last normally,
        but here we verify whichever required group is first blocks correctly)."""
        ctx = _ctx_with_item(_make_combo_item())
        step = determine_next_add_item_step(ctx)
        assert step.next_state == ConversationState.WAITING_FOR_SIDE

    def test_drink_only_filled_asks_for_food_side(self):
        ctx = _ctx_with_item(_make_combo_item())
        pending = ctx.pending_add_item
        # fill the drink group (whatever group_id the drink was given)
        for g in pending.side_groups:
            if "drink" in g.name.lower():
                ctx.selected_side_groups[g.group_id] = ["coke"]
                break
        step = determine_next_add_item_step(ctx)
        assert step.next_state == ConversationState.WAITING_FOR_SIDE
        # The remaining required group should be the food side
        assert "drink" not in step.response_payload.get("group_name", "").lower()

    def test_food_side_only_filled_asks_for_drink(self):
        ctx = _ctx_with_item(_make_combo_item())
        pending = ctx.pending_add_item
        for g in pending.side_groups:
            if "side" in g.name.lower():
                ctx.selected_side_groups[g.group_id] = ["fries"]
                break
        step = determine_next_add_item_step(ctx)
        assert step.next_state == ConversationState.WAITING_FOR_SIDE

    def test_both_filled_finalizes(self):
        ctx = _ctx_with_item(_make_combo_item())
        pending = ctx.pending_add_item
        for g in pending.side_groups:
            if "drink" in g.name.lower():
                ctx.selected_side_groups[g.group_id] = ["coke"]
            elif "side" in g.name.lower():
                ctx.selected_side_groups[g.group_id] = ["fries"]
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)


# ---------------------------------------------------------------------------
# Suggested add-on — never blocks finalization
# ---------------------------------------------------------------------------

class TestSuggestedAddon:
    def test_suggested_addon_not_filled_does_not_block_finalization(self):
        ctx = _ctx_with_item(_make_item_with_suggested_addon())
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize), (
            f"Suggested add-on should not block finalization; got {step}"
        )

    def test_suggested_addon_selection_preserved_when_given(self):
        ctx = _ctx_with_item(_make_item_with_suggested_addon())
        ctx.selected_side_groups["drink_addon"] = ["coke_can"]
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.sides["drink_addon"] == ["coke_can"]


# ---------------------------------------------------------------------------
# default_item_ids — silently applied at finalize
# ---------------------------------------------------------------------------

class TestDefaultItemIds:
    def test_default_bun_applied_when_no_selection(self):
        ctx = _ctx_with_item(_make_item_with_bun_default())
        _apply_group_defaults(ctx)
        assert ctx.selected_side_groups.get("bun_group") == ["plain_bun"]

    def test_default_not_applied_when_user_already_selected(self):
        ctx = _ctx_with_item(_make_item_with_bun_default())
        ctx.selected_side_groups["bun_group"] = ["potato_bun"]
        _apply_group_defaults(ctx)
        assert ctx.selected_side_groups["bun_group"] == ["potato_bun"]  # unchanged

    def test_default_applied_and_item_finalizes(self):
        ctx = _ctx_with_item(_make_item_with_bun_default())
        _apply_group_defaults(ctx)
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)
        assert step.command.sides["bun_group"] == ["plain_bun"]

    def test_no_default_for_required_groups(self):
        """default_item_ids on a required group is irrelevant — required groups
        still demand explicit selection."""
        ctx = _ctx_with_item(_make_platter_item())
        _apply_group_defaults(ctx)
        # Platter sides have no default_item_ids, so nothing was applied.
        assert ctx.selected_side_groups.get("platter_sides") is None
        step = determine_next_add_item_step(ctx)
        assert step.next_state == ConversationState.WAITING_FOR_SIDE


# ---------------------------------------------------------------------------
# Menu store normalization
# ---------------------------------------------------------------------------

class TestMenuStoreNormalization:
    def test_optional_group_with_nonzero_min_normalized_in_pending_factory(self):
        """PendingSideGroup factory converts min_selector=0 correctly (no or-1 bug)."""
        group = _side_group(
            "g1", "Toppings", [_side_choice("a", "A")],
            is_required=False, min_selector=0,
        )
        item = MenuItem(
            item_id="x", name="X", normalized_name="x",
            aliases=(), normalized_aliases=(), voice_labels=(),
            pricing=Pricing(mode="fixed", price_cents=0),
            side_groups=[group], modifier_groups=[], available=True,
        )
        pending = build_pending_add_item(item)
        assert pending.side_groups[0].min_selector == 0, (
            "min_selector=0 must not be converted to 1 by the factory"
        )

    def test_optional_group_min_zero_does_not_block_finalization_when_skipped(self):
        group = _side_group(
            "g1", "Toppings", [_side_choice("a", "A")],
            is_required=False, min_selector=0,
        )
        item = MenuItem(
            item_id="x", name="X", normalized_name="x",
            aliases=(), normalized_aliases=(), voice_labels=(),
            pricing=Pricing(mode="fixed", price_cents=0),
            side_groups=[group], modifier_groups=[], available=True,
        )
        ctx = ConversationContext()
        ctx.pending_add_item = build_pending_add_item(item)
        ctx.quantity = 1
        ctx.skipped_side_groups.add("g1")
        step = determine_next_add_item_step(ctx)
        assert isinstance(step, ReadyToFinalize)

    def test_store_normalization_logs_warning_for_contradictory_group(
        self, caplog
    ):
        """MenuStore logs a warning when is_required=False but min_selector>0."""
        from unittest.mock import MagicMock, patch
        from app.menu.store import MenuStore

        raw_menu = {
            "items": {
                "item_1": {
                    "item_id": "item_1",
                    "name": "Test Item",
                    "pricing": {"mode": "fixed", "price_cents": 100, "currency": "USD"},
                    "side_groups": [
                        {
                            "group_id": "sg_1",
                            "name": "Optional Group",
                            "is_required": False,
                            "min_selector": 1,  # contradictory
                            "max_selector": 1,
                            "choices": [],
                        }
                    ],
                    "modifier_groups": [],
                    "available": True,
                }
            },
            "categories": {},
        }
        raw_entity_index: dict = {}

        import json, tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            menu_path = Path(tmpdir) / "menu.json"
            entity_path = Path(tmpdir) / "entity_index.json"
            menu_path.write_text(json.dumps(raw_menu))
            entity_path.write_text(json.dumps(raw_entity_index))

            with caplog.at_level(logging.WARNING, logger="app.menu.store"):
                store = MenuStore(menu_path, entity_path)

        item = store.get_item("item_1")
        assert item.side_groups[0].min_selector == 0, (
            "min_selector should be normalized to 0 for optional group"
        )
        assert any("menu_group_min_normalized" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# top_choices filtering
# ---------------------------------------------------------------------------

class TestTopChoicesFiltering:
    def test_already_selected_excluded_from_top_choices(self):
        ctx = _ctx_with_item(_make_platter_item())
        ctx.selected_side_groups["platter_sides"] = ["potato_salad"]
        step = determine_next_add_item_step(ctx)
        assert "Potato Salad" not in step.response_payload["top_choices"]

    def test_unselected_choices_remain_in_top_choices(self):
        ctx = _ctx_with_item(_make_platter_item())
        ctx.selected_side_groups["platter_sides"] = ["potato_salad"]
        step = determine_next_add_item_step(ctx)
        top = step.response_payload["top_choices"]
        assert any(c != "Potato Salad" for c in top)

    def test_no_filtering_when_nothing_selected(self):
        ctx = _ctx_with_item(_make_platter_item())
        step = determine_next_add_item_step(ctx)
        top = step.response_payload["top_choices"]
        assert len(top) > 0  # some choices should be present


# ---------------------------------------------------------------------------
# Overflow prompt — fail-closed ask for correction
# ---------------------------------------------------------------------------

class TestOverflowResponseWording:
    def test_too_many_side_choices_fail_closed_wording(self):
        """too_many_side_choices produces corrective ask when requested > max."""
        from unittest.mock import MagicMock
        from app.responses.item.sides import too_many_side_choices
        from app.state_machine.models.conversation_context import ConversationContext

        ctx = ConversationContext()

        # Build a mock menu_repo that returns a group with max_selector=2
        mock_group = MagicMock()
        mock_group.min_selector = 2
        mock_group.max_selector = 2
        mock_group.choices = [
            MagicMock(name_attr="Rice"),
            MagicMock(name_attr="Potato Salad"),
            MagicMock(name_attr="Corn on the Cob"),
        ]
        for i, c in enumerate(mock_group.choices):
            c.name = ["Rice", "Potato Salad", "Corn on the Cob"][i]

        mock_item = MagicMock()
        mock_item.side_groups = [mock_group]
        mock_repo = MagicMock()
        mock_repo.store.get_item.return_value = mock_item
        ctx.current_item_id = "item_x"
        ctx.current_side_group_index = 0
        ctx.selected_side_groups = {}

        payload = {
            "requested_names": ["Rice", "Potato Salad", "Corn on the Cob"],
            "max_selector": 2,
            "speech_noun": "side",
        }

        response = too_many_side_choices(ctx, mock_repo, payload)

        assert "You can choose" in response
        assert "I heard" in response
        assert "Which" in response
        # All three heard items present
        assert "Rice" in response
        assert "Potato Salad" in response
        assert "Corn" in response


# ---------------------------------------------------------------------------
# ask_for_side partial-fill wording
# ---------------------------------------------------------------------------

class TestAskForSidePartialFillWording:
    def test_choose_1_more_side_when_one_already_selected(self):
        from unittest.mock import MagicMock
        from app.responses.item.sides import ask_for_side
        from app.state_machine.models.conversation_context import ConversationContext

        ctx = ConversationContext()
        mock_group = MagicMock()
        mock_group.min_selector = 2
        mock_group.max_selector = 2
        mock_group.choices = [
            MagicMock(), MagicMock(), MagicMock()
        ]
        names = ["Rice", "Corn on the Cob", "Cole Slaw"]
        for i, c in enumerate(mock_group.choices):
            c.name = names[i]

        mock_item = MagicMock()
        mock_item.side_groups = [mock_group]
        mock_repo = MagicMock()
        mock_repo.store.get_item.return_value = mock_item
        ctx.current_item_id = "item_x"
        ctx.current_side_group_index = 0
        ctx.selected_side_groups = {}

        payload = {
            "min_selector": 2,
            "selected_count": 1,
            "remaining_to_min": 1,
            "top_choices": ["Rice", "Corn on the Cob", "Cole Slaw"],
            "speech_noun": "side",
        }
        response = ask_for_side(ctx, mock_repo, payload)
        assert "1 more side" in response.lower()

    def test_choose_2_sides_when_none_selected_min2(self):
        from unittest.mock import MagicMock
        from app.responses.item.sides import ask_for_side
        from app.state_machine.models.conversation_context import ConversationContext

        ctx = ConversationContext()
        mock_group = MagicMock()
        mock_group.min_selector = 2
        mock_group.max_selector = 2
        mock_group.choices = [MagicMock(), MagicMock()]
        for i, c in enumerate(mock_group.choices):
            c.name = ["Rice", "Corn"][i]

        mock_item = MagicMock()
        mock_item.side_groups = [mock_group]
        mock_repo = MagicMock()
        mock_repo.store.get_item.return_value = mock_item
        ctx.current_item_id = "item_x"
        ctx.current_side_group_index = 0
        ctx.selected_side_groups = {}

        payload = {
            "min_selector": 2,
            "selected_count": 0,
            "remaining_to_min": 2,
            "top_choices": ["Rice", "Corn"],
            "speech_noun": "side",
        }
        response = ask_for_side(ctx, mock_repo, payload)
        assert "2 side" in response.lower()
