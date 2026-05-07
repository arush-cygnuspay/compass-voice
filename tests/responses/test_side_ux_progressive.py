# tests/responses/test_side_ux_progressive.py
"""Phase 4, 5, 6 — _format_options, build_side_prompt_lead, ask_for_side."""
from __future__ import annotations

import types

import pytest

from app.responses.item.format_utils import _format_options
from app.responses.item.sides import ask_for_side, build_side_prompt_lead


# ---------------------------------------------------------------------------
# D. _format_options — no "and N more"
# ---------------------------------------------------------------------------

class TestFormatOptions:
    def test_empty_list_returns_empty_string(self):
        assert _format_options([]) == ""

    def test_single_item(self):
        assert _format_options(["A"]) == "A"

    def test_two_items(self):
        assert _format_options(["A", "B"]) == "A or B"

    def test_three_items(self):
        assert _format_options(["A", "B", "C"]) == "A, B, or C"

    def test_four_items_within_default_max(self):
        result = _format_options(["A", "B", "C", "D"])
        assert result == "A, B, C, or D"
        assert "more" not in result

    def test_six_items_all_shown(self):
        opts = ["A", "B", "C", "D", "E", "F"]
        result = _format_options(opts, max_items=6)
        assert result == "A, B, C, D, E, or F"
        assert "more" not in result

    def test_seven_items_truncates_silently(self):
        opts = ["A", "B", "C", "D", "E", "F", "G"]
        result = _format_options(opts, max_items=6)
        assert "G" not in result
        assert "more" not in result
        assert result.endswith("or F")

    def test_seven_items_with_overflow_hint(self):
        opts = ["A", "B", "C", "D", "E", "F", "G"]
        result = _format_options(opts, max_items=6, overflow_hint="or say 'options' to hear them all")
        assert "G" not in result
        assert "or say 'options' to hear them all" in result
        assert "more" not in result

    def test_no_overflow_hint_when_within_max(self):
        opts = ["A", "B", "C"]
        result = _format_options(opts, max_items=6, overflow_hint="or say 'options' to hear them all")
        assert "or say 'options' to hear them all" not in result

    def test_and_n_more_never_emitted(self):
        for count in range(1, 12):
            opts = [f"Item {i}" for i in range(count)]
            result = _format_options(opts)
            assert "more" not in result, f"'more' found in result for {count} items: {result!r}"


# ---------------------------------------------------------------------------
# E. build_side_prompt_lead — progressive/ordinal wording
# ---------------------------------------------------------------------------

def _lead(position, total, is_drink=False, speech_noun=None):
    noun = speech_noun or ("drink" if is_drink else "side")
    is_last = position == total - 1
    return build_side_prompt_lead(
        position=position,
        total=total,
        is_drink_group=is_drink,
        is_last_side_prompt=is_last,
        speech_noun=noun,
    )


class TestBuildSidePromptLead:
    # Case A — single group
    def test_single_drink_group(self):
        result = _lead(0, 1, is_drink=True)
        assert result == "Which drink would you like?"
        assert "lastly" not in result.lower()

    def test_single_food_group(self):
        result = _lead(0, 1, is_drink=False)
        assert result == "Which side would you like?"

    # Case B — two groups, food then drink
    def test_two_groups_first_is_food(self):
        result = _lead(0, 2, is_drink=False)
        assert result == "Choose your side."

    def test_two_groups_second_is_drink(self):
        result = _lead(1, 2, is_drink=True)
        assert result == "Lastly, choose your drink."

    # Case C — three groups, food food drink
    def test_three_groups_first_is_food(self):
        result = _lead(0, 3, is_drink=False)
        assert result == "Choose your first side."

    def test_three_groups_second_is_food(self):
        result = _lead(1, 3, is_drink=False)
        assert result == "Now choose your second side."

    def test_three_groups_third_is_drink(self):
        result = _lead(2, 3, is_drink=True)
        assert result == "Lastly, choose your drink."

    # Case D — non-drink final group
    def test_two_food_groups_second_non_drink_final(self):
        result = _lead(1, 2, is_drink=False)
        assert "second" in result
        assert "lastly" not in result.lower()

    # Speech noun propagation
    def test_can_drinks_speaks_as_drink(self):
        # "Can Drinks" → speech_noun="drink"
        result = build_side_prompt_lead(
            position=1, total=2,
            is_drink_group=True, is_last_side_prompt=True,
            speech_noun="drink",
        )
        assert "drink" in result
        assert "Can Drinks" not in result
        assert "can drinks" not in result.lower()

    def test_family_meal_drinks_speaks_as_drink(self):
        result = build_side_prompt_lead(
            position=0, total=1,
            is_drink_group=True, is_last_side_prompt=True,
            speech_noun="drink",
        )
        assert "drink" in result
        assert "Family Meal Drinks" not in result

    def test_no_lastly_for_single_group(self):
        for is_drink in (True, False):
            result = _lead(0, 1, is_drink=is_drink)
            assert "lastly" not in result.lower()


# ---------------------------------------------------------------------------
# ask_for_side integration (uses payload metadata)
# ---------------------------------------------------------------------------

def _make_context(group_index=0):
    return types.SimpleNamespace(
        current_item_id="item_1",
        current_side_group_index=group_index,
        selected_side_groups={},
        selected_modifier_groups={},
    )


def _make_simple_menu_repo(group_name="Choose Side", min_sel=0, max_sel=1, choices=None):
    choices = choices or [
        types.SimpleNamespace(item_id="c1", name="Potato Salad", pricing_mode="fixed", variants=[], variant_names=[]),
        types.SimpleNamespace(item_id="c2", name="Corn on the Cob", pricing_mode="fixed", variants=[], variant_names=[]),
        types.SimpleNamespace(item_id="c3", name="Cole Slaw", pricing_mode="fixed", variants=[], variant_names=[]),
    ]
    g = types.SimpleNamespace(
        group_id="grp",
        name=group_name,
        choices=choices,
        choices_by_item_id={c.item_id: c for c in choices},
        min_selector=min_sel,
        max_selector=max_sel,
    )
    item = types.SimpleNamespace(name="Chicken", side_groups=[g], modifier_groups=[])
    store = types.SimpleNamespace(get_item=lambda _: item)
    return types.SimpleNamespace(store=store)


class TestAskForSide:
    def test_single_food_group_no_lastly(self):
        ctx = _make_context()
        repo = _make_simple_menu_repo()
        payload = {
            "top_choices": ["Potato Salad", "Corn on the Cob", "Cole Slaw"],
            "min_selector": 0, "max_selector": 1,
            "side_group_position": 0, "total_side_groups": 1,
            "is_drink_group": False, "is_last_side_prompt": True,
            "speech_noun": "side",
        }
        result = ask_for_side(ctx, repo, payload)
        assert "lastly" not in result.lower()
        assert "Potato Salad" in result or "Corn" in result

    def test_two_groups_first_says_choose_your_side(self):
        ctx = _make_context()
        repo = _make_simple_menu_repo()
        payload = {
            "top_choices": ["Potato Salad", "Corn on the Cob"],
            "min_selector": 1, "max_selector": 1,
            "side_group_position": 0, "total_side_groups": 2,
            "is_drink_group": False, "is_last_side_prompt": False,
            "speech_noun": "side",
        }
        result = ask_for_side(ctx, repo, payload)
        assert result.startswith("Choose your side.")

    def test_two_groups_second_is_drink_says_lastly(self):
        ctx = _make_context(group_index=1)
        repo = _make_simple_menu_repo("Can Drinks", min_sel=0)
        payload = {
            "top_choices": ["Coke", "Sprite", "Water"],
            "min_selector": 0, "max_selector": 1,
            "side_group_position": 1, "total_side_groups": 2,
            "is_drink_group": True, "is_last_side_prompt": True,
            "speech_noun": "drink",
        }
        result = ask_for_side(ctx, repo, payload)
        assert "Lastly, choose your drink." in result
        assert "Coke" in result or "Sprite" in result

    def test_three_groups_first_says_choose_your_first_side(self):
        ctx = _make_context()
        repo = _make_simple_menu_repo()
        payload = {
            "top_choices": ["A", "B"],
            "min_selector": 1,
            "side_group_position": 0, "total_side_groups": 3,
            "is_drink_group": False, "is_last_side_prompt": False,
            "speech_noun": "side",
        }
        result = ask_for_side(ctx, repo, payload)
        assert result.startswith("Choose your first side.")

    def test_six_options_all_listed(self):
        ctx = _make_context()
        repo = _make_simple_menu_repo()
        six = [f"Option {i}" for i in range(6)]
        payload = {
            "top_choices": six,
            "min_selector": 1,
            "side_group_position": 0, "total_side_groups": 1,
            "is_drink_group": False, "is_last_side_prompt": True,
            "speech_noun": "side",
        }
        result = ask_for_side(ctx, repo, payload)
        for opt in six:
            assert opt in result
        assert "more" not in result

    def test_seven_options_uses_overflow_hint(self):
        ctx = _make_context()
        repo = _make_simple_menu_repo()
        seven = [f"Option {i}" for i in range(7)]
        payload = {
            "top_choices": seven,
            "min_selector": 1,
            "side_group_position": 0, "total_side_groups": 1,
            "is_drink_group": False, "is_last_side_prompt": True,
            "speech_noun": "side",
        }
        result = ask_for_side(ctx, repo, payload)
        assert "Option 6" not in result  # 7th item truncated
        assert "options" in result.lower()  # overflow hint
        assert "more" not in result

    def test_no_and_n_more_in_side_prompt(self):
        ctx = _make_context()
        repo = _make_simple_menu_repo()
        many = [f"Choice {i}" for i in range(10)]
        payload = {
            "top_choices": many,
            "min_selector": 1,
            "side_group_position": 0, "total_side_groups": 1,
            "is_drink_group": False, "is_last_side_prompt": True,
            "speech_noun": "side",
        }
        result = ask_for_side(ctx, repo, payload)
        assert "and 1 more" not in result
        assert "and one more" not in result
        assert "and 2 more" not in result
        assert "and 4 more" not in result

    def test_optional_group_says_you_can_say_none(self):
        ctx = _make_context()
        repo = _make_simple_menu_repo(min_sel=0)
        payload = {
            "top_choices": ["A", "B"],
            "min_selector": 0,
            "side_group_position": 0, "total_side_groups": 1,
            "is_drink_group": False, "is_last_side_prompt": True,
            "speech_noun": "side",
        }
        result = ask_for_side(ctx, repo, payload)
        assert "You can say none" in result

    def test_required_group_no_you_can_say_none(self):
        ctx = _make_context()
        repo = _make_simple_menu_repo(min_sel=1)
        payload = {
            "top_choices": ["A", "B"],
            "min_selector": 1,
            "side_group_position": 0, "total_side_groups": 1,
            "is_drink_group": False, "is_last_side_prompt": True,
            "speech_noun": "side",
        }
        result = ask_for_side(ctx, repo, payload)
        assert "You can say none" not in result
