# tests/responses/test_modifier_ux_progressive.py
"""Phase F — build_modifier_prompt_lead and ask_for_modifier progressive wording."""
from __future__ import annotations

import types

import pytest

from app.responses.item.format_utils import _format_options
from app.responses.item.modifiers import ask_for_modifier, build_modifier_prompt_lead


# ---------------------------------------------------------------------------
# build_modifier_prompt_lead — progressive/ordinal wording
# ---------------------------------------------------------------------------

def _lead(position: int, total: int, speech_noun: str = "add-on") -> str:
    is_last = position == total - 1
    return build_modifier_prompt_lead(
        position=position,
        total=total,
        is_last=is_last,
        speech_noun=speech_noun,
    )


class TestBuildModifierPromptLead:
    # Single group
    def test_single_group_which_noun(self):
        assert _lead(0, 1, "sauce") == "Which sauce would you like?"

    def test_single_group_add_on_default(self):
        assert _lead(0, 1) == "Which add-on would you like?"

    # Two groups — first
    def test_two_groups_first_choose_your(self):
        result = _lead(0, 2, "topping")
        assert result == "Choose your topping."

    # Two groups — last
    def test_two_groups_last_says_lastly(self):
        result = _lead(1, 2, "topping")
        assert "Lastly" in result
        assert "topping" in result

    # Three groups — first
    def test_three_groups_first_says_first(self):
        result = _lead(0, 3, "sauce")
        assert result == "Choose your first sauce."

    # Three groups — middle
    def test_three_groups_middle_says_second(self):
        result = _lead(1, 3, "sauce")
        assert "second" in result
        assert "sauce" in result

    # Three groups — last
    def test_three_groups_last_says_lastly(self):
        result = _lead(2, 3, "sauce")
        assert result.startswith("Lastly")
        assert "sauce" in result

    # Speech noun — protein
    def test_protein_noun_single(self):
        result = _lead(0, 1, "protein")
        assert result == "Which protein would you like?"

    # Speech noun — cheese
    def test_cheese_noun_two_groups_first(self):
        result = _lead(0, 2, "cheese")
        assert result == "Choose your cheese."

    # No "lastly" for single
    def test_no_lastly_for_single_group(self):
        result = _lead(0, 1, "sauce")
        assert "lastly" not in result.lower()

    # Explicit is_last=False overrides position==total-1 default
    def test_explicit_is_last_false_on_last_position(self):
        result = build_modifier_prompt_lead(
            position=2, total=3, is_last=False, speech_noun="topping"
        )
        assert "Lastly" not in result


# ---------------------------------------------------------------------------
# ask_for_modifier integration — uses payload metadata
# ---------------------------------------------------------------------------

def _make_context():
    return types.SimpleNamespace(
        current_item_id="item_1",
        current_modifier_group_index=0,
        selected_modifier_groups={},
        selected_side_groups={},
    )


def _make_menu_repo(group_name="Choose Sauce", min_sel=1, max_sel=1, choices=None):
    choices = choices or [
        types.SimpleNamespace(modifier_id="ranch", name="Ranch"),
        types.SimpleNamespace(modifier_id="bbq", name="BBQ"),
        types.SimpleNamespace(modifier_id="honey_mustard", name="Honey Mustard"),
    ]
    g = types.SimpleNamespace(
        group_id="grp",
        name=group_name,
        choices=choices,
        choices_by_modifier_id={c.modifier_id: c for c in choices},
        min_selector=min_sel,
        max_selector=max_sel,
    )
    item = types.SimpleNamespace(name="Wings", modifier_groups=[g], side_groups=[])
    store = types.SimpleNamespace(get_item=lambda _: item)
    return types.SimpleNamespace(store=store)


class TestAskForModifier:
    def test_single_group_which_sauce(self):
        ctx = _make_context()
        repo = _make_menu_repo()
        payload = {
            "top_choices": ["Ranch", "BBQ", "Honey Mustard"],
            "total_choices": 3,
            "min_selector": 1, "max_selector": 1,
            "modifier_group_position": 0, "total_modifier_groups": 1,
            "is_last_modifier_prompt": True,
            "speech_noun": "sauce",
        }
        result = ask_for_modifier(ctx, repo, payload)
        assert result.startswith("Which sauce would you like?")
        assert "Ranch" in result or "BBQ" in result

    def test_two_groups_first_choose_your_topping(self):
        ctx = _make_context()
        repo = _make_menu_repo()
        payload = {
            "top_choices": ["Ranch", "BBQ"],
            "total_choices": 2,
            "min_selector": 1, "max_selector": 1,
            "modifier_group_position": 0, "total_modifier_groups": 2,
            "is_last_modifier_prompt": False,
            "speech_noun": "topping",
        }
        result = ask_for_modifier(ctx, repo, payload)
        assert result.startswith("Choose your topping.")

    def test_two_groups_last_says_lastly(self):
        ctx = _make_context()
        repo = _make_menu_repo()
        payload = {
            "top_choices": ["Ranch", "BBQ"],
            "total_choices": 2,
            "min_selector": 0, "max_selector": 1,
            "modifier_group_position": 1, "total_modifier_groups": 2,
            "is_last_modifier_prompt": True,
            "speech_noun": "sauce",
        }
        result = ask_for_modifier(ctx, repo, payload)
        assert "Lastly" in result
        assert "sauce" in result

    def test_optional_group_says_you_can_say_none(self):
        ctx = _make_context()
        repo = _make_menu_repo(min_sel=0)
        payload = {
            "top_choices": ["Ranch", "BBQ"],
            "total_choices": 2,
            "min_selector": 0, "max_selector": 1,
            "modifier_group_position": 0, "total_modifier_groups": 1,
            "is_last_modifier_prompt": True,
            "speech_noun": "sauce",
        }
        result = ask_for_modifier(ctx, repo, payload)
        assert "You can say none" in result

    def test_required_group_no_you_can_say_none(self):
        ctx = _make_context()
        repo = _make_menu_repo(min_sel=1)
        payload = {
            "top_choices": ["Ranch", "BBQ"],
            "total_choices": 2,
            "min_selector": 1, "max_selector": 1,
            "modifier_group_position": 0, "total_modifier_groups": 1,
            "is_last_modifier_prompt": True,
            "speech_noun": "sauce",
        }
        result = ask_for_modifier(ctx, repo, payload)
        assert "You can say none" not in result

    def test_six_options_all_listed_no_overflow_hint(self):
        ctx = _make_context()
        repo = _make_menu_repo()
        six = [f"Option {i}" for i in range(6)]
        payload = {
            "top_choices": six,
            "total_choices": 6,
            "min_selector": 1, "max_selector": 1,
            "modifier_group_position": 0, "total_modifier_groups": 1,
            "is_last_modifier_prompt": True,
            "speech_noun": "add-on",
        }
        result = ask_for_modifier(ctx, repo, payload)
        for opt in six:
            assert opt in result
        assert "options" not in result.lower() or "which" in result.lower()
        assert "more" not in result

    def test_seven_total_choices_triggers_overflow_hint(self):
        """Real production flow: top_choices pre-capped at 6, total_choices=7 → hint fires."""
        ctx = _make_context()
        repo = _make_menu_repo()
        top_six = [f"Option {i}" for i in range(6)]
        payload = {
            "top_choices": top_six,
            "total_choices": 7,  # 7th item not in top_choices (pre-capped at 6)
            "min_selector": 1, "max_selector": 1,
            "modifier_group_position": 0, "total_modifier_groups": 1,
            "is_last_modifier_prompt": True,
            "speech_noun": "add-on",
        }
        result = ask_for_modifier(ctx, repo, payload)
        assert "Option 6" not in result  # 7th item never in top_six
        assert "options" in result.lower()  # overflow hint fires via has_more
        assert "more" not in result

    def test_no_and_n_more_in_modifier_prompt(self):
        ctx = _make_context()
        repo = _make_menu_repo()
        many = [f"Choice {i}" for i in range(10)]
        payload = {
            "top_choices": many,
            "total_choices": 10,
            "min_selector": 1, "max_selector": 1,
            "modifier_group_position": 0, "total_modifier_groups": 1,
            "is_last_modifier_prompt": True,
            "speech_noun": "add-on",
        }
        result = ask_for_modifier(ctx, repo, payload)
        assert "and 1 more" not in result
        assert "and 4 more" not in result
        assert "more" not in result
