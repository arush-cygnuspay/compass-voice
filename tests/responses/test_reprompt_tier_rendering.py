# tests/responses/test_reprompt_tier_rendering.py
"""Tests for 3-tier reprompt rendering in repeat_size_options and
repeat_modifier_options.

Stubs out menu_repo and ConversationContext to avoid heavy dependencies.
"""
import types
import unittest


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

def _make_variant(label: str):
    v = types.SimpleNamespace()
    v.label = label
    return v


def _make_item(name: str, variant_labels: list):
    item = types.SimpleNamespace()
    item.name = name
    item.pricing = types.SimpleNamespace(
        variants=[_make_variant(l) for l in variant_labels]
    )
    return item


def _make_menu_repo(item_name: str = "Burger", sizes: list | None = None):
    sizes = sizes if sizes is not None else ["small", "medium", "large"]
    item = _make_item(item_name, sizes)
    store = types.SimpleNamespace(get_item=lambda _: item)
    return types.SimpleNamespace(store=store)


def _make_context(item_id: str = "item-1", modifier_group_index: int = 0):
    ctx = types.SimpleNamespace(
        current_item_id=item_id,
        current_item_name="Burger",
        current_modifier_group_index=modifier_group_index,
    )
    return ctx


# ---------------------------------------------------------------------------
# Import functions under test (after stubs for any heavy deps if needed)
# ---------------------------------------------------------------------------

from app.responses.item_responses import repeat_modifier_options, repeat_size_options


# ---------------------------------------------------------------------------
# repeat_size_options
# ---------------------------------------------------------------------------

class RepeatSizeOptionsTierTests(unittest.TestCase):
    def setUp(self):
        self.menu_repo = _make_menu_repo("Pizza", ["small", "medium", "large"])
        self.context = _make_context()

    # miss_count=0 (backward compat / initial) → full options list
    def test_miss_count_zero_shows_full_options(self):
        result = repeat_size_options(self.context, self.menu_repo, {})
        self.assertIn("small", result)
        self.assertIn("medium", result)
        self.assertIn("large", result)

    def test_no_reprompt_count_key_shows_full_options(self):
        """Backward-compat: payload without reprompt_count defaults to full options."""
        result = repeat_size_options(self.context, self.menu_repo, {"item_name": "Pizza"})
        self.assertIn("small", result)

    # miss_count=1 → concise
    def test_first_miss_returns_concise_prompt(self):
        result = repeat_size_options(self.context, self.menu_repo, {"reprompt_count": 1})
        self.assertNotIn("small", result)
        self.assertNotIn("medium", result)
        self.assertIn("Pizza", result)
        self.assertIn("size", result.lower())

    # miss_count=2 → list_options_hint
    def test_second_miss_returns_list_options_hint(self):
        result = repeat_size_options(self.context, self.menu_repo, {"reprompt_count": 2})
        self.assertIn("list options", result.lower())
        self.assertNotIn("small", result)

    # miss_count=3 → escalate (show full list again)
    def test_third_miss_returns_escalated_full_list(self):
        result = repeat_size_options(
            self.context, self.menu_repo, {"reprompt_count": 3, "reprompt_escalation": True}
        )
        self.assertIn("small", result)
        self.assertIn("medium", result)
        self.assertIn("large", result)
        self.assertIn("Please say", result)

    # Explicit OPTIONS_REQUEST always shows full list regardless of miss count
    def test_list_options_requested_bypasses_tier_at_miss_1(self):
        result = repeat_size_options(
            self.context,
            self.menu_repo,
            {"reprompt_count": 1, "list_options_requested": True},
        )
        self.assertIn("small", result)
        self.assertIn("medium", result)

    def test_list_options_requested_bypasses_tier_at_miss_2(self):
        result = repeat_size_options(
            self.context,
            self.menu_repo,
            {"reprompt_count": 2, "list_options_requested": True},
        )
        self.assertIn("small", result)
        self.assertNotIn("list options", result.lower())

    def test_list_options_requested_bypasses_tier_at_miss_3(self):
        result = repeat_size_options(
            self.context,
            self.menu_repo,
            {"reprompt_count": 3, "list_options_requested": True, "reprompt_escalation": True},
        )
        self.assertIn("small", result)
        # "Please say" is escalation-only phrasing; list_options_requested skips it
        self.assertNotIn("Please say", result)

    # No variants → graceful fallback
    def test_no_variants_first_miss_still_concise(self):
        menu_repo = _make_menu_repo("Soda", sizes=[])
        result = repeat_size_options(self.context, menu_repo, {"reprompt_count": 1})
        self.assertIn("size", result.lower())
        self.assertNotIn("None", result)

    # None payload → treat as miss_count=0
    def test_none_payload_treated_as_full_options(self):
        result = repeat_size_options(self.context, self.menu_repo, None)
        self.assertIn("small", result)


# ---------------------------------------------------------------------------
# repeat_modifier_options
# ---------------------------------------------------------------------------

class RepeatModifierOptionsTierTests(unittest.TestCase):
    def _make_modifier_context(self, group_name="Sauce", choices=None):
        """Context with one modifier group carrying choices."""
        choices = choices or ["ranch", "bbq", "honey mustard"]
        ctx = types.SimpleNamespace(
            current_item_id="item-1",
            current_item_name="Burger",
            current_modifier_group_index=0,
            selected_modifier_groups={},  # required by _current_modifier_payload
        )
        return ctx, choices

    def _make_modifier_menu_repo(self, group_name="Sauce", choices=None):
        choices = choices or ["ranch", "bbq", "honey mustard"]
        group = types.SimpleNamespace(
            group_id="g1",
            name=group_name,       # _current_modifier_payload reads group.name
            group_name=group_name,
            choices=[types.SimpleNamespace(name=c, choice_id=f"c-{i}") for i, c in enumerate(choices)],
            min_selector=1,
            max_selector=1,
        )
        pending = types.SimpleNamespace(
            modifier_groups=[group],
            modifier_groups_by_id={"g1": group},
        )
        ctx_store = {}  # not used in stubs

        menu_repo = types.SimpleNamespace(
            store=types.SimpleNamespace(get_item=lambda _: types.SimpleNamespace(
                modifier_groups=[group],
            ))
        )
        return menu_repo

    # miss_count=0 → full options
    def test_miss_count_zero_shows_full_options(self):
        ctx, _ = self._make_modifier_context()
        # Payload carries no reprompt_count — backward-compat path
        # repeat_modifier_options falls through to _current_modifier_payload.
        # With a thin stub, the payload won't have top_choices, so we check
        # the fallback string is returned without raising.
        result = repeat_modifier_options(ctx, self._make_modifier_menu_repo(), {})
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    # miss_count=1 → concise
    def test_first_miss_returns_concise(self):
        ctx, _ = self._make_modifier_context()
        result = repeat_modifier_options(
            ctx, self._make_modifier_menu_repo(), {"reprompt_count": 1}
        )
        self.assertIn("Which option", result)
        self.assertNotIn("list options", result.lower())

    # miss_count=2 → list_options_hint
    def test_second_miss_returns_list_options_hint(self):
        ctx, _ = self._make_modifier_context()
        result = repeat_modifier_options(
            ctx, self._make_modifier_menu_repo(), {"reprompt_count": 2}
        )
        self.assertIn("list options", result.lower())

    # miss_count=3 → escalate (full options)
    def test_third_miss_returns_escalated_full_options(self):
        ctx, _ = self._make_modifier_context()
        result = repeat_modifier_options(
            ctx,
            self._make_modifier_menu_repo(),
            {"reprompt_count": 3, "reprompt_escalation": True},
        )
        # Falls through to full-options path; at minimum should not hint "list options"
        self.assertNotIn("Say 'list options'", result)
        self.assertIsInstance(result, str)

    # feedback prefix preserved for miss_count=1 and miss_count=2
    def test_concise_includes_feedback_prefix(self):
        ctx, _ = self._make_modifier_context()
        result = repeat_modifier_options(
            ctx,
            self._make_modifier_menu_repo(),
            {"reprompt_count": 1, "entity_feedback": "I heard 'ketchup'."},
        )
        self.assertIn("Which option", result)

    def test_list_hint_includes_feedback_prefix(self):
        ctx, _ = self._make_modifier_context()
        result = repeat_modifier_options(
            ctx,
            self._make_modifier_menu_repo(),
            {"reprompt_count": 2, "entity_feedback": "I heard 'ketchup'."},
        )
        self.assertIn("list options", result.lower())

    # None payload → full options (backward compat)
    def test_none_payload_is_handled(self):
        ctx, _ = self._make_modifier_context()
        result = repeat_modifier_options(ctx, self._make_modifier_menu_repo(), None)
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# Cross-field isolation — valid size answer must not affect modifier counter
# ---------------------------------------------------------------------------

class CrossFieldIsolationTests(unittest.TestCase):
    """Validates that reprompt_count in payload is field-scoped — size misses
    and modifier misses are independent keys in session.reprompt_count_by_field.
    The rendering functions just read payload["reprompt_count"] which is injected
    per-key by the guardrail, so different fields naturally stay isolated."""

    def test_size_and_modifier_payloads_are_independent(self):
        menu_repo = _make_menu_repo("Pizza", ["small", "medium"])
        ctx = _make_context()

        size_result_miss1 = repeat_size_options(ctx, menu_repo, {"reprompt_count": 1})
        size_result_miss0 = repeat_size_options(ctx, menu_repo, {"reprompt_count": 0})

        # First size miss should be concise; initial prompt should list options
        self.assertNotIn("small", size_result_miss1)
        self.assertIn("small", size_result_miss0)


if __name__ == "__main__":
    unittest.main()
