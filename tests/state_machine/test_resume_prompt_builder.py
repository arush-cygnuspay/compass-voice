# tests/state_machine/test_resume_prompt_builder.py
"""Tests for ResumePromptBuilder and the resume-prompt rendering path.

Validates:
- Each WAITING_FOR_* state produces the correct prompt type.
- Payload always includes current_item_name, field, top_choices.
- top_choices are sourced from context.available_choices_values.
- Missing/None context fields degrade gracefully without crashing.
- Non-resume states (IDLE, CONFIRMING_*) return None.
- ask_for_size renderer uses payload.current_item_name when provided.
- ask_for_size renderer falls back to menu lookup when payload is empty.
- ask_for_side renderer uses payload.top_choices when context is stale.
- ask_for_modifier renderer uses payload.top_choices when context is stale.
- Disconnect/reconnect scenario: stored payload renders correctly.
- Edge cases: empty choices, stale current_item_id, partially hydrated session.
"""
from __future__ import annotations

import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Minimal stubs so imports succeed without the full environment
# ---------------------------------------------------------------------------
for _mod_name in ("dotenv", "redis"):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _mod_name == "dotenv":
            _stub.load_dotenv = lambda *a, **k: None
        else:
            _stub.Redis = object
        sys.modules[_mod_name] = _stub

from app.state_machine.resume_prompt_builder import ResumePromptBuilder, ResumePromptPayload
from app.state_machine.models.conversation_state import ConversationState
from app.responses.item_responses import ask_for_size, ask_for_side, ask_for_modifier


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------

class _FakeContext:
    current_item_id = "item-1"
    current_item_name = "Burger"
    current_side_group_index = 0
    current_modifier_group_index = 0
    selected_side_groups: dict = {}
    selected_modifier_groups: dict = {}
    pending_side_item_name: str | None = None
    available_choices_kind: str | None = None
    available_choices_values: tuple = ()


class _FakeSession:
    def __init__(self, state: ConversationState, context: _FakeContext | None = None):
        self.conversation_state = state
        self.conversation_context = context or _FakeContext()


class _FakeItem:
    def __init__(self, name: str, side_groups=None, modifier_groups=None, variants=None):
        self.name = name
        self.side_groups = side_groups or []
        self.modifier_groups = modifier_groups or []
        self.variants = variants or []


class _FakeGroup:
    def __init__(self, choices=None, *, prompt_noun=None, prompt_verb=None,
                 min_selector=0, max_selector=0, group_id="g1", name="Group"):
        self.choices = choices or []
        self.prompt_noun = prompt_noun
        self.prompt_verb = prompt_verb
        self.min_selector = min_selector
        self.max_selector = max_selector
        self.group_id = group_id
        self.name = name


class _FakeChoice:
    def __init__(self, name: str, item_id: str = "c1"):
        self.name = name
        self.item_id = item_id


class _FakeStore:
    def __init__(self, item: _FakeItem | None = None):
        self._item = item

    def get_item(self, item_id):
        if self._item is None:
            raise KeyError(f"Item not found: {item_id!r}")
        return self._item


class _FakeMenuRepo:
    def __init__(self, item: _FakeItem | None = None):
        self.store = _FakeStore(item)


_BROKEN_MENU = _FakeMenuRepo(item=None)   # always raises on get_item


# ---------------------------------------------------------------------------
# ResumePromptBuilder.build() — prompt type and payload fields
# ---------------------------------------------------------------------------

class BuilderPromptTypeTests(unittest.TestCase):

    def _build(self, state, **ctx_overrides):
        ctx = _FakeContext()
        for k, v in ctx_overrides.items():
            setattr(ctx, k, v)
        session = _FakeSession(state, ctx)
        return ResumePromptBuilder().build(session)

    def test_waiting_for_size_returns_ask_for_size(self):
        key, _ = self._build(ConversationState.WAITING_FOR_SIZE)
        self.assertEqual(key, "ask_for_size")

    def test_waiting_for_side_returns_ask_for_side(self):
        key, _ = self._build(ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(key, "ask_for_side")

    def test_waiting_for_side_size_returns_ask_for_side_size(self):
        key, _ = self._build(ConversationState.WAITING_FOR_SIDE_SIZE)
        self.assertEqual(key, "ask_for_side_size")

    def test_waiting_for_modifier_returns_ask_for_modifier(self):
        key, _ = self._build(ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(key, "ask_for_modifier")

    def test_waiting_for_quantity_returns_ask_for_quantity(self):
        key, _ = self._build(ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(key, "ask_for_quantity")

    def test_non_waiting_state_returns_none(self):
        result = self._build(ConversationState.IDLE)
        self.assertIsNone(result)

    def test_confirming_order_returns_none(self):
        result = self._build(ConversationState.CONFIRMING_ORDER)
        self.assertIsNone(result)


class BuilderPayloadFieldsTests(unittest.TestCase):

    def _build(self, state, **ctx_overrides):
        ctx = _FakeContext()
        for k, v in ctx_overrides.items():
            setattr(ctx, k, v)
        session = _FakeSession(state, ctx)
        return ResumePromptBuilder().build(session)

    # --- current_item_name ---

    def test_size_payload_includes_item_name(self):
        _, payload = self._build(
            ConversationState.WAITING_FOR_SIZE,
            current_item_name="Pizza",
        )
        self.assertEqual(payload["current_item_name"], "Pizza")

    def test_side_payload_includes_item_name(self):
        _, payload = self._build(
            ConversationState.WAITING_FOR_SIDE,
            current_item_name="Wrap",
        )
        self.assertEqual(payload["current_item_name"], "Wrap")

    def test_modifier_payload_includes_item_name(self):
        _, payload = self._build(
            ConversationState.WAITING_FOR_MODIFIER,
            current_item_name="Hot Dog",
        )
        self.assertEqual(payload["current_item_name"], "Hot Dog")

    def test_quantity_payload_includes_item_name_key(self):
        _, payload = self._build(
            ConversationState.WAITING_FOR_QUANTITY,
            current_item_name="Salad",
        )
        self.assertEqual(payload["item_name"], "Salad")

    # --- field tag ---

    def test_size_payload_field_is_size(self):
        _, p = self._build(ConversationState.WAITING_FOR_SIZE)
        self.assertEqual(p["field"], "size")

    def test_side_payload_field_is_side(self):
        _, p = self._build(ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(p["field"], "side")

    def test_modifier_payload_field_is_modifier(self):
        _, p = self._build(ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(p["field"], "modifier")

    def test_side_size_payload_field_is_side_size(self):
        _, p = self._build(ConversationState.WAITING_FOR_SIDE_SIZE)
        self.assertEqual(p["field"], "side_size")

    def test_quantity_payload_field_is_quantity(self):
        _, p = self._build(ConversationState.WAITING_FOR_QUANTITY)
        self.assertEqual(p["field"], "quantity")

    # --- top_choices from context.available_choices_values ---

    def test_size_top_choices_from_available_values(self):
        _, p = self._build(
            ConversationState.WAITING_FOR_SIZE,
            available_choices_values=("Small", "Medium", "Large"),
        )
        self.assertEqual(p["top_choices"], ["Small", "Medium", "Large"])

    def test_side_top_choices_capped_at_4(self):
        _, p = self._build(
            ConversationState.WAITING_FOR_SIDE,
            available_choices_values=("A", "B", "C", "D", "E", "F"),
        )
        self.assertEqual(p["top_choices"], ["A", "B", "C", "D"])

    def test_modifier_top_choices_from_available_values(self):
        _, p = self._build(
            ConversationState.WAITING_FOR_MODIFIER,
            available_choices_values=("Ketchup", "Mustard"),
        )
        self.assertEqual(p["top_choices"], ["Ketchup", "Mustard"])

    def test_empty_choices_yields_empty_list(self):
        _, p = self._build(
            ConversationState.WAITING_FOR_SIZE,
            available_choices_values=(),
        )
        self.assertEqual(p["top_choices"], [])

    # --- side_size specific keys ---

    def test_side_size_payload_includes_side_item_name(self):
        _, p = self._build(
            ConversationState.WAITING_FOR_SIDE_SIZE,
            pending_side_item_name="Caesar Salad",
        )
        self.assertEqual(p["side_item_name"], "Caesar Salad")

    def test_side_size_payload_includes_available_sizes(self):
        _, p = self._build(
            ConversationState.WAITING_FOR_SIDE_SIZE,
            available_choices_values=("Small", "Large"),
        )
        self.assertEqual(p["available_sizes"], ["Small", "Large"])

    # --- graceful fallbacks for missing context ---

    def test_none_item_name_uses_fallback_string(self):
        _, p = self._build(
            ConversationState.WAITING_FOR_SIZE,
            current_item_name=None,
        )
        self.assertIsNotNone(p["current_item_name"])
        self.assertNotEqual(p["current_item_name"], "")

    def test_payload_is_always_a_dict(self):
        for state in (
            ConversationState.WAITING_FOR_SIZE,
            ConversationState.WAITING_FOR_SIDE,
            ConversationState.WAITING_FOR_SIDE_SIZE,
            ConversationState.WAITING_FOR_MODIFIER,
            ConversationState.WAITING_FOR_QUANTITY,
        ):
            with self.subTest(state=state):
                result = self._build(state)
                self.assertIsNotNone(result)
                _, payload = result
                self.assertIsInstance(payload, dict)


# ---------------------------------------------------------------------------
# ResumePromptPayload dataclass
# ---------------------------------------------------------------------------

class ResumePromptPayloadTests(unittest.TestCase):

    def test_to_dict_includes_all_spec_fields(self):
        p = ResumePromptPayload(
            current_item_name="Burger",
            field="size",
            group="Sizes",
            top_choices=["Small", "Large"],
        )
        d = p.to_dict()
        self.assertEqual(d["current_item_name"], "Burger")
        self.assertEqual(d["field"], "size")
        self.assertEqual(d["group"], "Sizes")
        self.assertEqual(d["top_choices"], ["Small", "Large"])

    def test_defaults_are_safe(self):
        p = ResumePromptPayload()
        d = p.to_dict()
        self.assertIsNone(d["current_item_name"])
        self.assertIsNone(d["field"])
        self.assertIsNone(d["group"])
        self.assertEqual(d["top_choices"], [])


# ---------------------------------------------------------------------------
# ask_for_size renderer — payload-first behaviour
# ---------------------------------------------------------------------------

class AskForSizeRendererTests(unittest.TestCase):

    def test_uses_payload_item_name_when_provided(self):
        ctx = _FakeContext()
        ctx.current_item_id = None       # stale — menu lookup would fail
        payload = {"current_item_name": "Veggie Wrap"}
        result = ask_for_size(ctx, _BROKEN_MENU, payload)
        self.assertIn("Veggie Wrap", result)

    def test_falls_back_to_menu_lookup_when_payload_is_empty(self):
        item = _FakeItem(name="Classic Burger")
        ctx = _FakeContext()
        result = ask_for_size(ctx, _FakeMenuRepo(item), {})
        self.assertIn("Classic Burger", result)

    def test_falls_back_to_menu_lookup_when_no_payload(self):
        item = _FakeItem(name="Chicken Sandwich")
        ctx = _FakeContext()
        result = ask_for_size(ctx, _FakeMenuRepo(item), None)
        self.assertIn("Chicken Sandwich", result)

    def test_prompt_text_format(self):
        payload = {"current_item_name": "Tacos"}
        ctx = _FakeContext()
        result = ask_for_size(ctx, _BROKEN_MENU, payload)
        self.assertIn("What size would you like for Tacos", result)


# ---------------------------------------------------------------------------
# ask_for_side renderer — graceful degradation with stale context
# ---------------------------------------------------------------------------

class AskForSideRendererTests(unittest.TestCase):

    def test_uses_payload_top_choices_when_context_stale(self):
        ctx = _FakeContext()
        ctx.current_item_id = None   # triggers menu lookup failure
        payload = {
            "current_item_name": "Burger",
            "top_choices": ["Caesar Salad", "Soup"],
        }
        result = ask_for_side(ctx, _BROKEN_MENU, payload)
        self.assertIn("Caesar Salad", result)

    def test_uses_item_name_from_payload_when_menu_fails(self):
        # item_name appears in the "no examples" branch (empty top_choices)
        ctx = _FakeContext()
        ctx.current_item_id = None
        payload = {
            "current_item_name": "Steak",
            "top_choices": [],
        }
        result = ask_for_side(ctx, _BROKEN_MENU, payload)
        self.assertIn("Steak", result)

    def test_normal_flow_unaffected_when_context_intact(self):
        choices = [_FakeChoice("Fries", "c1"), _FakeChoice("Salad", "c2")]
        group = _FakeGroup(choices=choices, min_selector=0)
        item = _FakeItem(name="Burger", side_groups=[group])
        ctx = _FakeContext()
        result = ask_for_side(ctx, _FakeMenuRepo(item), {})
        self.assertIn("Fries", result)
        self.assertIn("You can say none", result)

    def test_optional_side_appends_none_hint(self):
        choices = [_FakeChoice("Fries")]
        group = _FakeGroup(choices=choices, min_selector=0)
        item = _FakeItem(name="Burger", side_groups=[group])
        ctx = _FakeContext()
        result = ask_for_side(ctx, _FakeMenuRepo(item), {})
        self.assertIn("none", result.lower())

    def test_empty_payload_top_choices_falls_back_gracefully(self):
        ctx = _FakeContext()
        ctx.current_item_id = None
        payload = {"current_item_name": "Pizza", "top_choices": []}
        result = ask_for_side(ctx, _BROKEN_MENU, payload)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


# ---------------------------------------------------------------------------
# ask_for_modifier renderer — graceful degradation with stale context
# ---------------------------------------------------------------------------

class AskForModifierRendererTests(unittest.TestCase):

    def test_uses_payload_top_choices_when_context_stale(self):
        ctx = _FakeContext()
        ctx.current_item_id = None
        payload = {
            "current_item_name": "Hot Dog",
            "top_choices": ["Ketchup", "Mustard"],
        }
        result = ask_for_modifier(ctx, _BROKEN_MENU, payload)
        self.assertIn("Ketchup", result)

    def test_uses_item_name_from_payload_when_menu_fails(self):
        # item_name appears in the "no examples" branch (empty top_choices)
        ctx = _FakeContext()
        ctx.current_item_id = None
        payload = {
            "current_item_name": "Wrap",
            "top_choices": [],
        }
        result = ask_for_modifier(ctx, _BROKEN_MENU, payload)
        self.assertIn("Wrap", result)

    def test_normal_flow_unaffected_when_context_intact(self):
        choices = [_FakeChoice("Extra Cheese", "m1"), _FakeChoice("Bacon", "m2")]
        group = _FakeGroup(choices=choices, min_selector=0)
        item = _FakeItem(name="Burger", modifier_groups=[group])
        ctx = _FakeContext()
        result = ask_for_modifier(ctx, _FakeMenuRepo(item), {})
        self.assertIn("Extra Cheese", result)

    def test_empty_choices_in_payload_returns_fallback_prompt(self):
        ctx = _FakeContext()
        ctx.current_item_id = None
        payload = {"current_item_name": "Salad", "top_choices": []}
        result = ask_for_modifier(ctx, _BROKEN_MENU, payload)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


# ---------------------------------------------------------------------------
# Disconnect/reconnect scenario
# ---------------------------------------------------------------------------

class DisconnectReconnectTests(unittest.TestCase):
    """Simulate storing the resume payload and re-rendering on reconnect."""

    def test_stored_size_payload_renders_without_item_id(self):
        # Builder runs while context is intact
        ctx = _FakeContext()
        ctx.current_item_name = "Club Sandwich"
        ctx.available_choices_values = ("Small", "Medium", "Large")
        session = _FakeSession(ConversationState.WAITING_FOR_SIZE, ctx)
        _, stored_payload = ResumePromptBuilder().build(session)

        # On reconnect: context is stale (no item_id)
        stale_ctx = _FakeContext()
        stale_ctx.current_item_id = None
        result = ask_for_size(stale_ctx, _BROKEN_MENU, stored_payload)
        self.assertIn("Club Sandwich", result)

    def test_stored_side_payload_renders_with_choices_on_reconnect(self):
        ctx = _FakeContext()
        ctx.current_item_name = "Burger"
        ctx.available_choices_values = ("Caesar Salad", "Soup", "Fries")
        session = _FakeSession(ConversationState.WAITING_FOR_SIDE, ctx)
        _, stored_payload = ResumePromptBuilder().build(session)

        stale_ctx = _FakeContext()
        stale_ctx.current_item_id = None
        result = ask_for_side(stale_ctx, _BROKEN_MENU, stored_payload)
        self.assertIn("Caesar Salad", result)

    def test_stored_modifier_payload_renders_with_choices_on_reconnect(self):
        ctx = _FakeContext()
        ctx.current_item_name = "Hot Dog"
        ctx.available_choices_values = ("Ketchup", "Mustard", "Relish")
        session = _FakeSession(ConversationState.WAITING_FOR_MODIFIER, ctx)
        _, stored_payload = ResumePromptBuilder().build(session)

        stale_ctx = _FakeContext()
        stale_ctx.current_item_id = None
        result = ask_for_modifier(stale_ctx, _BROKEN_MENU, stored_payload)
        self.assertIn("Ketchup", result)

    def test_reconnect_with_partially_hydrated_session_does_not_crash(self):
        ctx = _FakeContext()
        ctx.current_item_name = None
        ctx.available_choices_values = ()
        session = _FakeSession(ConversationState.WAITING_FOR_SIZE, ctx)
        result = ResumePromptBuilder().build(session)
        self.assertIsNotNone(result)
        _, payload = result
        # Renderer must not crash even with a thin payload
        stale_ctx = _FakeContext()
        stale_ctx.current_item_id = None
        rendered = ask_for_size(stale_ctx, _BROKEN_MENU, payload)
        self.assertIsInstance(rendered, str)


if __name__ == "__main__":
    unittest.main()
