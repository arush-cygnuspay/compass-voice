"""Integration tests for the group_skip_policy refactor in the
modifier and side handlers. Asserts behavior parity for the three
decision tiers (BLOCK_UNDER_MIN, SKIP_OPTIONAL, ADVANCE_MIN_MET) and
the new ``advance_min_met`` log event.
"""
import logging
import unittest

from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.nlu.intent_resolution.intent import Intent
from app.state_machine.handlers.item.add_item.waiting_for_modifier_handler import (
    WaitingForModifierHandler,
)
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import (
    WaitingForSideHandler,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import (
    ModifierSelection,
    PendingAddItem,
    PendingModifierChoice,
    PendingModifierGroup,
    PendingSideChoice,
    PendingSideGroup,
)


class _StubStore:
    def get_item(self, item_id):
        return None


class _StubMenuRepo(MenuRepository):
    def __init__(self) -> None:
        # bypass parent __init__ to avoid file IO in unit tests
        self.store = _StubStore()


class _LogCapture(logging.Handler):
    def __init__(self, *event_names: str) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[logging.LogRecord] = []
        self.event_names = set(event_names)

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() in self.event_names:
            self.events.append(record)


def _make_modifier_group(
    *,
    is_required: bool,
    min_selector: int,
    max_selector: int,
) -> PendingModifierGroup:
    cheese = PendingModifierChoice(
        modifier_id="cheese",
        name="Cheese",
        group_id="mods",
        normalized_name="cheese",
    )
    onions = PendingModifierChoice(
        modifier_id="onions",
        name="Onions",
        group_id="mods",
        normalized_name="onions",
    )
    return PendingModifierGroup(
        group_id="mods",
        name="Burger Modification",
        is_required=is_required,
        min_selector=min_selector,
        max_selector=max_selector,
        choices=[cheese, onions],
        choices_by_modifier_id={"cheese": cheese, "onions": onions},
        choices_by_normalized_name={"cheese": [cheese], "onions": [onions]},
        choice_names=("Cheese", "Onions"),
        normalized_choice_names=("cheese", "onions"),
        top_choice_names=("Cheese", "Onions"),
    )


def _make_modifier_context(group: PendingModifierGroup) -> ConversationContext:
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        modifier_groups=[group],
        modifier_groups_by_id={"mods": group},
        modifier_choice_by_id={c.modifier_id: c for c in group.choices},
    )
    return context


def _make_side_group(
    *,
    is_required: bool,
    min_selector: int,
    max_selector: int,
) -> PendingSideGroup:
    coke = PendingSideChoice(
        item_id="coke",
        name="Coke",
        pricing_mode="fixed",
        normalized_name="coke",
    )
    sprite = PendingSideChoice(
        item_id="sprite",
        name="Sprite",
        pricing_mode="fixed",
        normalized_name="sprite",
    )
    return PendingSideGroup(
        group_id="drinks",
        name="Choose your drink",
        is_required=is_required,
        min_selector=min_selector,
        max_selector=max_selector,
        choices=[coke, sprite],
        choices_by_item_id={"coke": coke, "sprite": sprite},
        choices_by_normalized_name={"coke": [coke], "sprite": [sprite]},
        choice_names=("Coke", "Sprite"),
        normalized_choice_names=("coke", "sprite"),
        top_choice_names=("Coke", "Sprite"),
    )


def _make_side_context(group: PendingSideGroup) -> ConversationContext:
    context = ConversationContext()
    context.current_item_id = "burger"
    context.current_item_name = "Burger"
    context.pending_add_item = PendingAddItem(
        item_id="burger",
        item_name="Burger",
        side_groups=[group],
        side_groups_by_id={"drinks": group},
        side_choice_by_item_id={c.item_id: c for c in group.choices},
    )
    return context


# ── Modifier handler tests ─────────────────────────────────────────


class WaitingForModifierGroupSkipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("app.state_machine.control_intent_resolver")
        self.previous_level = self.logger.level
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.setLevel(self.previous_level)

    def test_deny_on_optional_group_skips_and_advances(self):
        handler = WaitingForModifierHandler()
        context = _make_modifier_context(
            _make_modifier_group(is_required=False, min_selector=0, max_selector=1)
        )

        capture = _LogCapture("skipped_optional_group", "advance_min_met")
        self.logger.addHandler(capture)
        try:
            result = handler.handle(
                intent=Intent.DENY,
                context=context,
                user_text="no",
                session=None,
            )
        finally:
            self.logger.removeHandler(capture)

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertIn("mods", context.skipped_modifier_groups)
        skip_events = [e for e in capture.events if e.getMessage() == "skipped_optional_group"]
        self.assertEqual(len(skip_events), 1)

    def test_done_on_required_group_with_min_met_advances_without_skip(self):
        handler = WaitingForModifierHandler()
        group = _make_modifier_group(is_required=True, min_selector=2, max_selector=3)
        context = _make_modifier_context(group)
        # Pre-populate two selections so min is met.
        context.selected_modifier_groups["mods"] = [
            ModifierSelection(modifier_id="cheese", name="Cheese", action="add"),
            ModifierSelection(modifier_id="onions", name="Onions", action="add"),
        ]

        capture = _LogCapture("advance_min_met", "skipped_optional_group")
        self.logger.addHandler(capture)
        try:
            result = handler.handle(
                intent=Intent.UNKNOWN,
                context=context,
                user_text="thats it",
                session=None,
            )
        finally:
            self.logger.removeHandler(capture)

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertNotIn("mods", context.skipped_modifier_groups)
        advance_events = [e for e in capture.events if e.getMessage() == "advance_min_met"]
        self.assertEqual(len(advance_events), 1)
        self.assertEqual(getattr(advance_events[0], "field_name", None), "modifier")
        self.assertEqual(getattr(advance_events[0], "selected_count", None), 2)
        self.assertEqual(getattr(advance_events[0], "min_required", None), 2)
        self.assertEqual(getattr(advance_events[0], "kind", None), "done")
        skip_events = [e for e in capture.events if e.getMessage() == "skipped_optional_group"]
        self.assertEqual(len(skip_events), 0)

    def test_deny_on_required_group_under_min_blocks(self):
        handler = WaitingForModifierHandler()
        group = _make_modifier_group(is_required=True, min_selector=2, max_selector=3)
        context = _make_modifier_context(group)
        # Single selection — under min.
        context.selected_modifier_groups["mods"] = [
            ModifierSelection(modifier_id="cheese", name="Cheese", action="add"),
        ]

        result = handler.handle(
            intent=Intent.DENY,
            context=context,
            user_text="no",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_MODIFIER)
        self.assertEqual(result.response_key, "required_modifier_cannot_skip")
        self.assertEqual(result.response_payload["remaining_to_min"], 1)
        self.assertEqual(result.response_payload["selected_count"], 1)
        self.assertEqual(result.response_payload["min_required"], 2)
        self.assertEqual(result.response_payload["intent_kind"], "deny")


# ── Side handler tests ─────────────────────────────────────────────


class WaitingForSideGroupSkipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("app.state_machine.control_intent_resolver")
        self.previous_level = self.logger.level
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.setLevel(self.previous_level)

    def test_deny_on_optional_side_group_skips_and_advances(self):
        handler = WaitingForSideHandler(_StubMenuRepo())
        context = _make_side_context(
            _make_side_group(is_required=False, min_selector=0, max_selector=1)
        )

        capture = _LogCapture("skipped_optional_group", "advance_min_met")
        self.logger.addHandler(capture)
        try:
            result = handler.handle(
                intent=Intent.DENY,
                context=context,
                user_text="no",
                session=None,
            )
        finally:
            self.logger.removeHandler(capture)

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertIn("drinks", context.skipped_side_groups)
        skip_events = [e for e in capture.events if e.getMessage() == "skipped_optional_group"]
        self.assertEqual(len(skip_events), 1)

    def test_done_on_required_side_group_with_min_met_advances_without_skip(self):
        handler = WaitingForSideHandler(_StubMenuRepo())
        group = _make_side_group(is_required=True, min_selector=2, max_selector=3)
        context = _make_side_context(group)
        # Pre-populate two selections.
        context.selected_side_groups["drinks"] = ["coke", "sprite"]

        capture = _LogCapture("advance_min_met", "skipped_optional_group")
        self.logger.addHandler(capture)
        try:
            result = handler.handle(
                intent=Intent.UNKNOWN,
                context=context,
                user_text="thats it",
                session=None,
            )
        finally:
            self.logger.removeHandler(capture)

        self.assertNotEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertNotIn("drinks", context.skipped_side_groups)
        advance_events = [e for e in capture.events if e.getMessage() == "advance_min_met"]
        self.assertEqual(len(advance_events), 1)
        self.assertEqual(getattr(advance_events[0], "field_name", None), "side")
        self.assertEqual(getattr(advance_events[0], "selected_count", None), 2)
        self.assertEqual(getattr(advance_events[0], "min_required", None), 2)
        self.assertEqual(getattr(advance_events[0], "kind", None), "done")
        skip_events = [e for e in capture.events if e.getMessage() == "skipped_optional_group"]
        self.assertEqual(len(skip_events), 0)

    def test_deny_on_required_side_group_under_min_blocks(self):
        handler = WaitingForSideHandler(_StubMenuRepo())
        group = _make_side_group(is_required=True, min_selector=2, max_selector=3)
        context = _make_side_context(group)
        # One selection — under min.
        context.selected_side_groups["drinks"] = ["coke"]

        result = handler.handle(
            intent=Intent.DENY,
            context=context,
            user_text="no",
            session=None,
        )

        self.assertEqual(result.next_state, ConversationState.WAITING_FOR_SIDE)
        self.assertEqual(result.response_key, "required_side_cannot_skip")
        self.assertEqual(result.response_payload["remaining_to_min"], 1)
        self.assertEqual(result.response_payload["selected_count"], 1)
        self.assertEqual(result.response_payload["min_required"], 2)
        self.assertEqual(result.response_payload["intent_kind"], "deny")


if __name__ == "__main__":
    unittest.main()
