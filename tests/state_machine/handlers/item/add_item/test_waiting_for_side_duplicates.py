# tests/state_machine/handlers/item/add_item/test_waiting_for_side_duplicates.py
"""Integration tests for WaitingForSideHandler with allow_duplicate_selections=True.

Verifies:
- "Coke" after already having ["coke"] → context stores ["coke","coke"]
- slot_value_counts={"coke": 2} → context stores ["coke","coke"] in one turn
- max_selector is enforced against total count including duplicates
- _choice_payload shows all choices (not filtered by selected) when allow_dupes=True
- _choice_payload shows filtered choices when allow_dupes=False
"""
from app.menu.models import MenuItem, Pricing, SideChoice, SideGroup
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.state_machine.handlers.item.add_item.waiting_for_side_handler import WaitingForSideHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_menu_item(
    *,
    allow_duplicate_selections=True,
    max_selector=3,
    min_selector=1,
):
    side_choices = [
        SideChoice(
            item_id="coke",
            name="Coke",
            normalized_name="coke",
            pricing=Pricing(mode="fixed", price_cents=150),
        ),
        SideChoice(
            item_id="sprite",
            name="Sprite",
            normalized_name="sprite",
            pricing=Pricing(mode="fixed", price_cents=150),
        ),
    ]
    return MenuItem(
        item_id="burger",
        name="Burger",
        normalized_name="burger",
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=800),
        side_groups=[
            SideGroup(
                group_id="drinks",
                name="Drinks",
                normalized_name="drinks",
                is_required=True,
                min_selector=min_selector,
                max_selector=max_selector,
                choices=side_choices,
                allow_duplicate_selections=allow_duplicate_selections,
            )
        ],
        modifier_groups=[],
        available=True,
    )


def _make_context(menu_item: MenuItem, existing_sides: list[str] | None = None) -> ConversationContext:
    from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
    ctx = ConversationContext()
    ctx.current_item_id = menu_item.item_id
    ctx.current_item_name = menu_item.name
    ctx.pending_add_item = build_pending_add_item(menu_item)
    if existing_sides:
        ctx.selected_side_groups["drinks"] = list(existing_sides)
    return ctx


def _make_slot(name: str, value: str) -> SlotValue:
    return SlotValue(name=name, value=value, raw=value, start=None, end=None, confidence=None)


def _handle(ctx: ConversationContext, user_text: str, slots: list | None = None):
    ctx.last_slots = tuple(slots or [])
    return WaitingForSideHandler().handle(
        intent=Intent.ADD_ITEM,
        context=ctx,
        user_text=user_text,
    )


# ---------------------------------------------------------------------------
# Duplicate allowed: existing + new selection appends duplicate
# ---------------------------------------------------------------------------

class TestAllowDuplicatesIntegration:
    def test_re_select_coke_appends_duplicate(self):
        menu_item = _make_menu_item(allow_duplicate_selections=True, max_selector=3)
        ctx = _make_context(menu_item, existing_sides=["coke"])
        _handle(ctx, "coke", slots=[_make_slot("SIDE", "coke")])
        assert ctx.selected_side_groups.get("drinks", []).count("coke") == 2

    def test_three_cokes_via_slot_counts(self):
        """Two SIDE=coke slots → slot_value_counts={"coke": 2} → two added to empty group."""
        menu_item = _make_menu_item(allow_duplicate_selections=True, max_selector=3)
        ctx = _make_context(menu_item)
        slots = [_make_slot("SIDE", "coke"), _make_slot("SIDE", "coke")]
        _handle(ctx, "coke coke", slots=slots)
        assert ctx.selected_side_groups.get("drinks", []).count("coke") == 2

    def test_existing_one_plus_two_slot_counts_equals_three(self):
        menu_item = _make_menu_item(allow_duplicate_selections=True, max_selector=4)
        ctx = _make_context(menu_item, existing_sides=["coke"])
        slots = [_make_slot("SIDE", "coke"), _make_slot("SIDE", "coke")]
        _handle(ctx, "coke coke", slots=slots)
        assert ctx.selected_side_groups.get("drinks", []).count("coke") == 3

    def test_mixed_sides_both_stored(self):
        menu_item = _make_menu_item(allow_duplicate_selections=True, max_selector=4)
        ctx = _make_context(menu_item)
        slots = [_make_slot("SIDE", "coke"), _make_slot("SIDE", "coke"), _make_slot("SIDE", "sprite")]
        _handle(ctx, "coke coke and sprite", slots=slots)
        selected = ctx.selected_side_groups.get("drinks", [])
        assert selected.count("coke") == 2
        assert selected.count("sprite") == 1

    def test_max_selector_enforced_with_duplicates(self):
        """When max_selector=2 and user tries to add 3 cokes, only 2 accepted."""
        menu_item = _make_menu_item(allow_duplicate_selections=True, max_selector=2)
        ctx = _make_context(menu_item)
        slots = [_make_slot("SIDE", "coke")] * 3
        result = _handle(ctx, "coke coke coke", slots=slots)
        # Handler returns too_many_side_choices response and caps at 2
        selected = ctx.selected_side_groups.get("drinks", [])
        assert len(selected) <= 2

    def test_choice_payload_shows_all_choices_when_dupes_allowed(self):
        """_choice_payload should not filter out already-selected items."""
        menu_item = _make_menu_item(allow_duplicate_selections=True, max_selector=3)
        ctx = _make_context(menu_item, existing_sides=["coke"])
        # call _choice_payload via the handler's internal method
        handler = WaitingForSideHandler()
        group = ctx.pending_add_item.side_groups[0]
        payload = handler._choice_payload(ctx, group)
        all_choices = payload["all_choices"]
        assert "Coke" in all_choices
        assert "Sprite" in all_choices


# ---------------------------------------------------------------------------
# Duplicate NOT allowed: same choice blocked once selected
# ---------------------------------------------------------------------------

class TestNoDuplicatesIntegration:
    def test_re_select_blocked_when_no_dupes(self):
        menu_item = _make_menu_item(allow_duplicate_selections=False, max_selector=2)
        ctx = _make_context(menu_item, existing_sides=["coke"])
        _handle(ctx, "coke", slots=[_make_slot("SIDE", "coke")])
        # coke should NOT be added again (stays at 1)
        assert ctx.selected_side_groups.get("drinks", []).count("coke") == 1

    def test_choice_payload_filters_selected_when_no_dupes(self):
        menu_item = _make_menu_item(allow_duplicate_selections=False, max_selector=2)
        ctx = _make_context(menu_item, existing_sides=["coke"])
        handler = WaitingForSideHandler()
        group = ctx.pending_add_item.side_groups[0]
        payload = handler._choice_payload(ctx, group)
        # Coke already selected — should not appear in remaining choices
        assert "Coke" not in payload["all_choices"]
        assert "Sprite" in payload["all_choices"]


# ---------------------------------------------------------------------------
# State after duplicate side selection
# ---------------------------------------------------------------------------

class TestStateAfterDuplicateSelection:
    def test_state_stays_waiting_when_more_needed(self):
        """min_selector=2, user adds 1 → stay in WAITING_FOR_SIDE."""
        menu_item = _make_menu_item(min_selector=2, max_selector=3)
        ctx = _make_context(menu_item)
        result = _handle(ctx, "coke", slots=[_make_slot("SIDE", "coke")])
        assert result.next_state == ConversationState.WAITING_FOR_SIDE

    def test_state_advances_when_min_met(self):
        """min_selector=1, adding 1 → advance."""
        menu_item = _make_menu_item(min_selector=1, max_selector=3, allow_duplicate_selections=True)
        ctx = _make_context(menu_item)
        result = _handle(ctx, "coke", slots=[_make_slot("SIDE", "coke")])
        assert result.next_state != ConversationState.WAITING_FOR_SIDE
