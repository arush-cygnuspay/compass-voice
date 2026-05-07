# tests/state_machine/handlers/item/add_item/test_waiting_for_side_quantity_slot.py
"""Tests for QUANTITY slot → side multiplier in WaitingForSideHandler.

When a QUANTITY slot is present alongside a SIDE slot (e.g. "Coke twice"
→ SIDE=Coke, QUANTITY=twice → quantity=2), the handler should add 2 Cokes
rather than 1.

Covered scenarios:
- QUANTITY=2 integer slot → 2 of the matched side
- QUANTITY="twice" string slot → 2 (after normalize_quantity)
- QUANTITY="thrice" → 3
- QUANTITY=1 → no scaling (multiplier=1 is a no-op)
- QUANTITY=0 → ignored (not ≥ 2)
- No QUANTITY slot → existing single-slot behavior unchanged
- QUANTITY slot with repeated SIDE slots → QUANTITY scales the existing count
- max_selector cap still enforced when quantity pushes count over limit
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

def _make_menu_item(*, min_selector=1, max_selector=5, allow_duplicate_selections=True):
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
                choices=[
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
                ],
                allow_duplicate_selections=allow_duplicate_selections,
            )
        ],
        modifier_groups=[],
        available=True,
    )


def _make_context(menu_item: MenuItem, existing_sides: list | None = None) -> ConversationContext:
    ctx = ConversationContext()
    ctx.current_item_id = menu_item.item_id
    ctx.current_item_name = menu_item.name
    ctx.pending_add_item = build_pending_add_item(menu_item)
    if existing_sides:
        ctx.selected_side_groups["drinks"] = list(existing_sides)
    return ctx


def _slot(name: str, value) -> SlotValue:
    return SlotValue(name=name, value=value, raw=str(value), start=None, end=None, confidence=None)


def _handle(ctx: ConversationContext, user_text: str, slots: list) -> object:
    ctx.last_slots = tuple(slots)
    return WaitingForSideHandler().handle(
        intent=Intent.ADD_ITEM,
        context=ctx,
        user_text=user_text,
    )


def _selected_coke_count(ctx: ConversationContext) -> int:
    return ctx.selected_side_groups.get("drinks", []).count("coke")


# ---------------------------------------------------------------------------
# QUANTITY slot integer value
# ---------------------------------------------------------------------------

class TestQuantitySlotInteger:
    def test_quantity_2_int_slot_adds_2_cokes(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", 2)])
        assert _selected_coke_count(ctx) == 2

    def test_quantity_3_int_slot_adds_3_cokes(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", 3)])
        assert _selected_coke_count(ctx) == 3

    def test_quantity_1_int_slot_adds_1_coke_no_multiplication(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", 1)])
        assert _selected_coke_count(ctx) == 1

    def test_quantity_0_int_slot_ignored_adds_1(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", 0)])
        assert _selected_coke_count(ctx) == 1


# ---------------------------------------------------------------------------
# QUANTITY slot string phrases
# ---------------------------------------------------------------------------

class TestQuantitySlotStrings:
    def test_quantity_twice_string_adds_2(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke twice", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", "twice")])
        assert _selected_coke_count(ctx) == 2

    def test_quantity_thrice_string_adds_3(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke thrice", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", "thrice")])
        assert _selected_coke_count(ctx) == 3

    def test_quantity_double_string_adds_2(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "double coke", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", "double")])
        assert _selected_coke_count(ctx) == 2

    def test_quantity_triple_string_adds_3(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "triple coke", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", "triple")])
        assert _selected_coke_count(ctx) == 3

    def test_quantity_x2_string_adds_2(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke x2", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", "x2")])
        assert _selected_coke_count(ctx) == 2

    def test_quantity_two_string_adds_2(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "two cokes", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", "two")])
        assert _selected_coke_count(ctx) == 2

    def test_quantity_3_string_digit_adds_3(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke 3 times", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", "3")])
        assert _selected_coke_count(ctx) == 3

    def test_unrecognized_quantity_string_falls_back_to_1(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", "umm")])
        assert _selected_coke_count(ctx) == 1


# ---------------------------------------------------------------------------
# No QUANTITY slot — existing behavior unchanged
# ---------------------------------------------------------------------------

class TestNoQuantitySlot:
    def test_single_side_slot_adds_1(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke", slots=[_slot("SIDE", "coke")])
        assert _selected_coke_count(ctx) == 1

    def test_two_repeated_side_slots_adds_2(self):
        ctx = _make_context(_make_menu_item())
        _handle(ctx, "coke coke", slots=[_slot("SIDE", "coke"), _slot("SIDE", "coke")])
        assert _selected_coke_count(ctx) == 2


# ---------------------------------------------------------------------------
# QUANTITY slot scales existing repeated SIDE slots
# ---------------------------------------------------------------------------

class TestQuantityScalesRepeatedSlots:
    def test_two_side_slots_times_2_gives_4(self):
        ctx = _make_context(_make_menu_item(max_selector=10))
        _handle(
            ctx, "coke coke times 2",
            slots=[_slot("SIDE", "coke"), _slot("SIDE", "coke"), _slot("QUANTITY", 2)],
        )
        assert _selected_coke_count(ctx) == 4


# ---------------------------------------------------------------------------
# max_selector enforcement still applies
# ---------------------------------------------------------------------------

class TestMaxSelectorEnforced:
    def test_quantity_pushes_over_max_caps_at_max(self):
        ctx = _make_context(_make_menu_item(max_selector=2))
        # QUANTITY=3 × 1 SIDE=coke = 3 requested, but max=2
        result = _handle(ctx, "coke", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", 3)])
        selected = ctx.selected_side_groups.get("drinks", [])
        assert len(selected) <= 2


# ---------------------------------------------------------------------------
# allow_duplicate_selections=False — QUANTITY slot has no effect
# ---------------------------------------------------------------------------

class TestNoDuplicatesQuantityIgnored:
    def test_quantity_slot_not_applied_when_dupes_not_allowed(self):
        ctx = _make_context(_make_menu_item(allow_duplicate_selections=False))
        _handle(ctx, "coke", slots=[_slot("SIDE", "coke"), _slot("QUANTITY", 3)])
        # Should be at most 1 since duplicates are not allowed
        assert _selected_coke_count(ctx) == 1
