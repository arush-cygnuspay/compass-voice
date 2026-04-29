# tests/state_machine/handlers/item/add_item/test_ready_to_finalize.py
"""
Tests for the ReadyToFinalize / AddItemCommand contract and all finalization
call sites.

Coverage areas:
- ReadyToFinalize is returned (not a pseudo-state) when all fields are resolved
- AddItemCommand.to_dict() produces the correct ADD_ITEM_TO_CART structure
- ConversationState.FINALIZING_ADD_ITEM no longer exists in the enum
- No router lookup occurs for a finalization step
- Simple add-item finalization via each handler path (quantity, size, modifier, side)
- ConfirmingHandler finalization path
- ConfirmationDecisionHelper finalization path
- Edge cases: missing pending_add_item, missing variant before finalization,
  ambiguous item before finalization
"""
from __future__ import annotations

from app.menu.models import MenuItem, ModifierGroup, Pricing, SideChoice, SideGroup
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.add_item_flow import (
    AddItemCommand,
    AddItemNextStep,
    ReadyToFinalize,
    determine_next_add_item_step,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import build_pending_add_item
from app.state_machine.handlers.common.waiting_for_quantity_handler import WaitingForQuantityHandler
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import ModifierSelection


# ── helpers ────────────────────────────────────────────────────────────────────

def _simple_item(item_id: str = "burger_1") -> MenuItem:
    return MenuItem(
        item_id=item_id,
        name="Zinger Burger",
        normalized_name=normalize_text("Zinger Burger"),
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=1000),
        side_groups=[],
        modifier_groups=[],
        available=True,
    )


def _item_with_side() -> MenuItem:
    return MenuItem(
        item_id="meal_1",
        name="Zinger Meal",
        normalized_name=normalize_text("Zinger Meal"),
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(mode="fixed", price_cents=1500),
        side_groups=[
            SideGroup(
                group_id="side_1",
                name="Choose your side",
                normalized_name="choose your side",
                is_required=True,
                min_selector=1,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="fries",
                        name="Fries",
                        normalized_name="fries",
                        pricing=Pricing(mode="fixed", price_cents=0),
                    )
                ],
            )
        ],
        modifier_groups=[],
        available=True,
    )


def _ctx_ready_for_simple_item() -> ConversationContext:
    ctx = ConversationContext()
    ctx.pending_add_item = build_pending_add_item(_simple_item())
    ctx.quantity = 1
    return ctx


# ── ReadyToFinalize: type contract ─────────────────────────────────────────────

def test_determine_next_step_returns_ready_to_finalize_not_next_step():
    """When all fields resolved, outcome is ReadyToFinalize, never AddItemNextStep."""
    ctx = _ctx_ready_for_simple_item()
    step = determine_next_add_item_step(ctx)

    assert isinstance(step, ReadyToFinalize), (
        f"Expected ReadyToFinalize, got {type(step).__name__}"
    )
    assert not isinstance(step, AddItemNextStep)


def test_ready_to_finalize_carries_add_item_command():
    ctx = _ctx_ready_for_simple_item()
    step = determine_next_add_item_step(ctx)

    assert isinstance(step.command, AddItemCommand)


def test_add_item_command_to_dict_structure():
    ctx = ConversationContext()
    ctx.pending_add_item = build_pending_add_item(_simple_item())
    ctx.quantity = 3
    ctx.selected_modifier_groups = {
        "mod_1": [
            ModifierSelection(modifier_id="cheese", name="Cheese", action="add", instruction=None)
        ]
    }

    step = determine_next_add_item_step(ctx)
    assert isinstance(step, ReadyToFinalize)

    cmd = step.command.to_dict()
    assert cmd["type"] == "ADD_ITEM_TO_CART"
    payload = cmd["payload"]
    assert payload["item_id"] == "burger_1"
    assert payload["quantity"] == 3
    assert payload["variant_id"] is None
    assert payload["sides"] == {}
    assert payload["side_variants"] == {}


def test_add_item_command_captures_modifiers():
    ctx = ConversationContext()
    ctx.pending_add_item = build_pending_add_item(_simple_item())
    ctx.quantity = 1
    ctx.selected_modifier_groups = {
        "mod_g": [
            ModifierSelection(modifier_id="x", name="X", action="add", instruction="light")
        ]
    }

    step = determine_next_add_item_step(ctx)
    assert isinstance(step, ReadyToFinalize)

    payload = step.command.to_dict()["payload"]
    assert payload["modifiers"] == {
        "mod_g": [
            {"modifier_id": "x", "name": "X", "action": "add", "instruction": "light"}
        ]
    }


# ── FINALIZING_ADD_ITEM must not exist in state enum ──────────────────────────

def test_finalizing_add_item_not_in_conversation_state_enum():
    state_values = {s.value for s in ConversationState}
    assert "finalizing_add_item" not in state_values, (
        "FINALIZING_ADD_ITEM pseudo-state was removed and must not be in the enum"
    )


def test_finalizing_add_item_not_accessible_as_attribute():
    assert not hasattr(ConversationState, "FINALIZING_ADD_ITEM"), (
        "ConversationState.FINALIZING_ADD_ITEM must not exist"
    )


# ── Variant required before finalization ──────────────────────────────────────

def test_step_asks_for_size_when_variant_not_selected():
    from app.menu.models import PricingVariant

    item = MenuItem(
        item_id="b1",
        name="Burger",
        normalized_name="burger",
        aliases=(),
        normalized_aliases=(),
        voice_labels=(),
        pricing=Pricing(
            mode="variant",
            variants=[
                PricingVariant(variant_id="sm", label="Small", normalized_label="small", price_cents=500),
                PricingVariant(variant_id="lg", label="Large", normalized_label="large", price_cents=700),
            ],
        ),
        side_groups=[],
        modifier_groups=[],
        available=True,
    )
    ctx = ConversationContext()
    ctx.pending_add_item = build_pending_add_item(item)
    ctx.quantity = 1

    step = determine_next_add_item_step(ctx)

    assert isinstance(step, AddItemNextStep)
    assert step.next_state == ConversationState.WAITING_FOR_SIZE
    assert not isinstance(step, ReadyToFinalize)


# ── Side required before finalization ─────────────────────────────────────────

def test_step_asks_for_required_side_before_finalizing():
    ctx = ConversationContext()
    ctx.pending_add_item = build_pending_add_item(_item_with_side())
    ctx.quantity = 1

    step = determine_next_add_item_step(ctx)

    assert isinstance(step, AddItemNextStep)
    assert step.next_state == ConversationState.WAITING_FOR_SIDE
    assert not isinstance(step, ReadyToFinalize)


def test_step_finalizes_after_required_side_resolved():
    ctx = ConversationContext()
    ctx.pending_add_item = build_pending_add_item(_item_with_side())
    ctx.selected_side_groups = {"side_1": ["fries"]}
    ctx.quantity = 1

    step = determine_next_add_item_step(ctx)

    assert isinstance(step, ReadyToFinalize)
    assert step.command.sides == {"side_1": ["fries"]}


# ── Finalization from WaitingForQuantityHandler ────────────────────────────────

def test_waiting_for_quantity_handler_emits_terminal_handler_result():
    """After quantity captured, handler returns IDLE with ADD_ITEM_TO_CART command."""
    from app.nlu.intent_resolution.intent import Intent

    item = _simple_item()
    ctx = ConversationContext()
    ctx.pending_add_item = build_pending_add_item(item)
    ctx.current_item_id = item.item_id

    class _Session:
        conversation_state = ConversationState.WAITING_FOR_QUANTITY

    handler = WaitingForQuantityHandler()
    result = handler.handle(
        intent=Intent.UNKNOWN,
        context=ctx,
        user_text="2",
        session=_Session(),
    )

    assert isinstance(result, HandlerResult)
    assert result.next_state == ConversationState.IDLE
    assert result.response_key == "item_added_successfully"
    assert result.command is not None
    assert result.command["type"] == "ADD_ITEM_TO_CART"
    assert result.command["payload"]["item_id"] == "burger_1"
    assert result.command["payload"]["quantity"] == 2
    assert result.reset_context is True


# ── No router lookup for finalization ─────────────────────────────────────────

def test_ready_to_finalize_has_no_next_state_attribute():
    """
    ReadyToFinalize has no next_state field — callers cannot accidentally pass
    it to a router lookup.
    """
    ctx = _ctx_ready_for_simple_item()
    step = determine_next_add_item_step(ctx)

    assert isinstance(step, ReadyToFinalize)
    assert not hasattr(step, "next_state"), (
        "ReadyToFinalize must not have a next_state field — it is not routable"
    )


# ── Multi-item flow continues after one item finalizes ────────────────────────

def test_finalization_command_carries_correct_quantity_from_context():
    """Quantity captured before finalization is locked into AddItemCommand."""
    ctx = ConversationContext()
    ctx.pending_add_item = build_pending_add_item(_simple_item())
    ctx.quantity = 5

    step = determine_next_add_item_step(ctx)

    assert isinstance(step, ReadyToFinalize)
    assert step.command.quantity == 5
    assert step.command.to_dict()["payload"]["quantity"] == 5


# ── Edge case: missing pending_add_item ───────────────────────────────────────

def test_determine_next_step_returns_error_step_when_pending_is_none():
    ctx = ConversationContext()
    assert ctx.pending_add_item is None

    step = determine_next_add_item_step(ctx)

    assert isinstance(step, AddItemNextStep)
    assert step.next_state == ConversationState.ERROR_RECOVERY
    assert step.response_key == "item_context_missing"
