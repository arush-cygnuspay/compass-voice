"""Tests for the four components extracted from AddItemHandler.

Covers:
  - ConfirmationDecisionHelper: direct-add vs. waiting-state branching
  - ItemResolutionHandler: routing for ITEM, CATEGORY_SINGLE_ITEM, CATEGORY,
    ITEM_AMBIGUOUS, NOT_FOUND; modifier-only guard
  - MultiItemQueueCoordinator: queue setup, first-item resolution, ack payload
  - PrefillOrchestrator: prefill application, missing-group naming,
    prefilled-summary building, feedback/debug payloads
  - normalize_item_request_text: filler-prefix stripping
"""
from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from app.menu.models import (
    MenuItem,
    ModifierChoice,
    ModifierGroup,
    Pricing,
    SideChoice,
    SideGroup,
)
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.nlu.intent_resolution.intent import Intent
from app.nlu.multi_item_parser import ParsedItemSegment
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.confirmation_decision_helper import (
    ConfirmationDecisionHelper,
)
from app.state_machine.handlers.item.add_item.item_resolution_handler import (
    ItemResolutionHandler,
)
from app.state_machine.handlers.item.add_item.multi_group_prefill import MultiGroupPrefillEngine
from app.state_machine.handlers.item.add_item.multi_item_queue_coordinator import (
    MultiItemQueueCoordinator,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import (
    build_pending_add_item,
)
from app.state_machine.handlers.item.add_item.prefill_orchestrator import (
    PendingItemCaptureHelper,
    PrefillOrchestrator,
    normalize_item_request_text,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.pending_item_models import ModifierSelection


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

def _simple_item(*, with_side: bool = False, with_modifier: bool = False) -> MenuItem:
    side_groups = []
    if with_side:
        side_groups = [
            SideGroup(
                group_id="drink_group",
                name="Drink",
                normalized_name="drink",
                is_required=False,
                min_selector=0,
                max_selector=1,
                choices=[
                    SideChoice(
                        item_id="coke",
                        name="Coke",
                        normalized_name="coke",
                        pricing=Pricing(mode="fixed", price_cents=0),
                    )
                ],
            )
        ]
    modifier_groups = []
    if with_modifier:
        modifier_groups = [
            ModifierGroup(
                group_id="mod_group",
                name="Extras",
                normalized_name="extras",
                is_required=False,
                min_selector=0,
                max_selector=2,
                choices=[
                    ModifierChoice(
                        modifier_id="cheese",
                        name="Cheese",
                        normalized_name="cheese",
                        price_cents=50,
                    )
                ],
            )
        ]
    return MenuItem(
        item_id="burger_1",
        name="Zinger Burger",
        normalized_name="zinger burger",
        aliases=("zinger burger",),
        normalized_aliases=("zinger burger",),
        voice_labels=("zinger burger",),
        pricing=Pricing(mode="fixed", price_cents=500),
        side_groups=side_groups,
        modifier_groups=modifier_groups,
        available=True,
    )


class _FakeRepo:
    def __init__(self, result: MenuQueryResult):
        self._result = result

    def resolve_menu_query_normalized(self, text: str, limit: int = 5):
        return self._result

    def resolve_menu_query_from_slots_normalized(self, **kwargs):
        return self._result

    def find_near_miss_item_normalized(self, normalized_text, *, threshold=None):
        return None

    store = None


def _ctx() -> ConversationContext:
    ctx = ConversationContext()
    ctx.quantity = 1
    return ctx


# ---------------------------------------------------------------------------
# normalize_item_request_text
# ---------------------------------------------------------------------------

class TestNormalizeItemRequestText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("I want a burger", "burger"),
            ("add a large coke", "large coke"),
            ("Can I get the fries", "fries"),
            ("give me an apple pie", "apple pie"),
            ("bring a coffee", "coffee"),
            ("i would like to order two tacos", "to order two tacos"),
            ("burger", "burger"),
            ("", ""),
        ],
    )
    def test_strips_filler_prefixes(self, raw: str, expected: str) -> None:
        assert normalize_item_request_text(raw) == expected


# ---------------------------------------------------------------------------
# ConfirmationDecisionHelper
# ---------------------------------------------------------------------------

class TestConfirmationDecisionHelper:
    def _helper(self) -> ConfirmationDecisionHelper:
        return ConfirmationDecisionHelper()

    def test_finalizing_returns_item_added_successfully(self) -> None:
        item = _simple_item()
        ctx = _ctx()
        ctx.pending_add_item = build_pending_add_item(item)

        helper = self._helper()
        with patch(
            "app.state_machine.handlers.item.add_item.confirmation_decision_helper.determine_next_add_item_step"
        ) as mock_step:
            from app.state_machine.handlers.item.add_item.add_item_flow import (
                AddItemCommand,
                ReadyToFinalize,
            )
            stub_command = AddItemCommand(
                item_id="burger_1",
                quantity=1,
                variant_id=None,
                sides={},
                side_variants={},
                modifiers={},
            )
            mock_step.return_value = ReadyToFinalize(command=stub_command)
            result = helper.build_handler_result(
                context=ctx,
                item=item,
                prefilled_summary="with Coke",
                prefill_feedback="",
                prefill_debug={},
            )

        assert result.response_key == "item_added_successfully"
        assert result.next_state == ConversationState.IDLE
        assert result.command == stub_command.to_dict()
        assert result.reset_context is True
        assert result.response_payload["item_name"] == "Zinger Burger"
        assert result.response_payload["prefilled_summary"] == "with Coke"

    def test_waiting_state_carries_prefill_info(self) -> None:
        item = _simple_item()
        ctx = _ctx()
        ctx.pending_add_item = build_pending_add_item(item)

        helper = self._helper()
        with patch(
            "app.state_machine.handlers.item.add_item.confirmation_decision_helper.determine_next_add_item_step"
        ) as mock_step:
            from app.state_machine.handlers.item.add_item.add_item_flow import AddItemNextStep
            mock_step.return_value = AddItemNextStep(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="ask_for_quantity",
                response_payload={"item_name": "Zinger Burger"},
            )
            result = helper.build_handler_result(
                context=ctx,
                item=item,
                prefilled_summary="",
                prefill_feedback="I couldn't find jelly.",
                prefill_debug={"bindings": []},
            )

        assert result.next_state == ConversationState.WAITING_FOR_QUANTITY
        assert result.response_key == "ask_for_quantity"
        assert result.response_payload["prefill_feedback"] == "I couldn't find jelly."
        assert result.response_payload["prefill_debug"] == {"bindings": []}

    def test_no_prefill_summary_key_omitted_from_waiting_payload(self) -> None:
        item = _simple_item()
        ctx = _ctx()
        ctx.pending_add_item = build_pending_add_item(item)

        helper = self._helper()
        with patch(
            "app.state_machine.handlers.item.add_item.confirmation_decision_helper.determine_next_add_item_step"
        ) as mock_step:
            from app.state_machine.handlers.item.add_item.add_item_flow import AddItemNextStep
            mock_step.return_value = AddItemNextStep(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="ask_for_quantity",
                response_payload={},
            )
            result = helper.build_handler_result(
                context=ctx,
                item=item,
                prefilled_summary="",
                prefill_feedback="",
                prefill_debug={},
            )

        assert "prefilled_summary" not in result.response_payload
        assert "prefill_feedback" not in result.response_payload


# ---------------------------------------------------------------------------
# ItemResolutionHandler
# ---------------------------------------------------------------------------

class TestItemResolutionHandler:
    def _handler(self, result: MenuQueryResult) -> ItemResolutionHandler:
        orchestrator = MagicMock(spec=PrefillOrchestrator)
        orchestrator.enter_add_flow_for_item.return_value = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            response_payload={"item_name": "Zinger Burger"},
        )
        return ItemResolutionHandler(
            menu_repo=_FakeRepo(result),
            prefill_orchestrator=orchestrator,
        )

    def test_single_item_calls_prefill_orchestrator(self) -> None:
        item = _simple_item()
        result = MenuQueryResult(type=MenuQueryType.ITEM, item=item)
        handler = self._handler(result)
        ctx = _ctx()

        out = handler.resolve_item_and_enter_flow(
            context=ctx,
            result=result,
            requested_item_text="zinger burger",
            original_user_text="zinger burger",
            slots=(),
        )

        assert out.response_key == "item_added_successfully"
        handler.prefill_orchestrator.enter_add_flow_for_item.assert_called_once()

    def test_category_single_item_calls_prefill_orchestrator(self) -> None:
        item = _simple_item()
        result = MenuQueryResult(type=MenuQueryType.CATEGORY_SINGLE_ITEM, items=[item])
        handler = self._handler(result)
        ctx = _ctx()

        out = handler.resolve_item_and_enter_flow(
            context=ctx,
            result=result,
            requested_item_text="burger",
            original_user_text="burger",
            slots=(),
        )

        handler.prefill_orchestrator.enter_add_flow_for_item.assert_called_once()
        assert out.response_key == "item_added_successfully"

    def test_category_returns_confirming_item(self) -> None:
        item = _simple_item()
        result = MenuQueryResult(
            type=MenuQueryType.CATEGORY,
            category_id="cat_1",
            category_name="Burgers",
            items=[item],
        )
        handler = self._handler(result)
        ctx = _ctx()

        out = handler.resolve_item_and_enter_flow(
            context=ctx,
            result=result,
            requested_item_text="burgers",
            original_user_text="burgers",
            slots=(),
        )

        assert out.next_state == ConversationState.CONFIRMING_ITEM
        assert out.response_key == "confirm_item_from_category"
        assert ctx.awaiting_confirmation_for["reason"] == "category_detected"

    def test_ambiguous_returns_confirming_item(self) -> None:
        item = _simple_item()
        result = MenuQueryResult(
            type=MenuQueryType.ITEM_AMBIGUOUS,
            matched_items=[item, item],
        )
        handler = self._handler(result)
        ctx = _ctx()

        out = handler.resolve_item_and_enter_flow(
            context=ctx,
            result=result,
            requested_item_text="burger",
            original_user_text="burger",
            slots=(),
        )

        assert out.next_state == ConversationState.CONFIRMING_ITEM
        assert out.response_key == "confirm_item_ambiguous"

    def test_not_found_returns_item_not_found(self) -> None:
        result = MenuQueryResult(type=MenuQueryType.NOT_FOUND)
        handler = self._handler(result)
        ctx = _ctx()

        out = handler.resolve_item_and_enter_flow(
            context=ctx,
            result=result,
            requested_item_text="unicorn sandwich",
            original_user_text="unicorn sandwich",
            slots=(),
        )

        assert out.next_state == ConversationState.IDLE
        assert out.response_key == "item_not_found"
        assert out.response_payload["query"] == "unicorn sandwich"

    @pytest.mark.parametrize(
        ("text", "modifier", "expected"),
        [
            ("cheese", "cheese", True),
            ("add cheese", "cheese", True),
            ("extra cheese", "cheese", True),
            ("no cheese", "cheese", True),
            ("without cheese", "cheese", True),
            ("chicken burger with cheese", "cheese", False),
            ("", "cheese", False),
            ("add", "", False),
        ],
    )
    def test_looks_like_modifier_only_request(
        self, text: str, modifier: str, expected: bool
    ) -> None:
        assert (
            ItemResolutionHandler.looks_like_modifier_only_request(
                normalized_user_text=normalize_text(text),
                modifier_value=modifier,
            )
            == expected
        )


# ---------------------------------------------------------------------------
# MultiItemQueueCoordinator
# ---------------------------------------------------------------------------

class TestMultiItemQueueCoordinator:
    def _make_segment(self, name: str, qty: int = 1) -> ParsedItemSegment:
        slot = SlotValue(name="ITEM", value=name, raw=name, start=None, end=None, confidence=1.0)
        return ParsedItemSegment(
            raw_text=name.lower(),
            item_slot_value=name,
            quantity=qty,
            slots=(slot,),
        )

    def _coordinator(self) -> MultiItemQueueCoordinator:
        item = _simple_item()
        fake_result = MenuQueryResult(type=MenuQueryType.ITEM, item=item)
        orchestrator = MagicMock(spec=PrefillOrchestrator)
        orchestrator.enter_add_flow_for_item.return_value = HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="item_added_successfully",
            response_payload={"item_name": item.name, "quantity": 1},
        )
        res_handler = ItemResolutionHandler(
            menu_repo=_FakeRepo(fake_result),
            prefill_orchestrator=orchestrator,
        )
        return MultiItemQueueCoordinator(
            menu_repo=_FakeRepo(fake_result),
            item_resolution_handler=res_handler,
        )

    def test_queues_remaining_items(self) -> None:
        coordinator = self._coordinator()
        ctx = _ctx()
        segments = [
            self._make_segment("Burger"),
            self._make_segment("Coke"),
            self._make_segment("Fries", qty=2),
        ]

        coordinator.handle(
            context=ctx,
            segments=segments,
            get_last_slots=lambda c: (),
        )

        assert len(ctx.pending_item_queue) == 2
        queued = list(ctx.pending_item_queue)
        assert queued[0].item_slot_value == "Coke"
        assert queued[1].item_slot_value == "Fries"
        assert queued[1].quantity == 2

    def test_result_has_multi_item_ack_payload(self) -> None:
        coordinator = self._coordinator()
        ctx = _ctx()
        segments = [self._make_segment("Burger"), self._make_segment("Coke")]

        result = coordinator.handle(
            context=ctx,
            segments=segments,
            get_last_slots=lambda c: (),
        )

        assert result.response_payload["multi_item_ack"] is True
        assert result.response_payload["queue_count"] == 1
        assert result.response_payload["current_item_name"] == "Burger"
        assert result.response_payload["queued_item_names"] == ["Coke"]

    def test_first_quantity_injected_into_context(self) -> None:
        coordinator = self._coordinator()
        ctx = _ctx()
        ctx.quantity = None
        segments = [self._make_segment("Burger", qty=3), self._make_segment("Coke")]

        coordinator.handle(
            context=ctx,
            segments=segments,
            get_last_slots=lambda c: (),
        )

        assert ctx.quantity == 3

    @pytest.mark.parametrize(
        ("qty", "name", "expected"),
        [
            (1, "Burger", "Burger"),
            (2, "Coke", "2 Coke"),
            (None, "Fries", "Fries"),
        ],
    )
    def test_build_segment_summary(self, qty, name, expected) -> None:
        seg = ParsedItemSegment(raw_text=name.lower(), item_slot_value=name, quantity=qty, slots=())
        assert MultiItemQueueCoordinator._build_segment_summary(seg) == expected


# ---------------------------------------------------------------------------
# PrefillOrchestrator — apply_prefill_result + missing_group_names
# ---------------------------------------------------------------------------

class TestPrefillOrchestrator:
    def _orchestrator(self) -> PrefillOrchestrator:
        return PrefillOrchestrator(
            capture_helper=PendingItemCaptureHelper(),
            prefill_engine=MultiGroupPrefillEngine(),
        )

    def test_apply_prefill_result_sets_variant(self) -> None:
        orchestrator = self._orchestrator()
        ctx = _ctx()
        item = _simple_item()
        ctx.pending_add_item = build_pending_add_item(item)

        from app.state_machine.handlers.item.add_item.multi_group_prefill import PrefillResult
        result = PrefillResult(
            variant_id="v1",
            side_selections={},
            modifier_selections={},
            feedback=[],
            unresolved_phrases=[],
            debug={},
        )
        orchestrator._apply_prefill_result(context=ctx, result=result)
        assert ctx.selected_variant_id == "v1"
        assert ctx.size_target is None

    def test_apply_prefill_result_sets_side_selections(self) -> None:
        orchestrator = self._orchestrator()
        ctx = _ctx()
        item = _simple_item(with_side=True)
        ctx.pending_add_item = build_pending_add_item(item)

        from app.state_machine.handlers.item.add_item.multi_group_prefill import PrefillResult
        result = PrefillResult(
            variant_id=None,
            side_selections={"drink_group": ["coke"]},
            modifier_selections={},
            feedback=[],
            unresolved_phrases=[],
            debug={},
        )
        orchestrator._apply_prefill_result(context=ctx, result=result)
        assert ctx.selected_side_groups["drink_group"] == ["coke"]

    def test_missing_group_names_includes_quantity_when_unset(self) -> None:
        orchestrator = self._orchestrator()
        ctx = ConversationContext()  # quantity not set
        item = _simple_item()
        ctx.pending_add_item = build_pending_add_item(item)

        missing = orchestrator._missing_group_names(ctx)
        assert "Quantity" in missing

    def test_missing_group_names_excludes_quantity_when_set(self) -> None:
        orchestrator = self._orchestrator()
        ctx = _ctx()  # quantity=1
        item = _simple_item()
        ctx.pending_add_item = build_pending_add_item(item)

        missing = orchestrator._missing_group_names(ctx)
        assert "Quantity" not in missing

    def test_missing_group_names_includes_optional_side_when_unresolved(self) -> None:
        orchestrator = self._orchestrator()
        ctx = _ctx()
        item = _simple_item(with_side=True)
        ctx.pending_add_item = build_pending_add_item(item)

        missing = orchestrator._missing_group_names(ctx)
        assert "Drink" in missing

    def test_missing_group_names_excludes_side_when_skipped(self) -> None:
        orchestrator = self._orchestrator()
        ctx = _ctx()
        item = _simple_item(with_side=True)
        ctx.pending_add_item = build_pending_add_item(item)
        ctx.skipped_side_groups.add("drink_group")

        missing = orchestrator._missing_group_names(ctx)
        assert "Drink" not in missing

    def test_build_prefilled_summary_with_modifier(self) -> None:
        orchestrator = self._orchestrator()
        ctx = _ctx()
        item = _simple_item(with_modifier=True)
        ctx.pending_add_item = build_pending_add_item(item)
        ctx.selected_modifier_groups["mod_group"] = [
            ModifierSelection(
                modifier_id="cheese",
                name="Cheese",
                action="add",
                instruction=None,
            )
        ]

        summary = orchestrator._build_prefilled_summary(ctx)
        assert summary == "with Cheese"

    def test_build_prefilled_summary_remove_modifier(self) -> None:
        orchestrator = self._orchestrator()
        ctx = _ctx()
        item = _simple_item(with_modifier=True)
        ctx.pending_add_item = build_pending_add_item(item)
        ctx.selected_modifier_groups["mod_group"] = [
            ModifierSelection(
                modifier_id="cheese",
                name="Cheese",
                action="remove",
                instruction=None,
            )
        ]

        summary = orchestrator._build_prefilled_summary(ctx)
        assert summary == "with no Cheese"

    def test_build_prefilled_summary_extra_modifier(self) -> None:
        orchestrator = self._orchestrator()
        ctx = _ctx()
        item = _simple_item(with_modifier=True)
        ctx.pending_add_item = build_pending_add_item(item)
        ctx.selected_modifier_groups["mod_group"] = [
            ModifierSelection(
                modifier_id="cheese",
                name="Cheese",
                action="add",
                instruction="extra",
            )
        ]

        summary = orchestrator._build_prefilled_summary(ctx)
        assert summary == "with extra Cheese"

    def test_build_prefilled_summary_empty_when_nothing_prefilled(self) -> None:
        orchestrator = self._orchestrator()
        ctx = _ctx()
        item = _simple_item()
        ctx.pending_add_item = build_pending_add_item(item)

        summary = orchestrator._build_prefilled_summary(ctx)
        assert summary == ""

    def test_collapse_unresolved_deduplicates_canonical_forms(self) -> None:
        item = _simple_item()
        pending = build_pending_add_item(item)
        result = PrefillOrchestrator._collapse_unresolved_for_feedback(
            ["jelly", "jelly", "extra jelly"],
            pending=pending,
        )
        assert len(result) == 1
        assert result[0] == "jelly"

    def test_collapse_unresolved_strips_known_tokens(self) -> None:
        item = _simple_item()
        pending = build_pending_add_item(item)
        result = PrefillOrchestrator._collapse_unresolved_for_feedback(
            ["and with"],
            pending=pending,
        )
        assert result == []
