# app/state_machine/models/conversation_context.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.pending_action import PendingAction
from app.core.quantity_formatter import parse_item_quantity
from app.nlu.nlu_result import NLUResult
from app.state_machine.models.conversation_context_serde import (
    _deserialize_segment_slots,
    _modifier_selection_from_dict,
    _modifier_selection_to_dict,
    _pending_add_item_from_dict,
    _pending_add_item_to_dict,
)
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.delivery_address import DeliveryAddress
from app.state_machine.models.pending_item_models import (
    InterruptProposal,
    ModifierSelection,
    PendingAddItem,
    QueuedItemRequest,
)



@dataclass(slots=True)
class ConversationContext:
    current_item_id: Optional[str] = None
    current_item_name: Optional[str] = None
    candidate_item_id: Optional[str] = None

    selected_variant_id: Optional[str] = None
    size_target: Optional[Dict[str, Any]] = None

    current_side_group_index: int = 0
    selected_side_groups: Dict[str, list[str]] = field(default_factory=dict)
    skipped_side_groups: set[str] = field(default_factory=set)
    selected_side_variants: Dict[str, str] = field(default_factory=dict)

    pending_side_item_id: Optional[str] = None
    pending_side_item_name: Optional[str] = None
    pending_side_group_id: Optional[str] = None

    current_modifier_group_index: int = 0
    selected_modifier_groups: Dict[str, list[ModifierSelection]] = field(default_factory=dict)
    skipped_modifier_groups: set[str] = field(default_factory=set)

    quantity: Optional[int] = None

    pending_action: Optional[PendingAction] = None
    return_state: Optional[ConversationState] = None
    awaiting_confirmation_for: Optional[Dict[str, Any]] = None

    active_step_key: Optional[str] = None
    current_prompt_field: Optional[str] = None
    available_choices_kind: Optional[str] = None
    available_choices_values: tuple[str, ...] = ()

    awaiting_flow_confirmation: bool = False
    interrupt_proposal: Optional[InterruptProposal] = None
    resume_order_confirmation_after_edit: bool = False

    last_user_text: Optional[str] = None
    last_nlu: Optional[NLUResult] = None
    last_intent_confidence: Optional[float] = None
    last_slots: tuple[Any, ...] = ()

    pending_add_item: Optional[PendingAddItem] = None

    # Multi-item queue: items waiting to be processed after the current one completes.
    # Uses deque for O(1) popleft() during queue drain (avoids O(n) list.pop(0)).
    pending_item_queue: deque[QueuedItemRequest] = field(default_factory=deque)

    order_type: Optional[str] = None
    delivery_address_required: bool = False
    delivery_address_confirmed: bool = False
    onboarding_complete: bool = False

    # "phone" (mobile / cell) or "landline". If "landline" the call is
    # redirected to a human agent and we do not collect order details.
    caller_device_type: Optional[str] = None

    delivery_address: DeliveryAddress = field(default_factory=DeliveryAddress)

    # NEW: lightweight group collection metadata
    group_prompt_cursors: Dict[str, int] = field(default_factory=dict)
    group_multi_select_announced: set[str] = field(default_factory=set)

    # Per-field reprompt attempt counters. Owned here so any handler
    # can participate in a uniform escalation contract.
    reprompt_attempts: Dict[str, int] = field(default_factory=dict)

    # Per-item NOT_FOUND attempt counters, keyed by normalized query text.
    # NOT cleared by reset_item_scope — persists across disambiguation turns
    # within the same order session so the loop guard survives retries.
    item_not_found_attempts: Dict[str, int] = field(default_factory=dict)

    def bump_reprompt(self, field_name: str) -> int:
        """Increment and return the new attempt count for the field."""
        current = int(self.reprompt_attempts.get(field_name, 0)) + 1
        self.reprompt_attempts[field_name] = current
        return current

    def reset_reprompt(self, field_name: str) -> None:
        self.reprompt_attempts.pop(field_name, None)

    def reprompt_count(self, field_name: str) -> int:
        return int(self.reprompt_attempts.get(field_name, 0))

    def bump_not_found(self, query_key: str) -> int:
        """Increment and return the NOT_FOUND attempt count for the normalized item query."""
        current = int(self.item_not_found_attempts.get(query_key, 0)) + 1
        self.item_not_found_attempts[query_key] = current
        return current

    def not_found_count(self, query_key: str) -> int:
        return int(self.item_not_found_attempts.get(query_key, 0))

    def reset_not_found(self, query_key: str) -> None:
        self.item_not_found_attempts.pop(query_key, None)

    def set_last_nlu(self, user_text: str, nlu: NLUResult) -> None:
        self.last_user_text = user_text
        self.last_nlu = nlu
        self.last_intent_confidence = nlu.intent_confidence
        self.last_slots = nlu.slots

    def reset_item_scope(self) -> None:
        """Clear current item/add/modify flow data only.

        Covers everything a single add-item or modify-item turn owns:
        item identity, variants, sides, modifiers, quantity, pending
        actions, prompt fields, and flow-confirmation state.

        Does NOT touch: pending_item_queue, order_type, delivery_address,
        last_nlu/slots, or caller_device_type.
        """
        self.current_item_id = None
        self.current_item_name = None
        self.candidate_item_id = None

        self.selected_variant_id = None
        self.size_target = None

        self.current_side_group_index = 0
        self.selected_side_groups.clear()
        self.skipped_side_groups.clear()
        self.selected_side_variants.clear()
        self.pending_side_item_id = None
        self.pending_side_item_name = None
        self.pending_side_group_id = None

        self.current_modifier_group_index = 0
        self.selected_modifier_groups.clear()
        self.skipped_modifier_groups.clear()

        self.quantity = None

        self.pending_action = None
        self.return_state = None
        self.awaiting_confirmation_for = None
        self.active_step_key = None

        self.current_prompt_field = None
        self.available_choices_kind = None
        self.available_choices_values = ()

        self.awaiting_flow_confirmation = False
        self.interrupt_proposal = None
        self.resume_order_confirmation_after_edit = False
        self.pending_add_item = None

        self.group_prompt_cursors.clear()
        self.group_multi_select_announced.clear()
        self.reprompt_attempts.clear()

    def reset_order_scope(self) -> None:
        """Clear item scope AND the pending item queue.

        Use when the order is cancelled or restarted so that stale
        queued items from multi-item utterances cannot bleed into the
        next order.
        """
        self.reset_item_scope()
        self.pending_item_queue.clear()
        self.item_not_found_attempts.clear()

    def reset_session_scope(self) -> None:
        """Full context reset — use only on order completion or session end.

        Clears everything including NLU history, delivery address, and the
        pending item queue.
        """
        self.reset_item_scope()
        self.pending_item_queue.clear()
        self.last_user_text = None
        self.last_nlu = None
        self.last_intent_confidence = None
        self.last_slots = ()
        self.delivery_address = DeliveryAddress()

    # ── Backward-compatible wrappers (do not remove — test code uses these) ──

    def reset_task(self) -> None:
        self.reset_item_scope()

    def clear_item_queue(self) -> None:
        self.pending_item_queue.clear()

    def reset_all(self) -> None:
        self.reset_session_scope()

    def reset(self) -> None:
        self.reset_item_scope()

    def to_dict(self) -> dict:
        return {
            "current_item_id": self.current_item_id,
            "current_item_name": self.current_item_name,
            "candidate_item_id": self.candidate_item_id,
            "selected_variant_id": self.selected_variant_id,
            "size_target": self.size_target,
            "current_side_group_index": self.current_side_group_index,
            "selected_side_groups": self.selected_side_groups,
            "skipped_side_groups": list(self.skipped_side_groups),
            "selected_side_variants": self.selected_side_variants,
            "pending_side_item_id": self.pending_side_item_id,
            "pending_side_item_name": self.pending_side_item_name,
            "pending_side_group_id": self.pending_side_group_id,
            "current_modifier_group_index": self.current_modifier_group_index,
            "selected_modifier_groups": {
                group_id: [_modifier_selection_to_dict(sel) for sel in selections]
                for group_id, selections in self.selected_modifier_groups.items()
            },
            "skipped_modifier_groups": list(self.skipped_modifier_groups),
            "quantity": self.quantity,
            "pending_action": self.pending_action.value if self.pending_action else None,
            "return_state": self.return_state.value if self.return_state else None,
            "awaiting_confirmation_for": self.awaiting_confirmation_for,
            "active_step_key": self.active_step_key,
            "current_prompt_field": self.current_prompt_field,
            "available_choices_kind": self.available_choices_kind,
            "available_choices_values": list(self.available_choices_values),
            "awaiting_flow_confirmation": self.awaiting_flow_confirmation,
            "interrupt_proposal": self.interrupt_proposal.to_dict() if self.interrupt_proposal else None,
            "resume_order_confirmation_after_edit": self.resume_order_confirmation_after_edit,
            "pending_add_item": _pending_add_item_to_dict(self.pending_add_item) if self.pending_add_item else None,
            "pending_item_queue": [
                {
                    "raw_text": item.raw_text,
                    "item_slot_value": item.item_slot_value,
                    "quantity": item.quantity,
                    "acknowledged": item.acknowledged,
                    "segment_slots": [
                        {
                            "name": getattr(s, "name", ""),
                            "value": getattr(s, "value", ""),
                            "raw": getattr(s, "raw", ""),
                            "start": getattr(s, "start", None),
                            "end": getattr(s, "end", None),
                            "confidence": getattr(s, "confidence", None),
                        }
                        for s in (item.segment_slots or ())
                    ],
                }
                for item in self.pending_item_queue
            ],
            "order_type": self.order_type,
            "delivery_address_required": self.delivery_address_required,
            "delivery_address_confirmed": self.delivery_address_confirmed,
            "onboarding_complete": self.onboarding_complete,
            "caller_device_type": self.caller_device_type,
            "delivery_address": self.delivery_address.to_dict(),
            "group_prompt_cursors": dict(self.group_prompt_cursors),
            "group_multi_select_announced": list(self.group_multi_select_announced),
            "reprompt_attempts": dict(self.reprompt_attempts),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ConversationContext":
        data = data or {}
        ctx = cls()

        ctx.current_item_id = data.get("current_item_id")
        ctx.current_item_name = data.get("current_item_name")
        ctx.candidate_item_id = data.get("candidate_item_id")

        ctx.selected_variant_id = data.get("selected_variant_id")
        ctx.size_target = dict(data["size_target"]) if data.get("size_target") else None

        ctx.current_side_group_index = data.get("current_side_group_index", 0)
        ctx.selected_side_groups = dict(data.get("selected_side_groups", {}))
        ctx.skipped_side_groups = set(data.get("skipped_side_groups", []))
        ctx.selected_side_variants = dict(data.get("selected_side_variants", {}))
        ctx.pending_side_item_id = data.get("pending_side_item_id")
        ctx.pending_side_item_name = data.get("pending_side_item_name")
        ctx.pending_side_group_id = data.get("pending_side_group_id")

        ctx.current_modifier_group_index = data.get("current_modifier_group_index", 0)
        raw_selected_modifier_groups = data.get("selected_modifier_groups", {})
        ctx.selected_modifier_groups = {
            group_id: [_modifier_selection_from_dict(sel) for sel in selections]
            for group_id, selections in raw_selected_modifier_groups.items()
        }
        ctx.skipped_modifier_groups = set(data.get("skipped_modifier_groups", []))

        raw_qty = data.get("quantity")
        ctx.quantity = parse_item_quantity(raw_qty) if raw_qty is not None else None

        pending_action = data.get("pending_action")
        ctx.pending_action = PendingAction(pending_action) if pending_action else None

        return_state = data.get("return_state")
        ctx.return_state = ConversationState(return_state) if return_state else None

        awaiting_confirmation_for = data.get("awaiting_confirmation_for")
        ctx.awaiting_confirmation_for = dict(awaiting_confirmation_for) if awaiting_confirmation_for else None

        ctx.active_step_key = data.get("active_step_key")
        ctx.current_prompt_field = data.get("current_prompt_field")
        ctx.available_choices_kind = data.get("available_choices_kind")
        ctx.available_choices_values = tuple(data.get("available_choices_values", []))
        ctx.awaiting_flow_confirmation = data.get("awaiting_flow_confirmation", False)
        ctx.interrupt_proposal = InterruptProposal.from_dict(data.get("interrupt_proposal"))
        ctx.resume_order_confirmation_after_edit = bool(
            data.get("resume_order_confirmation_after_edit", False)
        )

        pending_add_item = data.get("pending_add_item")
        ctx.pending_add_item = _pending_add_item_from_dict(pending_add_item) if pending_add_item else None

        ctx.pending_item_queue = deque(
            QueuedItemRequest(
                raw_text=item_data.get("raw_text", ""),
                item_slot_value=item_data.get("item_slot_value"),
                quantity=item_data.get("quantity"),
                acknowledged=bool(item_data.get("acknowledged", False)),
                segment_slots=_deserialize_segment_slots(item_data.get("segment_slots", [])),
            )
            for item_data in data.get("pending_item_queue", [])
            if item_data.get("raw_text")
        )

        ctx.order_type = data.get("order_type")
        ctx.delivery_address_required = bool(data.get("delivery_address_required", False))
        ctx.delivery_address_confirmed = bool(data.get("delivery_address_confirmed", False))
        ctx.onboarding_complete = bool(data.get("onboarding_complete", False))
        ctx.caller_device_type = data.get("caller_device_type")

        legacy_delivery_address = {
            "area": data.get("delivery_area"),
            "area_serviceable": data.get("delivery_area_serviceable"),
            "customer_phone_number": data.get("customer_phone_number"),
            "order_number": data.get("order_number"),
            "payment_link": data.get("payment_link"),
        }

        delivery_address_data = data.get("delivery_address")
        if delivery_address_data:
            ctx.delivery_address = DeliveryAddress.from_dict(delivery_address_data)
        else:
            ctx.delivery_address = DeliveryAddress.from_dict(legacy_delivery_address)

        ctx.group_prompt_cursors = dict(data.get("group_prompt_cursors", {}))
        ctx.group_multi_select_announced = set(data.get("group_multi_select_announced", []))
        ctx.reprompt_attempts = {
            str(k): int(v or 0)
            for k, v in (data.get("reprompt_attempts") or {}).items()
        }

        return ctx

