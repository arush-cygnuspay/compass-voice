# app/state_machine/models/conversation_context.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.pending_action import PendingAction
from app.nlu.nlu_result import NLUResult
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.models.delivery_address import DeliveryAddress
from app.state_machine.models.pending_item_models import PendingVariantChoice, PendingSideChoice, PendingModifierChoice, \
    PendingSideGroup, PendingModifierGroup, PendingAddItem, InterruptProposal, ModifierSelection


def _append_bucket_value(mapping: dict[str, list[Any]], key: str, value: Any) -> None:
    bucket = mapping.get(key)
    if bucket is None:
        mapping[key] = [value]
        return
    bucket.append(value)


def _index_variants(
    variants: list[PendingVariantChoice],
) -> tuple[
    dict[str, PendingVariantChoice],
    dict[str, PendingVariantChoice],
    tuple[str, ...],
    tuple[str, ...],
]:
    by_id: dict[str, PendingVariantChoice] = {}
    by_normalized_name: dict[str, PendingVariantChoice] = {}
    names: list[str] = []

    for variant in variants:
        by_id[variant.variant_id] = variant

        if variant.normalized_name and variant.normalized_name not in by_normalized_name:
            by_normalized_name[variant.normalized_name] = variant

        if variant.name:
            names.append(variant.name)

    name_tuple = tuple(names)
    return by_id, by_normalized_name, name_tuple, name_tuple[:4]


def _pending_variant_to_dict(value: PendingVariantChoice) -> dict:
    return {
        "variant_id": value.variant_id,
        "name": value.name,
        "normalized_name": value.normalized_name,
    }


def _pending_side_choice_to_dict(value: PendingSideChoice) -> dict:
    return {
        "item_id": value.item_id,
        "name": value.name,
        "pricing_mode": value.pricing_mode,
        "normalized_name": value.normalized_name,
        "variants": [_pending_variant_to_dict(variant) for variant in value.variants],
    }


def _pending_modifier_choice_to_dict(value: PendingModifierChoice) -> dict:
    return {
        "modifier_id": value.modifier_id,
        "name": value.name,
        "group_id": value.group_id,
        "normalized_name": value.normalized_name,
    }


def _pending_side_group_to_dict(value: PendingSideGroup) -> dict:
    return {
        "group_id": value.group_id,
        "name": value.name,
        "is_required": value.is_required,
        "min_selector": value.min_selector,
        "max_selector": value.max_selector,
        "choices": [_pending_side_choice_to_dict(choice) for choice in value.choices],
    }


def _pending_modifier_group_to_dict(value: PendingModifierGroup) -> dict:
    return {
        "group_id": value.group_id,
        "name": value.name,
        "is_required": value.is_required,
        "min_selector": value.min_selector,
        "max_selector": value.max_selector,
        "choices": [_pending_modifier_choice_to_dict(choice) for choice in value.choices],
    }


def _pending_add_item_to_dict(value: PendingAddItem) -> dict:
    return {
        "item_id": value.item_id,
        "item_name": value.item_name,
        "item_variants": [_pending_variant_to_dict(variant) for variant in value.item_variants],
        "side_groups": [_pending_side_group_to_dict(group) for group in value.side_groups],
        "modifier_groups": [_pending_modifier_group_to_dict(group) for group in value.modifier_groups],
    }


def _pending_variant_from_dict(data: dict) -> PendingVariantChoice:
    name = data["name"]
    normalized_name = data.get("normalized_name") or normalize_text(name)

    return PendingVariantChoice(
        variant_id=data["variant_id"],
        name=name,
        normalized_name=normalized_name,
    )


def _pending_side_choice_from_dict(data: dict) -> PendingSideChoice:
    variants = [_pending_variant_from_dict(variant) for variant in data.get("variants", [])]
    (
        variants_by_id,
        variants_by_normalized_name,
        variant_names,
        top_variant_names,
    ) = _index_variants(variants)

    name = data["name"]
    normalized_name = data.get("normalized_name") or normalize_text(name)

    return PendingSideChoice(
        item_id=data["item_id"],
        name=name,
        pricing_mode=data["pricing_mode"],
        normalized_name=normalized_name,
        variants=variants,
        variants_by_id=variants_by_id,
        variants_by_normalized_name=variants_by_normalized_name,
        variant_names=variant_names,
        top_variant_names=top_variant_names,
    )


def _pending_modifier_choice_from_dict(data: dict) -> PendingModifierChoice:
    name = data["name"]
    normalized_name = data.get("normalized_name") or normalize_text(name)

    return PendingModifierChoice(
        modifier_id=data["modifier_id"],
        name=name,
        group_id=data["group_id"],
        normalized_name=normalized_name,
    )


def _pending_side_group_from_dict(data: dict) -> PendingSideGroup:
    choices = [_pending_side_choice_from_dict(choice) for choice in data.get("choices", [])]

    choices_by_item_id: dict[str, PendingSideChoice] = {}
    choices_by_normalized_name: dict[str, list[PendingSideChoice]] = {}
    choice_names: list[str] = []
    normalized_choice_names: list[str] = []

    for choice in choices:
        choices_by_item_id[choice.item_id] = choice

        if choice.normalized_name:
            _append_bucket_value(choices_by_normalized_name, choice.normalized_name, choice)
            normalized_choice_names.append(choice.normalized_name)

        if choice.name:
            choice_names.append(choice.name)

    return PendingSideGroup(
        group_id=data["group_id"],
        name=data["name"],
        is_required=bool(data["is_required"]),
        min_selector=int(data["min_selector"]),
        max_selector=int(data["max_selector"]),
        choices=choices,
        choices_by_item_id=choices_by_item_id,
        choices_by_normalized_name=choices_by_normalized_name,
        choice_names=tuple(choice_names),
        normalized_choice_names=tuple(normalized_choice_names),
        top_choice_names=tuple(choice_names[:3]),
    )


def _pending_modifier_group_from_dict(data: dict) -> PendingModifierGroup:
    choices = [_pending_modifier_choice_from_dict(choice) for choice in data.get("choices", [])]

    choices_by_modifier_id: dict[str, PendingModifierChoice] = {}
    choices_by_normalized_name: dict[str, list[PendingModifierChoice]] = {}
    choice_names: list[str] = []
    normalized_choice_names: list[str] = []

    for choice in choices:
        choices_by_modifier_id[choice.modifier_id] = choice

        if choice.normalized_name:
            _append_bucket_value(choices_by_normalized_name, choice.normalized_name, choice)
            normalized_choice_names.append(choice.normalized_name)

        if choice.name:
            choice_names.append(choice.name)

    return PendingModifierGroup(
        group_id=data["group_id"],
        name=data["name"],
        is_required=bool(data["is_required"]),
        min_selector=int(data["min_selector"]),
        max_selector=int(data["max_selector"]),
        choices=choices,
        choices_by_modifier_id=choices_by_modifier_id,
        choices_by_normalized_name=choices_by_normalized_name,
        choice_names=tuple(choice_names),
        normalized_choice_names=tuple(normalized_choice_names),
        top_choice_names=tuple(choice_names[:4]),
    )


def _pending_add_item_from_dict(data: dict) -> PendingAddItem:
    item_variants = [_pending_variant_from_dict(variant) for variant in data.get("item_variants", [])]
    (
        item_variants_by_id,
        item_variants_by_normalized_name,
        item_variant_names,
        top_item_variant_names,
    ) = _index_variants(item_variants)

    side_groups = [_pending_side_group_from_dict(group) for group in data.get("side_groups", [])]
    modifier_groups = [_pending_modifier_group_from_dict(group) for group in data.get("modifier_groups", [])]

    side_groups_by_id: dict[str, PendingSideGroup] = {}
    side_choice_by_item_id: dict[str, PendingSideChoice] = {}
    for group in side_groups:
        side_groups_by_id[group.group_id] = group
        for choice in group.choices:
            side_choice_by_item_id[choice.item_id] = choice

    modifier_groups_by_id: dict[str, PendingModifierGroup] = {}
    modifier_choice_by_id: dict[str, PendingModifierChoice] = {}
    for group in modifier_groups:
        modifier_groups_by_id[group.group_id] = group
        for choice in group.choices:
            modifier_choice_by_id[choice.modifier_id] = choice

    return PendingAddItem(
        item_id=data["item_id"],
        item_name=data["item_name"],
        item_variants=item_variants,
        side_groups=side_groups,
        modifier_groups=modifier_groups,
        item_variants_by_id=item_variants_by_id,
        item_variants_by_normalized_name=item_variants_by_normalized_name,
        item_variant_names=item_variant_names,
        top_item_variant_names=top_item_variant_names,
        side_groups_by_id=side_groups_by_id,
        side_choice_by_item_id=side_choice_by_item_id,
        modifier_groups_by_id=modifier_groups_by_id,
        modifier_choice_by_id=modifier_choice_by_id,
    )


def _modifier_selection_to_dict(value: ModifierSelection) -> dict:
    return {
        "modifier_id": value.modifier_id,
        "name": value.name,
        "action": value.action,
        "instruction": value.instruction,
    }

def _modifier_selection_from_dict(data: dict) -> ModifierSelection:
    return ModifierSelection(
        modifier_id=data["modifier_id"],
        name=data["name"],
        action=data.get("action", "add"),
        instruction=data.get("instruction"),
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

    last_user_text: Optional[str] = None
    last_nlu: Optional[NLUResult] = None
    last_intent_confidence: Optional[float] = None
    last_slots: tuple[Any, ...] = ()

    pending_add_item: Optional[PendingAddItem] = None

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

    def set_last_nlu(self, user_text: str, nlu: NLUResult) -> None:
        self.last_user_text = user_text
        self.last_nlu = nlu
        self.last_intent_confidence = nlu.intent_confidence
        self.last_slots = nlu.slots

    def reset_task(self) -> None:
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
        self.pending_add_item = None

        self.group_prompt_cursors.clear()
        self.group_multi_select_announced.clear()

    def reset_all(self) -> None:
        self.reset_task()
        self.last_user_text = None
        self.last_nlu = None
        self.last_intent_confidence = None
        self.last_slots = ()
        self.delivery_address = DeliveryAddress()

    def reset(self) -> None:
        self.reset_task()

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
            "pending_add_item": _pending_add_item_to_dict(self.pending_add_item) if self.pending_add_item else None,
            "order_type": self.order_type,
            "delivery_address_required": self.delivery_address_required,
            "delivery_address_confirmed": self.delivery_address_confirmed,
            "onboarding_complete": self.onboarding_complete,
            "caller_device_type": self.caller_device_type,
            "delivery_address": self.delivery_address.to_dict(),
            "group_prompt_cursors": dict(self.group_prompt_cursors),
            "group_multi_select_announced": list(self.group_multi_select_announced),
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

        ctx.quantity = data.get("quantity")

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

        pending_add_item = data.get("pending_add_item")
        ctx.pending_add_item = _pending_add_item_from_dict(pending_add_item) if pending_add_item else None

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

        return ctx

