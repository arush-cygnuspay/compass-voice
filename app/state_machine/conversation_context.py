# app/state_machine/conversation_context.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.pending_action import PendingAction
from app.nlu.nlu_result import NLUResult
from app.state_machine.conversation_state import ConversationState


@dataclass(slots=True)
class InterruptProposal:
    text: Optional[str] = None
    predicted_main_intent: Optional[str] = None
    predicted_sub_intent: Optional[str] = None

    def to_dict(self) -> Optional[dict]:
        if (
            self.text is None
            and self.predicted_main_intent is None
            and self.predicted_sub_intent is None
        ):
            return None

        return {
            "text": self.text,
            "predicted_main_intent": self.predicted_main_intent,
            "predicted_sub_intent": self.predicted_sub_intent,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["InterruptProposal"]:
        if not data:
            return None
        return cls(
            text=data.get("text"),
            predicted_main_intent=data.get("predicted_main_intent"),
            predicted_sub_intent=data.get("predicted_sub_intent"),
        )


@dataclass(slots=True)
class PendingVariantChoice:
    variant_id: str
    name: str
    normalized_name: str = ""


@dataclass(slots=True)
class PendingSideChoice:
    item_id: str
    name: str
    pricing_mode: str
    normalized_name: str = ""
    variants: list[PendingVariantChoice] = field(default_factory=list)


@dataclass(slots=True)
class PendingModifierChoice:
    modifier_id: str
    name: str
    group_id: str
    normalized_name: str = ""


@dataclass(slots=True)
class PendingSideGroup:
    group_id: str
    name: str
    is_required: bool
    min_selector: int
    max_selector: int
    choices: list[PendingSideChoice] = field(default_factory=list)


@dataclass(slots=True)
class PendingModifierGroup:
    group_id: str
    name: str
    is_required: bool
    min_selector: int
    max_selector: int
    choices: list[PendingModifierChoice] = field(default_factory=list)


@dataclass(slots=True)
class PendingAddItem:
    item_id: str
    item_name: str
    item_variants: list[PendingVariantChoice] = field(default_factory=list)
    side_groups: list[PendingSideGroup] = field(default_factory=list)
    modifier_groups: list[PendingModifierGroup] = field(default_factory=list)


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
        "variants": [_pending_variant_to_dict(v) for v in value.variants],
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
        "item_variants": [_pending_variant_to_dict(v) for v in value.item_variants],
        "side_groups": [_pending_side_group_to_dict(g) for g in value.side_groups],
        "modifier_groups": [_pending_modifier_group_to_dict(g) for g in value.modifier_groups],
    }


def _pending_variant_from_dict(data: dict) -> PendingVariantChoice:
    return PendingVariantChoice(
        variant_id=data["variant_id"],
        name=data["name"],
        normalized_name=data.get("normalized_name", ""),
    )


def _pending_side_choice_from_dict(data: dict) -> PendingSideChoice:
    return PendingSideChoice(
        item_id=data["item_id"],
        name=data["name"],
        pricing_mode=data["pricing_mode"],
        normalized_name=data.get("normalized_name", ""),
        variants=[_pending_variant_from_dict(v) for v in data.get("variants", [])],
    )


def _pending_modifier_choice_from_dict(data: dict) -> PendingModifierChoice:
    return PendingModifierChoice(
        modifier_id=data["modifier_id"],
        name=data["name"],
        group_id=data["group_id"],
        normalized_name=data.get("normalized_name", ""),
    )


def _pending_side_group_from_dict(data: dict) -> PendingSideGroup:
    return PendingSideGroup(
        group_id=data["group_id"],
        name=data["name"],
        is_required=bool(data["is_required"]),
        min_selector=int(data["min_selector"]),
        max_selector=int(data["max_selector"]),
        choices=[_pending_side_choice_from_dict(choice) for choice in data.get("choices", [])],
    )


def _pending_modifier_group_from_dict(data: dict) -> PendingModifierGroup:
    return PendingModifierGroup(
        group_id=data["group_id"],
        name=data["name"],
        is_required=bool(data["is_required"]),
        min_selector=int(data["min_selector"]),
        max_selector=int(data["max_selector"]),
        choices=[_pending_modifier_choice_from_dict(choice) for choice in data.get("choices", [])],
    )


def _pending_add_item_from_dict(data: dict) -> PendingAddItem:
    return PendingAddItem(
        item_id=data["item_id"],
        item_name=data["item_name"],
        item_variants=[_pending_variant_from_dict(v) for v in data.get("item_variants", [])],
        side_groups=[_pending_side_group_from_dict(g) for g in data.get("side_groups", [])],
        modifier_groups=[_pending_modifier_group_from_dict(g) for g in data.get("modifier_groups", [])],
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
    selected_modifier_groups: Dict[str, list[str]] = field(default_factory=dict)
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

    def reset_all(self) -> None:
        self.reset_task()
        self.last_user_text = None
        self.last_nlu = None
        self.last_intent_confidence = None
        self.last_slots = ()

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
            "selected_modifier_groups": self.selected_modifier_groups,
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
        ctx.selected_modifier_groups = dict(data.get("selected_modifier_groups", {}))
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

        return ctx