# app/state_machine/models/pending_item_models.py

from dataclasses import dataclass, field
from typing import Optional


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
    normalized_name: str


@dataclass(slots=True)
class PendingSideChoice:
    item_id: str
    name: str
    pricing_mode: str
    normalized_name: str
    variants: list[PendingVariantChoice] = field(default_factory=list)
    variants_by_id: dict[str, PendingVariantChoice] = field(default_factory=dict)
    variants_by_normalized_name: dict[str, PendingVariantChoice] = field(default_factory=dict)
    variant_names: tuple[str, ...] = ()
    top_variant_names: tuple[str, ...] = ()


@dataclass(slots=True)
class PendingSideGroup:
    group_id: str
    name: str
    is_required: bool
    min_selector: int
    max_selector: int
    choices: list[PendingSideChoice] = field(default_factory=list)
    choices_by_item_id: dict[str, PendingSideChoice] = field(default_factory=dict)
    choices_by_normalized_name: dict[str, list[PendingSideChoice]] = field(default_factory=dict)
    choice_names: tuple[str, ...] = ()
    normalized_choice_names: tuple[str, ...] = ()
    top_choice_names: tuple[str, ...] = ()


@dataclass(slots=True)
class PendingModifierChoice:
    modifier_id: str
    name: str
    group_id: str
    normalized_name: str


@dataclass(slots=True)
class PendingModifierGroup:
    group_id: str
    name: str
    is_required: bool
    min_selector: int
    max_selector: int
    choices: list[PendingModifierChoice] = field(default_factory=list)
    choices_by_modifier_id: dict[str, PendingModifierChoice] = field(default_factory=dict)
    choices_by_normalized_name: dict[str, list[PendingModifierChoice]] = field(default_factory=dict)
    choice_names: tuple[str, ...] = ()
    normalized_choice_names: tuple[str, ...] = ()
    top_choice_names: tuple[str, ...] = ()


@dataclass(slots=True)
class PendingAddItem:
    item_id: str
    item_name: str
    item_variants: list[PendingVariantChoice] = field(default_factory=list)
    side_groups: list[PendingSideGroup] = field(default_factory=list)
    modifier_groups: list[PendingModifierGroup] = field(default_factory=list)
    item_variants_by_id: dict[str, PendingVariantChoice] = field(default_factory=dict)
    item_variants_by_normalized_name: dict[str, PendingVariantChoice] = field(default_factory=dict)
    item_variant_names: tuple[str, ...] = ()
    top_item_variant_names: tuple[str, ...] = ()
    side_groups_by_id: dict[str, PendingSideGroup] = field(default_factory=dict)
    side_choice_by_item_id: dict[str, PendingSideChoice] = field(default_factory=dict)
    modifier_groups_by_id: dict[str, PendingModifierGroup] = field(default_factory=dict)
    modifier_choice_by_id: dict[str, PendingModifierChoice] = field(default_factory=dict)
