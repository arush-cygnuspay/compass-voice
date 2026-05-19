# app/state_machine/models/pending_item_models.py

from __future__ import annotations

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
    match_texts: tuple[str, ...] = ()
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
    # Mirrors SideGroup.allow_duplicate_selections.
    allow_duplicate_selections: bool = True
    # Mirrors SideGroup.is_suggested_addon.
    is_suggested_addon: bool = False
    # Mirrors SideGroup.default_item_ids.
    default_item_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class PendingModifierChoice:
    modifier_id: str
    name: str
    group_id: str
    normalized_name: str
    match_texts: tuple[str, ...] = ()


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
    # Normalized aliases and voice labels from the resolved MenuItem.
    # Used by prefill and feedback logic to suppress the alias/compact forms
    # that the user spoke (e.g. "cheeseburger") from "I couldn't find" output.
    item_aliases: tuple[str, ...] = ()
    item_voice_labels: tuple[str, ...] = ()


@dataclass(slots=True)
class ModifierSelection:
    modifier_id: str
    name: str
    action: str = "add"         # add | remove
    instruction: Optional[str] = None  # extra | less | light | on_side | None


@dataclass(slots=True)
class QueuedItemRequest:
    """
    A lightweight snapshot of a parsed item from a multi-item utterance.

    Stored in the item queue until its turn to enter the add-item flow.
    The raw_text is re-fed to AddItemHandler when dequeued.
    """
    raw_text: str                       # the text segment for this item
    item_slot_value: Optional[str] = None  # ITEM slot value if detected
    quantity: Optional[int] = None      # quantity if detected
    acknowledged: bool = False          # whether user has been told we heard this item
    # Preserved NLU slots from the multi-item parser so queued items
    # retain their modifier/side/size context when dequeued.
    segment_slots: tuple = ()           # tuple[SlotValue, ...]


@dataclass(slots=True, frozen=True)
class StagedSide:
    """A side/drink pre-resolved from a compound utterance, preserved for drain."""
    name: str
    variant_label: str | None = None   # e.g. "small", "large"
    quantity: int = 1
    group_hint: str | None = None      # side group name hint for faster matching


@dataclass(slots=True, frozen=True)
class StagedModifier:
    """A modifier pre-resolved from a compound utterance."""
    name: str
    operation: str = "add"             # add | remove | extra | light


@dataclass(slots=True)
class StagedItemPlan:
    """Structured snapshot of an items[1..N] entry from a multi-item plan.

    Stored in context.staged_item_queue and drained by ItemQueueService
    via PrefillOrchestrator — no re-entry into GPT/local planner.
    """
    item_id: str                                   # empty str = resolve by name at drain
    item_name: str
    quantity: int = 1
    variant_id: str | None = None                  # pre-resolved variant id
    variant_label: str | None = None               # e.g. "large", "6 piece"
    requested_sides: tuple[StagedSide, ...] = ()
    requested_modifiers: tuple[StagedModifier, ...] = ()
    raw_span: str = ""                             # original text span, for logging
    plan_source: str = ""                          # "gpt_planner" | "local_planner" | etc.
