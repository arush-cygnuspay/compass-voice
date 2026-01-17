# app/state_machine/context.py

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class ConversationContext:
    """
    Holds all transient data related to the current conversational task.
    This context aligns with the menu domain model but does NOT duplicate it.
    """

    # === Item focus ===
    # Canonical menu item currently being worked on
    current_item_id: Optional[str] = None
    current_item_name: Optional[str] = None  # optional, UX-only

    # Candidate item awaiting confirmation
    candidate_item_id: Optional[str] = None

    # === Pricing / variants ===
    selected_variant_id: Optional[str] = None   # e.g. small / medium / large

    # === Side selections (by group) ===
    # Which side group is currently being asked
    current_side_group_index: int = 0

    # group_id -> list[item_id]
    selected_side_groups: Dict[str, List[str]] = field(default_factory=dict)

    # === Modifier selections (by group) ===
    current_modifier_group_index: int = 0

    # group_id -> list[modifier_id]
    selected_modifier_groups: Dict[str, List[str]] = field(default_factory=dict)

    # === Quantity ===
    quantity: Optional[int] = None

    # === Flow control ===
    pending_action: Optional[str] = None  # add | modify | remove

    # What exactly is awaiting confirmation
    awaiting_confirmation_for: Optional[Dict[str, Any]] = None
    # example:
    # {
    #   "type": "side" | "modifier" | "item" | "variant" | "quantity",
    #   "group_id": "...",
    #   "value_id": "...",
    # }

    skipped_modifier_groups: set[str] = field(default_factory=set) # in case a modifier group can be skipped

    def reset(self) -> None:
        """
        Resets the context after task completion or cancellation.
        """
        self.current_item_id = None
        self.current_item_name = None
        self.candidate_item_id = None
        self.selected_variant_id = None
        self.current_side_group_index = 0
        self.current_modifier_group_index = 0
        self.selected_side_groups.clear()
        self.selected_modifier_groups.clear()
        self.quantity = None
        self.pending_action = None
        self.awaiting_confirmation_for = None
