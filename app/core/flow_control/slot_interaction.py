# app/core/flow_control/slot_interaction.py

from enum import Enum, auto


class SlotInteraction(Enum):
    ASK_OPTIONS = auto()
    SKIP = auto()
    CANCEL = auto()
    NONE = auto()
