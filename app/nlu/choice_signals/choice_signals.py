# app/nlu/choice_signals.py
from enum import Enum, auto

class ChoiceSignal(Enum):
    ASK_OPTIONS = auto()
    DENY = auto()
    CONFIRM = auto()
    CANCEL = auto()
    NONE = auto()
