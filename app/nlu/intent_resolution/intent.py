# app/nlu/intent.py

from enum import Enum, auto


class Intent(Enum):
    ADD_ITEM = auto()
    MODIFY_ITEM = auto()
    REMOVE_ITEM = auto()

    SHOW_MENU = auto()
    SHOW_CART = auto()
    SHOW_TOTAL = auto()

    CONFIRM = auto()     # yes / correct / okay
    DENY = auto()        # no / wrong
    CANCEL = auto()      # cancel / stop / forget it

    UNKNOWN = auto()
