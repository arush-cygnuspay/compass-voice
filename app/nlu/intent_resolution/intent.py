# app/nlu/intent_resolution/intent.py

from enum import Enum, auto


class Intent(Enum):
    ADD_ITEM = auto()
    MODIFY_ITEM = auto()
    REMOVE_ITEM = auto()

    END_ADDING = auto()  # user indicates no more items -> "that's all", "nothing else"
    START_ORDER = auto()  # "checkout", "place order"
    ORDER_STATUS = auto()  # "is my order placed?"

    PAYMENT_REQUEST = auto()  # user requested payment
    PAYMENT_DONE = auto()  # "I have paid" -> user intends to tell he has paid

    SHOW_MENU = auto()
    SHOW_CART = auto()
    SHOW_TOTAL = auto()
    CLEAR_CART = auto()

    CONFIRM = auto()     # yes / correct / okay
    DENY = auto()        # no / wrong
    CANCEL = auto()      # cancel / stop / forget it

    META_CLARIFY = auto()  # "right?", "correct?"

    UNKNOWN = auto()
