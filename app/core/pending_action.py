# app/core/pending_action.py

from enum import Enum


class PendingAction(str, Enum):
    ADD_ITEM = "add_item"
    MODIFY_ITEM = "modify_item"
    REMOVE_ITEM = "remove_item"