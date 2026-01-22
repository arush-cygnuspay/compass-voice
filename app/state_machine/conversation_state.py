# app/state_machine/conversation_state.py

from enum import Enum, auto


class ConversationState(Enum):
    """
    Represents the single active conversational state of the system.
    At any given time, the system MUST be in exactly one state.
    """

    WAITING_FOR_PAYMENT = auto()
    GREETING = auto()                 # Initial welcome message
    IDLE = auto()                     # Neutral, no active task

    ADDING_ITEM = auto()              # User wants to add an item
    CONFIRMING_ITEM = auto()          # Confirm detected item with user

    WAITING_FOR_SIDE = auto()         # Required/optional side selection
    WAITING_FOR_MODIFIER = auto()     # Modifier selection
    WAITING_FOR_SIZE = auto()         # Size selection
    WAITING_FOR_QUANTITY = auto()     # Quantity clarification

    MODIFYING_ITEM = auto()           # Modifying an existing cart item
    REMOVING_ITEM = auto()            # Removing item or part of item

    CONFIRMING = auto()                # general confirmation depends on the context and state

    SHOWING_CART = auto()             # Showing cart content
    SHOWING_TOTAL = auto()            # Showing cart total

    CONFIRMING_ORDER = auto()         # Order confirmation state

    CANCELLATION_CONFIRMATION = auto()# Asking user to cancel current flow
    ERROR_RECOVERY = auto()            # Fallback / repair state

