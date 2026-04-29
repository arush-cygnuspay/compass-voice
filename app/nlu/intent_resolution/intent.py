# app/nlu/intent_resolution/intent.py

from __future__ import annotations

from enum import Enum


class Intent(Enum):
    """
    Canonical application intents used for routing and flow policy.

    Notes:
    - ML sub-intents map into these canonical intents via intent_mapping.py
    - Some intents are produced by ML directly
    - Some intents are internal normalized intents (e.g. START_ORDER, END_ADDING)
    """

    ADD_ITEM = "add_item"
    REMOVE_ITEM = "remove_item"
    MODIFY_ITEM = "modify_item"
    CLEAR_CART = "clear_cart"

    CHANGE_QUANTITY = "change_quantity"
    REPLACE_ITEM = "replace_item"
    UNDO_LAST = "undo_last"

    SHOW_CART = "show_cart"
    SHOW_TOTAL = "show_total"
    CART_ITEM_QUERY = "cart_item_query"

    SHOW_MENU = "show_menu"
    ASK_MENU_INFO = "ask_menu_info"
    ASK_PRICE = "ask_price"

    ASK_OPTIONS = "ask_options"
    AVAILABILITY_QUERY = "availability_query"
    RECOMMENDATION_QUERY = "recommendation_query"

    BROWSE_MENU = "browse_menu"
    BROWSE_CATEGORY = "browse_category"
    ASK_ITEM_INFO = "ask_item_info"

    START_ORDER = "start_order"
    CONFIRM_ORDER = "confirm_order"
    FINISH_ORDER = "finish_order"
    REVIEW_ORDER = "review_order"
    CANCEL_ORDER = "cancel_order"
    END_ADDING = "end_adding"
    CHECKOUT = "checkout"

    ORDER_STATUS = "order_status"
    ORDER_ERROR_STATUS = "order_error_status"
    ORDER_PLACEMENT_STATUS = "order_placement_status"
    ORDER_PROCESSING_STATUS = "order_processing_status"
    ORDER_STATUS_GENERAL = "order_status_general"

    PAYMENT_REQUEST = "payment_request"
    PAYMENT_METHODS_QUERY = "payment_methods_query"
    PAYMENT_STATUS = "payment_status"
    PAYMENT_DONE = "payment_done"

    CONFIRM = "confirm"
    DENY = "deny"
    CANCEL = "cancel"
    META_CLARIFY = "meta_clarify"
    AFFIRM = "affirm"

    REQUEST_AGENT = "request_agent"

    GREETING = "greeting"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"

    UNKNOWN = "unknown"