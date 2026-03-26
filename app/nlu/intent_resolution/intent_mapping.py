# app/nlu/intent_resolution/intent_mapping.py

from app.nlu.intent_resolution.intent import Intent


SUB_INTENT_TO_INTENT: dict[str, Intent] = {
    # --------------------------------------------------
    # Cart mutations
    # --------------------------------------------------
    "add_item": Intent.ADD_ITEM,
    "remove_item": Intent.REMOVE_ITEM,
    "modify_item": Intent.MODIFY_ITEM,
    "change_quantity": Intent.CHANGE_QUANTITY,
    "replace_item": Intent.REPLACE_ITEM,
    "undo_last": Intent.UNDO_LAST,
    "clear_cart": Intent.CLEAR_CART,

    # --------------------------------------------------
    # Cart queries
    # --------------------------------------------------
    "show_cart": Intent.SHOW_CART,
    "show_total": Intent.SHOW_TOTAL,
    "cart_item_query": Intent.CART_ITEM_QUERY,
    "review_order": Intent.REVIEW_ORDER,

    # --------------------------------------------------
    # Confirmation / conversation control
    # --------------------------------------------------
    "affirm": Intent.CONFIRM,
    "deny": Intent.DENY,
    "cancel_order": Intent.CANCEL_ORDER,

    # --------------------------------------------------
    # Menu browsing / info
    # --------------------------------------------------
    "browse_menu": Intent.BROWSE_MENU,
    "browse_category": Intent.BROWSE_CATEGORY,
    "ask_item_info": Intent.ASK_ITEM_INFO,
    "ask_options": Intent.ASK_OPTIONS,
    "availability_query": Intent.AVAILABILITY_QUERY,
    "ask_price": Intent.ASK_PRICE,
    "recommendation_query": Intent.RECOMMENDATION_QUERY,

    # --------------------------------------------------
    # Order flow
    # --------------------------------------------------
    "checkout": Intent.CHECKOUT,
    "confirm_order": Intent.CONFIRM_ORDER,
    "finish_order": Intent.FINISH_ORDER,

    # --------------------------------------------------
    # Order status
    # --------------------------------------------------
    "order_error_status": Intent.ORDER_ERROR_STATUS,
    "order_placement_status": Intent.ORDER_PLACEMENT_STATUS,
    "order_processing_status": Intent.ORDER_PROCESSING_STATUS,
    "order_status_general": Intent.ORDER_STATUS_GENERAL,

    # --------------------------------------------------
    # Payment
    # --------------------------------------------------
    "payment_request": Intent.PAYMENT_REQUEST,
    "payment_methods_query": Intent.PAYMENT_METHODS_QUERY,
    "payment_status": Intent.PAYMENT_STATUS,
    "payment_done": Intent.PAYMENT_DONE,

    # --------------------------------------------------
    # Greetings
    # --------------------------------------------------
    "greeting": Intent.GREETING,
    "morning": Intent.MORNING,
    "afternoon": Intent.AFTERNOON,
    "evening": Intent.EVENING,
    "night": Intent.NIGHT,
}