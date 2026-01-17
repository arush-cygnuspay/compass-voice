# app/nlu/intent_resolution/priority.py
from app.nlu.intent_resolution.intent import Intent

INTENT_PRIORITY: list[Intent] = [
    # # ---- Social / Control ----
    # Intent.GREETING,
    # Intent.THANKS,
    # Intent.GOODBYE,
    # Intent.YES,
    # Intent.NO,
    #
    # # ---- Cart State ----
    # Intent.CLEAR_CART,
    # Intent.CART_TOTAL,
    # Intent.CART_DETAILS,
    # Intent.CART_ITEM_QUERY,

    # ---- Cart Mutation ----
    Intent.REMOVE_ITEM,
    Intent.MODIFY_ITEM,
    Intent.ADD_ITEM,

    # # ---- Item Information ----
    # Intent.ITEM_PRICE,
    # Intent.ITEM_DESCRIPTION,
    # Intent.ITEM_AVAILABILITY,
    # Intent.ITEM_VARIANTS,
    # Intent.ITEM_COMPARISON,
    #
    # # ---- Menu / Restaurant ----
    # Intent.MENU,
    # Intent.MENU_CATEGORY,
    # Intent.RESTAURANT_INFO,
    #
    # # ---- Order / Payment ----
    # Intent.CONFIRM_ORDER,
    # Intent.ORDER_STATUS,
    # Intent.PAYMENT_HELP,
    # Intent.PAYMENT_METHOD_QUERY,
    # Intent.PAYMENT_LOCATION,
    # Intent.SEND_PAYMENT_LINK,
    # Intent.CONFIRM_PAYMENT,
]
