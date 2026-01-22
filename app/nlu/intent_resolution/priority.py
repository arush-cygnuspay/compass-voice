# app/nlu/intent_resolution/priority.py
from app.nlu.intent_resolution.intent import Intent

INTENT_PRIORITY: list[Intent] = [

    # ---- Terminal / Irreversible ----
    Intent.PAYMENT_DONE,

    # ---- Hard Stops / Escapes ----
    Intent.CANCEL,

    # ---- Confirmation / Rejection ----
    Intent.CONFIRM,
    Intent.DENY,

    # ---- Payment Flow ----
    Intent.PAYMENT_REQUEST,

    # ---- INFO (item/categories/price) ----
    Intent.ASK_PRICE,
    Intent.ASK_MENU_INFO,

    # ---- Explicit Order Control ----
    Intent.ORDER_STATUS,
    Intent.START_ORDER,

    # ---- Flow Transition ----
    Intent.END_ADDING,

    # ---- Informational ----
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.SHOW_MENU,
    Intent.CLEAR_CART,

    # ---- Cart Mutation ----
    Intent.ADD_ITEM,
    Intent.MODIFY_ITEM,
    Intent.REMOVE_ITEM,

    # ---- Meta / Repair ----
    Intent.META_CLARIFY,

    # ---- Fallback ----
    Intent.UNKNOWN,
]

