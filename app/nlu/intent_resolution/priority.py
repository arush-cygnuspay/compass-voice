# app/nlu/intent_resolution/priority.py
from app.nlu.intent_resolution.intent import Intent

INTENT_PRIORITY: list[Intent] = [

    # ---- Terminal / Irreversible ----
    Intent.PAYMENT_DONE,

    # ---- Hard Stops / Escapes ----
    Intent.CANCEL,

    # ---- Payment Flow ----
    Intent.PAYMENT_REQUEST,

    # ---- Explicit Order Control ----
    Intent.ORDER_STATUS,
    Intent.START_ORDER,

    # ---- Confirmation / Rejection ----
    Intent.CONFIRM,
    Intent.DENY,

    # ---- Flow Transition ----
    Intent.END_ADDING,

    # ---- Cart Mutation ----
    Intent.ADD_ITEM,
    Intent.MODIFY_ITEM,
    Intent.REMOVE_ITEM,

    # ---- Informational ----
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.SHOW_MENU,

    # ---- Meta / Repair ----
    Intent.META_CLARIFY,

    # ---- Fallback ----
    Intent.UNKNOWN,
]

