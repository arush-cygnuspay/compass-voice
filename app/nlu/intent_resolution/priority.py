# app/nlu/intent_resolution/priority.py
from app.nlu.intent_resolution.intent import Intent

INTENT_PRIORITY: list[Intent] = [

    # ---- Terminal / Irreversible ----
    Intent.PAYMENT_DONE,

    # ---- Explicit Order Control ----
    Intent.ORDER_STATUS,
    Intent.START_ORDER,

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

    # ---- Conversational Responses ----
    Intent.CONFIRM,
    Intent.DENY,
    Intent.CANCEL,

    # ---- Low-Power Meta ----
    Intent.META_CLARIFY,

    # ---- Fallback ----
    Intent.UNKNOWN,
]

