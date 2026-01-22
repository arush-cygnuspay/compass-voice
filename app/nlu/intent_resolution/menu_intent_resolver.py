# app/nlu/intent_resolution/menu_intent_resolver.py

from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_patterns.info import *

def match_ask_menu_info(text: str) -> set[Intent]:
    """
    Detects informational menu queries (item OR category).
    MUST NOT trigger on cart/order queries.
    """

    # 🚫 HARD GUARD: cart/order language
    if CART_GUARD_PAT.search(text):
        return set()

    if ASK_CATEGORY_LIST_PAT.search(text):
        return {Intent.ASK_MENU_INFO}

    if ASK_ITEM_INFO_PAT.search(text):
        return {Intent.ASK_MENU_INFO}

    if ASK_VIEW_PAT.search(text):
        return {Intent.ASK_MENU_INFO}

    if WHICH_CATEGORY_PAT.search(text):
        return {Intent.ASK_MENU_INFO}

    if SOFT_MENU_INQUIRY_PAT.search(text):
        return {Intent.ASK_MENU_INFO}

    if BARE_CATEGORY_HINT_PAT.match(text):
        return {Intent.ASK_MENU_INFO}

    return set()


def match_price_intent(text: str) -> set[Intent]:
    if ASK_PRICE_PAT.search(text):
        return {Intent.ASK_PRICE}
    return set()