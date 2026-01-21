# app/nlu/intent_resolution/common_intent_resolver.py
from typing import Optional

from app.nlu.intent_patterns.common import *
from app.nlu.intent_resolution.intent import Intent


def match_yes_no(text: str) -> Optional[Intent]:
    """
    Resolves YES / NO confirmation intent.

    Used in:
    - order confirmation
    - payment confirmation
    - generic confirm prompts

    Priority:
    - NO > YES
    """

    # -------------------------
    # NO / DENY (higher priority)
    # -------------------------
    if (
        NO_STRONG_PAT.search(text)
        or NO_CANCEL_PAT.search(text)
        or NO_NEGATION_PAT.search(text)
        or NO_REVERSAL_PAT.search(text)
        or NO_DEFER_PAT.search(text)
    ):
        return Intent.DENY

    # -------------------------
    # YES / CONFIRM
    # -------------------------
    if (
        YES_STRONG_PAT.search(text)
        or YES_ACTION_PAT.search(text)
        or YES_SOFT_PAT.search(text)
    ):
        return Intent.CONFIRM

    return None

