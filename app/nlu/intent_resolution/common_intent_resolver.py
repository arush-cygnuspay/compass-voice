# app/nlu/intent_resolution/common_intent_resolver.py
from typing import Optional

from app.nlu.intent_patterns.common import *
from app.nlu.intent_resolution.intent import Intent

# app/nlu/intent_resolution/yes_no.py

def match_yes_no(text: str) -> Optional[Intent]:
    """
    Resolve yes / no / cancel intent.

    Semantic priority:
    1. CANCEL  → abort current task
    2. DENY    → refuse current option
    3. CONFIRM → accept
    """

    if NO_CANCEL_PAT.search(text):
        return Intent.CANCEL

    if NO_STRONG_PAT.search(text):
        return Intent.DENY

    if NO_NEGATION_PAT.search(text):
        return Intent.DENY

    if YES_STRONG_PAT.search(text):
        return Intent.CONFIRM

    return None



