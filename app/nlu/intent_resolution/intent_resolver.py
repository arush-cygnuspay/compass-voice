# app/nlu/intent_resolution/intent_resolver.py

from typing import Set

from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.intent_resolution.item_intent_resolver import match_add_item
from app.nlu.intent_resolution.priority import INTENT_PRIORITY
from app.utils.text_utils import normalize_text


def resolve_intent(text: str) -> IntentResult:
    """
    Resolve linguistic intent from raw user text.

    GUARANTEES:
    - Pure
    - Deterministic
    - Regex-based only
    """

    if not text:
        return IntentResult(Intent.UNKNOWN, "")

    normalized = normalize_text(text)
    matches: Set[Intent] = set()

    # -----------------------------------------
    # ADD ITEM (only one implemented for now)
    # -----------------------------------------
    if match_add_item(normalized):
        matches.add(Intent.ADD_ITEM)

    if match_yes_no(normalized):
        matches.add(Intent.CONFIRM)

    # -----------------------------------------
    # Priority resolution
    # -----------------------------------------
    for intent in INTENT_PRIORITY:
        if intent in matches:
            return IntentResult(intent=intent, raw_text=normalized)

    return IntentResult(Intent.UNKNOWN, normalized)
