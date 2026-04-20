from __future__ import annotations

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.conversation_context import ConversationContext

ORDER_SLOT_NAMES = {
    "ITEM",
    "MENU_ITEM",
    "CATEGORY",
    "MENU_CATEGORY",
    "SIDE",
    "MODIFIER",
}

ORDERING_PREFIXES = (
    "i want",
    "i would like",
    "id like",
    "i'd like",
    "ill take",
    "i'll take",
    "can i get",
    "could i get",
    "give me",
    "add ",
    "make it",
    "bring me",
    "send me",
    "get me",
    "let me get",
)

NON_ORDERING_SLOT_VALUES = {
    "pickup",
    "pick up",
    "delivery",
    "deliver",
    "yes",
    "yeah",
    "yep",
    "no",
    "nope",
    "zip",
    "zipcode",
    "postal code",
    "house number",
    "street",
    "address",
    "apartment",
    "suite",
    "unit",
    "none",
    "skip",
}


def looks_like_ordering_request(
    context: ConversationContext,
    text: str,
    *,
    include_slots: bool = True,
) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False

    if any(normalized.startswith(prefix) for prefix in ORDERING_PREFIXES):
        return True

    if not include_slots:
        return False

    for slot in context.last_slots or ():
        slot_name = str(getattr(slot, "name", "")).upper()
        if slot_name not in ORDER_SLOT_NAMES:
            continue

        value = getattr(slot, "value", None)
        if not isinstance(value, str):
            continue

        normalized_value = normalize_text(value)
        if normalized_value and normalized_value not in NON_ORDERING_SLOT_VALUES:
            return True

    return False
