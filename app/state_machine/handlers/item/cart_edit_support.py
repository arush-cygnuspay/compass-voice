from __future__ import annotations

from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.menu.slot_helpers import slot_values
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.utils.item_matching import score_item

EDIT_PREFIXES: tuple[str, ...] = (
    "please ",
    "can you ",
    "could you ",
    "i want to ",
    "i want ",
    "i need to ",
    "remove ",
    "delete ",
    "take off ",
    "undo ",
    "replace ",
    "swap ",
    "change ",
    "modify ",
    "edit ",
)

REPLACEMENT_SEPARATORS: tuple[str, ...] = (
    " instead of ",
    " with ",
    " for ",
    " to ",
)


def extract_item_slot_values(context) -> list[str]:
    return [
        normalize_text(value)
        for value in slot_values(getattr(context, "last_slots", ()) or (), "ITEM", "MENU_ITEM")
        if normalize_text(value)
    ]


def strip_edit_prefixes(text: str) -> str:
    normalized = normalize_text(text or "")
    changed = True
    while changed and normalized:
        changed = False
        for prefix in EDIT_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                changed = True
                break
    return normalized


def split_replacement_request(text: str) -> tuple[str, str]:
    normalized = strip_edit_prefixes(text)
    for separator in REPLACEMENT_SEPARATORS:
        if separator not in normalized:
            continue
        left, right = normalized.split(separator, 1)
        return left.strip(), right.strip()
    return normalized, ""


def resolve_menu_item_from_text(
    menu_repo: MenuRepository,
    text: str,
    *,
    exclude_item_ids: set[str] | None = None,
):
    normalized = normalize_text(text or "")
    if not normalized:
        return None

    result = menu_repo.resolve_menu_query(normalized, limit=5)
    candidates = []

    if result.type == MenuQueryType.ITEM and result.item is not None:
        candidates = [result.item]
    elif (
        result.type == MenuQueryType.CATEGORY_SINGLE_ITEM
        and result.items
        and len(result.items) == 1
    ):
        candidates = [result.items[0]]

    exclude_item_ids = exclude_item_ids or set()
    for item in candidates:
        if item.item_id not in exclude_item_ids:
            return item

    return None


def match_cart_item_from_text(
    *,
    menu_repo: MenuRepository,
    session,
    candidate_texts: list[str],
):
    cart_items = session.cart.get_items() if session is not None else []
    if not cart_items:
        return None

    best_cart_item = None
    best_score = 0.0

    for candidate_text in candidate_texts:
        normalized_candidate = normalize_text(candidate_text or "")
        if not normalized_candidate:
            continue

        for cart_item in cart_items:
            menu_item = menu_repo.get_item(cart_item.item_id)
            labels = [menu_item.name, *getattr(menu_item, "aliases", ()), *getattr(menu_item, "voice_labels", ())]
            score = max(
                (score_item(normalized_candidate, normalize_text(label)) for label in labels if label),
                default=0.0,
            )
            if score > best_score:
                best_score = score
                best_cart_item = cart_item

    return best_cart_item if best_score >= 2.5 else None
