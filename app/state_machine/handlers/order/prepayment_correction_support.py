# app/state_machine/handlers/order/prepayment_correction_support.py
from __future__ import annotations

import re

from app.cart.cart_item import CartItem
from app.menu.repository import MenuRepository
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.cart_edit_support import (
    extract_item_slot_values,
    match_cart_item_from_text,
    strip_edit_prefixes,
)
from app.utils.quantity_detection import normalize_quantity


_LEADING_QUANTITY_FRAGMENT = re.compile(
    r"\b(?:make it|change it to|make that|change that to|set that to|set it to)\s+"
    r"(?P<qty>\d+|a|an|single|one|two|three|four|five|six|seven|eight|nine|ten)\b"
)

_ANY_QUANTITY_FRAGMENT = re.compile(
    r"\b(?P<qty>\d+|a|an|single|one|two|three|four|five|six|seven|eight|nine|ten)\b"
)


def clone_cart_item_with_quantity(cart_item, quantity: int) -> CartItem:
    return CartItem(
        cart_item_id=cart_item.cart_item_id,
        item_id=cart_item.item_id,
        quantity=quantity,
        variant_id=cart_item.variant_id,
        sides={group_id: list(item_ids) for group_id, item_ids in cart_item.sides.items()},
        side_variants=dict(cart_item.side_variants),
        modifiers={group_id: list(values) for group_id, values in cart_item.modifiers.items()},
    )


def extract_requested_quantity(text: str) -> int | None:
    normalized = normalize_text(text or "")
    if not normalized:
        return None

    leading = _LEADING_QUANTITY_FRAGMENT.search(normalized)
    if leading is not None:
        quantity = normalize_quantity(leading.group("qty"))
        if quantity is not None and quantity > 0:
            return quantity

    instead_of = normalized.split(" instead of ", 1)[0].strip()
    if instead_of and instead_of != normalized:
        match = _ANY_QUANTITY_FRAGMENT.search(instead_of)
        if match is not None:
            quantity = normalize_quantity(match.group("qty"))
            if quantity is not None and quantity > 0:
                return quantity

    return None


def resolve_cart_item_for_quantity_change(
    *,
    menu_repo: MenuRepository,
    session,
    context,
    user_text: str,
):
    normalized = strip_edit_prefixes(user_text)
    fragments: list[str] = []

    slot_candidates = extract_item_slot_values(context)
    fragments.extend(slot_candidates)

    if " instead of " in normalized:
        fragments.append(normalized.split(" instead of ", 1)[0].strip())

    for prefix in ("make it ", "change it to ", "make that ", "change that to ", "set that to "):
        if normalized.startswith(prefix):
            fragments.append(normalized[len(prefix):].strip())

    fragments.append(normalized)

    return match_cart_item_from_text(
        menu_repo=menu_repo,
        session=session,
        candidate_texts=[fragment for fragment in fragments if fragment],
    )
