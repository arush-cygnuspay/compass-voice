# app/state_machine/policy/intent_coercion.py
"""FSM-aware intent coercion for idle add-item cases.

Runs AFTER NLU resolution and BEFORE StateRouter.route() so that the
router always sees the correct intent.  The StateRouter and handlers never
need to know that rewriting happened.

Three rules (see Phase E spec):

Rule 1 — idle + UNKNOWN/low-confidence + ITEM slots + menu evidence
    → ADD_ITEM, reason "idle_item_slot_with_menu_evidence"

Rule 2 — idle + MODIFY_ITEM/REPLACE_ITEM + ITEM slots + no cart target
    → ADD_ITEM, reason "idle_modify_no_target_with_item_slot"

Rule 3 — idle + MODIFY_ITEM + ITEM+SIZE slots + empty cart
    → ADD_ITEM, reason "idle_item_variant_no_cart_target"

Non-coercible intents (checkout, cancel, help, control) are never touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_result import NLUResult
from app.state_machine.models.conversation_state import ConversationState

# Intents that should never be coerced into ADD_ITEM under any circumstances.
_PROTECTED_INTENTS: frozenset[Intent] = frozenset({
    Intent.CHECKOUT,
    Intent.CONFIRM_ORDER,
    Intent.FINISH_ORDER,
    Intent.START_ORDER,
    Intent.END_ADDING,
    Intent.PAYMENT_REQUEST,
    Intent.REVIEW_ORDER,
    Intent.CANCEL_ORDER,
    Intent.REMOVE_ITEM,
    Intent.UNDO_LAST,
    Intent.REQUEST_AGENT,
    Intent.GREETING,
    Intent.MORNING,
    Intent.AFTERNOON,
    Intent.EVENING,
    Intent.NIGHT,
    Intent.SHOW_CART,
    Intent.SHOW_TOTAL,
    Intent.CLEAR_CART,
    Intent.PAYMENT_STATUS,
    Intent.ORDER_STATUS_GENERAL,
    # Affirmation / denial / confirmation / cancellation — these are user
    # control responses that must never be silently rewritten to ADD_ITEM.
    # "yeah, coke" (AFFIRM) or "no, spicy chicken" (DENY) carry a slot value
    # but the intent itself is a direct reply to a yes/no question; coercing
    # them would hijack the in-progress flow state.
    Intent.AFFIRM,
    Intent.DENY,
    Intent.CONFIRM,
    Intent.CANCEL,
})

# Slot labels that indicate item/menu evidence.
_ITEM_SLOT_LABELS: frozenset[str] = frozenset({"ITEM", "MENU_ITEM", "CATEGORY"})
_SIZE_SLOT_LABELS: frozenset[str] = frozenset({"SIZE", "VARIANT"})


@dataclass(frozen=True, slots=True)
class CoercionResult:
    intent_result: IntentResult
    coercion_reason: str | None  # None → no coercion applied


def _slot_values_for_labels(
    slots: tuple[Any, ...],
    labels: frozenset[str],
) -> list[str]:
    return [
        str(getattr(s, "value", "") or "")
        for s in slots
        if str(getattr(s, "name", "")).upper() in labels
        and str(getattr(s, "value", "") or "").strip()
    ]


def _has_menu_evidence(item_texts: list[str], menu_repo: Any) -> bool:
    """Return True when at least one item-slot value matches a known menu entity."""
    for text in item_texts:
        from app.nlu.query_normalization.text_preprocessor import normalize_text
        norm = normalize_text(text)
        if not norm:
            continue
        # Exact name / alias / voice-label lookups — all O(1).
        if menu_repo.store.find_item_exact(norm):
            return True
        if menu_repo.store.find_item_ids_by_alias(norm):
            return True
        if menu_repo.store.find_item_ids_by_voice_label(norm):
            return True
        # Entity index (covers multi-word names indexed at ingest time).
        if menu_repo.store.find_entity(norm, allowed_types={"item"}):
            return True
    return False


def _has_cart_target_for_item(item_texts: list[str], cart: Any, menu_repo: Any) -> bool:
    """Return True when the cart contains an item that matches at least one slot value.

    CartItem only carries item_id — names are resolved via menu_repo.get_item().
    cart.get_items() is the correct public accessor (cart.items does not exist).
    """
    if cart.is_empty():
        return False
    from app.nlu.query_normalization.text_preprocessor import normalize_text
    cart_names: set[str] = set()
    for ci in (cart.get_items() if callable(getattr(cart, "get_items", None)) else ()):
        try:
            menu_item = menu_repo.get_item(ci.item_id)
            norm = normalize_text(getattr(menu_item, "name", "") or "")
            if norm:
                cart_names.add(norm)
        except (KeyError, AttributeError):
            pass
    for text in item_texts:
        if normalize_text(text) in cart_names:
            return True
    return False


class IntentCoercionPolicy:
    """Stateless policy — instantiated once and reused across all turns."""

    def __init__(self, menu_repo: Any) -> None:
        self._menu_repo = menu_repo

    def coerce(
        self,
        *,
        state: ConversationState,
        intent_result: IntentResult,
        nlu: NLUResult,
        cart: Any,
    ) -> CoercionResult:
        """Return a (possibly rewritten) IntentResult + the coercion reason.

        Only rewrites in IDLE state.  No-op for all other states.
        """
        intent = intent_result.intent

        if state != ConversationState.IDLE:
            return CoercionResult(intent_result=intent_result, coercion_reason=None)

        if intent in _PROTECTED_INTENTS:
            return CoercionResult(intent_result=intent_result, coercion_reason=None)

        slots = nlu.slots or ()
        item_texts = _slot_values_for_labels(slots, _ITEM_SLOT_LABELS)
        size_texts = _slot_values_for_labels(slots, _SIZE_SLOT_LABELS)

        # ── Rule 3: MODIFY_ITEM + ITEM+SIZE + empty cart ─────────────────
        if (
            intent == Intent.MODIFY_ITEM
            and item_texts
            and size_texts
            and cart.is_empty()
        ):
            return CoercionResult(
                intent_result=IntentResult(
                    intent=Intent.ADD_ITEM,
                    raw_text=intent_result.raw_text,
                ),
                coercion_reason="idle_item_variant_no_cart_target",
            )

        # ── Rule 2: MODIFY_ITEM/REPLACE_ITEM + ITEM + no cart target ─────
        if intent in {Intent.MODIFY_ITEM, Intent.REPLACE_ITEM} and item_texts:
            if not _has_cart_target_for_item(item_texts, cart, self._menu_repo):
                return CoercionResult(
                    intent_result=IntentResult(
                        intent=Intent.ADD_ITEM,
                        raw_text=intent_result.raw_text,
                    ),
                    coercion_reason="idle_modify_no_target_with_item_slot",
                )

        # ── Rule 1: UNKNOWN/ADD_ITEM-like + ITEM slots + menu evidence ────
        if intent in {Intent.UNKNOWN, Intent.ADD_ITEM} and item_texts:
            if _has_menu_evidence(item_texts, self._menu_repo):
                if intent == Intent.UNKNOWN:
                    return CoercionResult(
                        intent_result=IntentResult(
                            intent=Intent.ADD_ITEM,
                            raw_text=intent_result.raw_text,
                        ),
                        coercion_reason="idle_item_slot_with_menu_evidence",
                    )
                # ADD_ITEM already — no rewrite needed, but stamp the reason.
                return CoercionResult(
                    intent_result=intent_result,
                    coercion_reason=None,
                )

        return CoercionResult(intent_result=intent_result, coercion_reason=None)
