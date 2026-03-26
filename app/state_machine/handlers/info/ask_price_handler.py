# app/state_machine/handlers/info/ask_price_handler.py

from __future__ import annotations

import re

from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.menu.slot_helpers import first_slot_value
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.session.session import Session
from app.state_machine.handlers.base_handler import BaseHandler
from app.state_machine.conversation_context import ConversationContext
from app.state_machine.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult



SIZE_WORDS = (
    "extra large",
    "large",
    "medium",
    "small",
    "regular",
    "mini",
    "xl",
)


class AskPriceHandler(BaseHandler):
    """
    Handles item price inquiries.

    Read-only:
    - no cart mutation
    - no dialog-flow mutation
    - always returns to IDLE
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context: ConversationContext,
        user_text: str,
        session: Session | None = None,
    ) -> HandlerResult:
        if intent != Intent.ASK_PRICE:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="unhandled_intent",
            )

        normalized_text = normalize_text(user_text or "")
        slots = getattr(context, "last_slots", ()) or ()

        item_slot_value = first_slot_value(slots, "ITEM", "MENU_ITEM")
        category_slot_value = first_slot_value(slots, "CATEGORY", "MENU_CATEGORY")

        if item_slot_value or category_slot_value:
            result = self.menu_repo.resolve_menu_query_from_slots(
                user_text=normalized_text,
                slots=slots,
                fallback_to_text=True,
                limit=5,
            )
        else:
            result = self.menu_repo.resolve_menu_query(
                normalized_text,
                limit=5,
            )

        if result.type == MenuQueryType.ITEM and result.item is not None:
            return self._build_price_result(
                item=result.item,
                normalized_text=normalized_text,
                slots=slots,
            )

        if (
            result.type == MenuQueryType.CATEGORY_SINGLE_ITEM
            and result.items
            and len(result.items) == 1
        ):
            return self._build_price_result(
                item=result.items[0],
                normalized_text=normalized_text,
                slots=slots,
            )

        if result.type == MenuQueryType.ITEM_AMBIGUOUS:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="menu_ambiguity",
                response_payload={
                    "options": [item.name for item in (result.matched_items or [])],
                },
            )

        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="price_not_found",
        )

    def _build_price_result(self, *, item, normalized_text: str, slots) -> HandlerResult:
        pricing = item.pricing
        pricing_payload = self._serialize_pricing(pricing)
        requested_size = self._extract_requested_size(normalized_text, slots)

        if pricing.mode == "variant" and requested_size:
            matched_variant = self._match_variant_label(requested_size, pricing.variants or [])
            if matched_variant is not None:
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="show_item_price",
                    response_payload={
                        "item_name": item.name,
                        "pricing": pricing_payload,
                        "variant_label": matched_variant.label,
                        "variant_price_cents": matched_variant.price_cents,
                    },
                )

        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="show_item_price",
            response_payload={
                "item_name": item.name,
                "pricing": pricing_payload,
            },
        )

    def _serialize_pricing(self, pricing) -> dict:
        return {
            "mode": pricing.mode,
            "price_cents": pricing.price_cents,
            "currency": getattr(pricing, "currency", "USD"),
            "variants": [
                {
                    "variant_id": variant.variant_id,
                    "label": variant.label,
                    "price_cents": variant.price_cents,
                }
                for variant in (pricing.variants or [])
            ],
        }

    def _extract_requested_size(self, normalized_text: str, slots) -> str | None:
        slot_size = first_slot_value(slots, "SIZE")
        if isinstance(slot_size, str) and slot_size.strip():
            return normalize_text(slot_size)

        for size in SIZE_WORDS:
            match = re.search(rf"\b{re.escape(size)}\b", normalized_text)
            if match:
                return normalize_text(size)

        return None

    def _match_variant_label(self, requested_size: str, variants) -> object | None:
        requested_size = normalize_text(requested_size)
        if not requested_size:
            return None

        for variant in variants:
            label = normalize_text(getattr(variant, "label", "") or "")
            if label == requested_size:
                return variant

        for variant in variants:
            label = normalize_text(getattr(variant, "label", "") or "")
            if requested_size and label and (
                requested_size in label or label in requested_size
            ):
                return variant

        return None
