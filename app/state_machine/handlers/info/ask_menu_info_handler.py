from __future__ import annotations

from app.menu.query_result import MenuQueryType
from app.menu.repository import MenuRepository
from app.menu.slot_helpers import first_slot_value
from app.nlu.intent_resolution.intent import Intent
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.conversation_state import ConversationState
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.base_handler import BaseHandler


class AskMenuInfoHandler(BaseHandler):
    """
    Read-only menu exploration / information handler.

    Design:
    - intent-first behavior
    - BROWSE_MENU / SHOW_MENU => show menu categories
    - BROWSE_CATEGORY => category-first resolution
    - ASK_OPTIONS in IDLE => prefer category view if category is present
    - ASK_* info intents => if CATEGORY slot exists, honor it first
    - never mutates cart or add-item flow state
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self.menu_repo = menu_repo

    def handle(
        self,
        intent: Intent,
        context,
        user_text: str,
        session=None,
    ) -> HandlerResult:
        normalized_text = normalize_text(user_text or "")
        slots = getattr(context, "last_slots", ()) or ()

        if intent in {Intent.BROWSE_MENU, Intent.SHOW_MENU}:
            return self._show_browse_categories()

        if intent == Intent.BROWSE_CATEGORY:
            category = self._resolve_category_for_browse(
                user_text=normalized_text,
                slots=slots,
            )
            if category is not None:
                return self._show_category_from_dict(category)

            return self._show_browse_categories()

        if intent == Intent.ASK_OPTIONS:
            category = self._resolve_category_for_browse(
                user_text=normalized_text,
                slots=slots,
            )
            if category is not None:
                return self._show_category_from_dict(category)

            result = self.menu_repo.resolve_menu_query(normalized_text, limit=5)
            fallback = self._fallback_to_recent_category_options(
                session=session,
                result=result,
            )
            if fallback is not None:
                return fallback

            return self._menu_result_to_handler_result(intent=intent, result=result)

        if intent == Intent.AVAILABILITY_QUERY:
            current_item_id = getattr(context, "current_item_id", None)

            if current_item_id:
                modifier_match = self.menu_repo.resolve_modifier_availability_for_item_normalized(
                    normalized_text=normalized_text,
                    item_id=current_item_id,
                )
                if modifier_match is not None:
                    return HandlerResult(
                        next_state=ConversationState.IDLE,
                        response_key="show_modifier_availability",
                        response_payload=modifier_match,
                    )

            category_slot_value = first_slot_value(slots, "CATEGORY", "MENU_CATEGORY")
            if category_slot_value:
                category_result = self.menu_repo.resolve_category_query(
                    category_slot_value,
                    limit=5,
                )
                if category_result is not None:
                    return self._menu_result_to_handler_result(intent=intent, result=category_result)

            result = self.menu_repo.resolve_idle_availability_query_normalized(
                normalized_text,
                limit=5,
            )

            if result.type == MenuQueryType.NOT_FOUND:
                modifier_hits = self.menu_repo.store.find_modifier_entities(normalized_text)
                if modifier_hits:
                    return HandlerResult(
                        next_state=ConversationState.IDLE,
                        response_key="modifier_available_with_item_context",
                        response_payload={"modifier_name": user_text},
                    )

            fallback = self._fallback_to_recent_category_options(
                session=session,
                result=result,
            )
            if fallback is not None:
                return fallback

            return self._menu_result_to_handler_result(intent=intent, result=result)

        if intent in {
            Intent.ASK_ITEM_INFO,
            Intent.ASK_MENU_INFO,
            Intent.RECOMMENDATION_QUERY,
            Intent.UNKNOWN,
        }:
            category_slot_value = first_slot_value(slots, "CATEGORY", "MENU_CATEGORY")
            if category_slot_value:
                category_result = self.menu_repo.resolve_category_query(
                    category_slot_value,
                    limit=5,
                )
                if category_result is not None:
                    return self._menu_result_to_handler_result(
                        intent=intent,
                        result=category_result,
                    )

            result = self.menu_repo.resolve_menu_query(normalized_text, limit=5)
            fallback = self._fallback_to_recent_category_options(
                session=session,
                result=result,
            )
            if fallback is not None:
                return fallback

            return self._menu_result_to_handler_result(intent=intent, result=result)

        result = self.menu_repo.resolve_menu_query(normalized_text, limit=5)
        fallback = self._fallback_to_recent_category_options(
            session=session,
            result=result,
        )
        if fallback is not None:
            return fallback

        return self._menu_result_to_handler_result(intent=intent, result=result)

    def _show_browse_categories(self) -> HandlerResult:
        categories = self._get_browse_categories(limit=6)

        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="show_menu_categories",
            response_payload={
                "categories": [
                    category.get("name")
                    for category in categories
                    if category.get("name")
                ],
            },
        )

    def _show_category_from_dict(self, category: dict) -> HandlerResult:
        item_ids = category.get("item_ids", []) or []
        items = []

        for item_id in item_ids:
            if item_id not in self.menu_repo.store.items:
                continue
            try:
                items.append(self.menu_repo.store.get_item(item_id))
            except KeyError:
                continue

        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="show_category",
            response_payload={
                "category_id": category.get("category_id"),
                "category_name": category.get("name"),
                "items": [item.name for item in items[:10]],
            },
        )

    def _get_browse_categories(self, limit: int = 6) -> list[dict]:
        _items, categories = self.menu_repo.build_not_found_recovery(
            "",
            item_limit=0,
            category_limit=limit,
        )
        return categories or []

    def _resolve_category_for_browse(
        self,
        *,
        user_text: str,
        slots,
    ) -> dict | None:
        candidates: list[str] = []

        def add_candidate(value: str | None) -> None:
            if not isinstance(value, str):
                return
            value = normalize_text(value)
            if value and value not in candidates:
                candidates.append(value)

        add_candidate(first_slot_value(slots, "CATEGORY", "MENU_CATEGORY"))
        add_candidate(first_slot_value(slots, "ITEM", "MENU_ITEM"))
        add_candidate(user_text)

        for candidate in candidates:
            category_result = self.menu_repo.resolve_category_query(candidate, limit=10)
            if category_result is not None and category_result.category_id:
                return self.menu_repo.store.categories.get(category_result.category_id)

        return None

    def _fallback_to_recent_category_options(
        self,
        *,
        session,
        result,
    ) -> HandlerResult | None:
        """
        When a follow-up after a category listing cannot be resolved
        (often due to STT distortion), re-offer the last shown options
        instead of failing generically.
        """
        if session is None:
            return None

        if result.type != MenuQueryType.NOT_FOUND:
            return None

        last_key = getattr(session, "last_response_key", None) or ""
        last_payload = getattr(session, "last_response_payload", None) or {}

        if last_key not in {"show_category", "menu_ambiguity"}:
            return None

        options = [
            option
            for option in (last_payload.get("items") or last_payload.get("options") or [])
            if isinstance(option, str) and option.strip()
        ]
        if not options:
            return None

        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="menu_ambiguity",
            response_payload={"options": options[:5]},
        )

    def _menu_result_to_handler_result(
        self,
        *,
        intent: Intent,
        result,
    ) -> HandlerResult:
        if result.type == MenuQueryType.CATEGORY:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="show_category",
                response_payload={
                    "category_id": result.category_id,
                    "category_name": result.category_name,
                    "items": [item.name for item in (result.items or [])],
                },
            )

        if result.type == MenuQueryType.CATEGORY_SINGLE_ITEM:
            single_item = (result.items or [None])[0]
            if single_item is None:
                return HandlerResult(
                    next_state=ConversationState.IDLE,
                    response_key="menu_not_found",
                )

            response_key = (
                "show_item_availability"
                if intent == Intent.AVAILABILITY_QUERY
                else "show_item_info"
            )

            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key=response_key,
                response_payload={
                    "item_name": single_item.name,
                    "pricing": {
                        "mode": getattr(single_item, "pricing_mode", None),
                        "price_cents": getattr(single_item, "price_cents", None),
                        "variants": list(getattr(single_item, "variants", []) or []),
                    },
                },
            )

        if result.type == MenuQueryType.ITEM and result.item is not None:
            response_key = (
                "show_item_availability"
                if intent == Intent.AVAILABILITY_QUERY
                else "show_item_info"
            )

            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key=response_key,
                response_payload={
                    "item_name": result.item.name,
                    "description": getattr(result.item, "description", "") or "",
                    "pricing": {
                        "mode": getattr(result.item, "pricing_mode", None),
                        "price_cents": getattr(result.item, "price_cents", None),
                        "variants": list(getattr(result.item, "variants", []) or []),
                    },
                },
            )

        if result.type == MenuQueryType.ITEM_AMBIGUOUS:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="menu_ambiguity",
                response_payload={
                    "options": [item.name for item in (result.matched_items or [])],
                },
            )

        if result.type == MenuQueryType.CATEGORY_AMBIGUOUS:
            return HandlerResult(
                next_state=ConversationState.IDLE,
                response_key="menu_ambiguity",
                response_payload={
                    "options": [
                        category.get("name")
                        for category in (result.matched_categories or [])
                        if category.get("name")
                    ],
                },
            )

        return HandlerResult(
            next_state=ConversationState.IDLE,
            response_key="menu_not_found",
        )