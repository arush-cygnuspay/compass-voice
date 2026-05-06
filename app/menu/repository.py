from __future__ import annotations

from app.menu.models import ItemResolution, MenuItem
from app.menu.query_result import MenuQueryResult
from app.menu.query_service import MenuQueryService, NearMissResult
from app.menu.store import MenuStore

__all__ = ["MenuRepository", "NearMissResult"]
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text


class MenuRepository:
    """Backward-compatible facade over ``MenuQueryService``.

    All public methods delegate to ``MenuQueryService``.  Callers that
    construct ``MenuRepository(store)`` continue to work unchanged.
    """

    def __init__(self, store: MenuStore):
        self.store = store
        self._service = MenuQueryService(store)

    # ------------------------------------------------------------------
    # DATA ACCESS
    # ------------------------------------------------------------------

    def get_item(self, item_id: str) -> MenuItem:
        return self._service.get_item(item_id)

    # ------------------------------------------------------------------
    # NOT-FOUND RECOVERY
    # ------------------------------------------------------------------

    def build_not_found_recovery(
        self,
        text: str,
        *,
        item_limit: int = 3,
        category_limit: int = 4,
    ) -> tuple[list[MenuItem], list[dict]]:
        return self._service.build_not_found_recovery(
            text, item_limit=item_limit, category_limit=category_limit
        )

    def build_not_found_recovery_normalized(
        self,
        normalized_text: str,
        *,
        item_limit: int = 3,
        category_limit: int = 4,
    ) -> tuple[list[MenuItem], list[dict]]:
        return self._service.build_not_found_recovery_normalized(
            normalized_text, item_limit=item_limit, category_limit=category_limit
        )

    def find_near_miss_item_normalized(
        self,
        normalized_text: str,
        *,
        threshold: float | None = None,
    ) -> NearMissResult | None:
        return self._service.find_near_miss_item_normalized(
            normalized_text, threshold=threshold
        )

    # ------------------------------------------------------------------
    # CATEGORY RESOLUTION
    # ------------------------------------------------------------------

    def resolve_category_query(self, text: str, *, limit: int = 5) -> MenuQueryResult | None:
        return self._service.resolve_category_query(text, limit=limit)

    def resolve_category_query_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult | None:
        return self._service.resolve_category_query_normalized(normalized_text, limit=limit)

    # ------------------------------------------------------------------
    # SLOT-FIRST RESOLUTION
    # ------------------------------------------------------------------

    def resolve_menu_query_from_slots(
        self,
        *,
        user_text: str,
        slots: list[SlotValue] | tuple[SlotValue, ...],
        fallback_to_text: bool = True,
        limit: int = 5,
    ) -> MenuQueryResult:
        return self._service.resolve_menu_query_from_slots(
            user_text=user_text,
            slots=slots,
            fallback_to_text=fallback_to_text,
            limit=limit,
        )

    def resolve_menu_query_from_slots_normalized(
        self,
        *,
        normalized_user_text: str,
        slots: list[SlotValue] | tuple[SlotValue, ...],
        fallback_to_text: bool = True,
        limit: int = 5,
    ) -> MenuQueryResult:
        return self._service.resolve_menu_query_from_slots_normalized(
            normalized_user_text=normalized_user_text,
            slots=slots,
            fallback_to_text=fallback_to_text,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # GENERAL FREE-TEXT RESOLUTION
    # ------------------------------------------------------------------

    def resolve_menu_query(self, text: str, *, limit: int = 5) -> MenuQueryResult:
        return self._service.resolve_menu_query(text, limit=limit)

    def resolve_menu_query_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult:
        return self._service.resolve_menu_query_normalized(normalized_text, limit=limit)

    # ------------------------------------------------------------------
    # IDLE AVAILABILITY RESOLUTION
    # ------------------------------------------------------------------

    def resolve_idle_availability_query_normalized(
        self,
        normalized_text: str,
        *,
        limit: int = 5,
    ) -> MenuQueryResult:
        return self._service.resolve_idle_availability_query_normalized(
            normalized_text, limit=limit
        )

    # ------------------------------------------------------------------
    # CANDIDATE-LOCAL / SIDE / MODIFIER RESOLUTION
    # ------------------------------------------------------------------

    def resolve_item_within_candidates_normalized(
        self,
        *,
        normalized_text: str,
        candidate_item_ids: list[str] | tuple[str, ...],
    ) -> MenuItem | None:
        return self._service.resolve_item_within_candidates_normalized(
            normalized_text=normalized_text,
            candidate_item_ids=candidate_item_ids,
        )

    def resolve_side_choice_within_group_normalized(
        self,
        *,
        normalized_text: str,
        group_id: str,
        candidate_names_by_id: dict[str, tuple[str, ...]],
    ) -> list[str]:
        return self._service.resolve_side_choice_within_group_normalized(
            normalized_text=normalized_text,
            group_id=group_id,
            candidate_names_by_id=candidate_names_by_id,
        )

    def resolve_modifier_choice_within_group_normalized(
        self,
        *,
        normalized_text: str,
        group_id: str,
        candidate_names_by_id: dict[str, tuple[str, ...]],
    ) -> list[str]:
        return self._service.resolve_modifier_choice_within_group_normalized(
            normalized_text=normalized_text,
            group_id=group_id,
            candidate_names_by_id=candidate_names_by_id,
        )

    def resolve_modifier_availability_for_item_normalized(
        self,
        *,
        normalized_text: str,
        item_id: str,
    ) -> dict | None:
        return self._service.resolve_modifier_availability_for_item_normalized(
            normalized_text=normalized_text,
            item_id=item_id,
        )

    def resolve_item(self, text: str) -> ItemResolution | None:
        return self._service.resolve_item(text)

    def resolve_item_normalized(self, normalized_text: str) -> ItemResolution | None:
        return self._service.resolve_item_normalized(normalized_text)
