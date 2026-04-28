# app/menu/indexer.py
"""Menu index adapter.

Provides focused candidate-lookup operations over MenuStore's pre-built
in-memory indexes.  Code moved verbatim from MenuRepository._candidate_items_from_text
and MenuRepository._has_explicit_item_evidence.
"""
from __future__ import annotations

from app.menu.models import MenuItem
from app.menu.store import MenuStore


class MenuIndexer:
    """Thin adapter over MenuStore that surfaces candidate-lookup operations.

    ``MenuMatcher`` and ``MenuQueryService`` depend on this class instead of
    ``MenuStore`` directly so the store's internal structure remains an
    implementation detail of the repository layer.
    """

    def __init__(self, store: MenuStore) -> None:
        self.store = store

    def candidate_items(self, normalized_text: str) -> list[MenuItem]:
        """Return the candidate item pool for *normalized_text*.

        Queries entity index, exact-name index, alias index, and voice-label
        index in order.  Falls back to the entire item catalog when no index
        produces a hit (preserves original MenuRepository behavior).
        """
        entity_candidates = self.store.find_entity(normalized_text, allowed_types={"item"})
        candidate_ids: set[str] = {
            entry.get("item_id")
            for entry in entity_candidates
            if entry.get("item_id")
        }

        exact_item = self.store.find_item_exact(normalized_text)
        if exact_item is not None:
            candidate_ids.add(exact_item.item_id)

        for item_id in self.store.find_item_ids_by_alias(normalized_text):
            candidate_ids.add(item_id)

        for item_id in self.store.find_item_ids_by_voice_label(normalized_text):
            candidate_ids.add(item_id)

        if candidate_ids:
            return [
                self.store.get_item(item_id)
                for item_id in candidate_ids
                if item_id in self.store.items
            ]

        return list(self.store.items.values())

    def has_item_evidence(self, normalized_text: str) -> bool:
        """Return True when any index contains an explicit hit for *normalized_text*.

        Used by the slot-first resolution path to prioritise item queries that
        the store already knows about.
        """
        if not normalized_text:
            return False

        if self.store.find_entity(normalized_text, allowed_types={"item"}):
            return True

        if self.store.find_item_exact(normalized_text) is not None:
            return True

        if self.store.find_item_ids_by_alias(normalized_text):
            return True

        if self.store.find_item_ids_by_voice_label(normalized_text):
            return True

        return False
