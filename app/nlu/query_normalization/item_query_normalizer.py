# app/nlu/query_normalization/item_query_normalizer.py
import re
from app.nlu.query_normalization.base import basic_cleanup
from app.nlu.intent_patterns.add_item import (
    FILLER_PAT,
    ITEM_SEPARATOR_PAT,
)
from app.nlu.query_normalization.query_normalizer import QueryNormalizer
from app.nlu.intent_resolution.intent import Intent

class ItemQueryNormalizer(QueryNormalizer):
    """
    Normalizes text when resolving a MENU ITEM.
    Used in IDLE / ADD_ITEM phase.
    """

    def normalize(self, text: str, intent: Intent) -> str:
        cleaned = basic_cleanup(text)

        # Remove add-intent fillers
        cleaned = FILLER_PAT.sub("", cleaned)

        # Normalize multi-item separators
        cleaned = ITEM_SEPARATOR_PAT.sub(" ", cleaned)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
