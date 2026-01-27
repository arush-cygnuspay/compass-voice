# app/nlu/query_normalization/item_query_normalizer.py
import re
from app.nlu.query_normalization.base import basic_cleanup
from app.nlu.intent_patterns.add_item import *
from app.nlu.intent_patterns.remove_item import REMOVE_FILLER_PAT
from app.nlu.query_normalization.query_normalizer import QueryNormalizer
from app.nlu.intent_resolution.intent import Intent

class ItemQueryNormalizer(QueryNormalizer):
    """
    Normalizes text when resolving a MENU ITEM.
    Used in IDLE / ADD_ITEM / REMOVE_ITEM phase.
    """

    def normalize(self, text: str, intent: Intent) -> str:
        cleaned = basic_cleanup(text)

        # Remove add/remove intent fluff
        cleaned = FILLER_PAT.sub("", cleaned)
        cleaned = REMOVE_FILLER_PAT.sub("", cleaned)

        # Normalize multi-item separators
        cleaned = ITEM_SEPARATOR_PAT.sub(" ", cleaned)

        # Token-level semantic cleanup
        tokens = cleaned.split()
        semantic_tokens = [
            t for t in tokens
            if t not in MENU_STOPWORDS
        ]

        cleaned = " ".join(semantic_tokens)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned
