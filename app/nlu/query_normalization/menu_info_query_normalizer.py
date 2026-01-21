# app/nlu/query_normalization/menu_info_query_normalizer.py

import re
from app.nlu.intent_patterns.info import (
    CATEGORY_ENTITY_PAT,
    BARE_ENTITY_PAT,
    WHICH_CATEGORY_PAT,
)
from app.nlu.query_normalization.query_normalizer import QueryNormalizer
from app.nlu.intent_resolution.intent import Intent


def _light_cleanup(text: str) -> str:
    """
    Lightweight normalization for menu info queries.

    Rules:
    - Lowercase
    - Trim whitespace
    - Collapse spaces
    - Preserve word structure (no aggressive stripping)
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


class MenuInfoQueryNormalizer(QueryNormalizer):
    """
    Extracts a clean entity candidate from ASK_MENU_INFO queries.

    This normalizer:
    - Removes ONLY linguistic wrappers
    - Preserves the entity surface form
    - Never guesses
    """

    def normalize(self, text: str, intent: Intent) -> str:
        cleaned = _light_cleanup(text)

        # -------------------------------------------------
        # 1️⃣ WHICH patterns (highest precision)
        # -------------------------------------------------
        m = WHICH_CATEGORY_PAT.search(cleaned)
        if m:
            return m.group("entity").strip()

        # -------------------------------------------------
        # 2️⃣ Explicit category capture patterns
        # -------------------------------------------------
        m = CATEGORY_ENTITY_PAT.search(cleaned)
        if m:
            return m.group("entity").strip()

        # -------------------------------------------------
        # 3️⃣ Bare plural category (e.g. "desserts")
        # -------------------------------------------------
        if BARE_ENTITY_PAT.fullmatch(cleaned):
            return cleaned

        # -------------------------------------------------
        # 4️⃣ Fallback (safe)
        # -------------------------------------------------
        return cleaned
