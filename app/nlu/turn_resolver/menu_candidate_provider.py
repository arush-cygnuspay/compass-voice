# app/nlu/turn_resolver/menu_candidate_provider.py
"""Menu candidate provider for idle item resolution.

Converts raw MenuStore / MenuIndexer lookups into compact dicts
suitable for GPT context (no full menu, no prices, capped count).

Safety contract
---------------
* Never returns the full menu — always capped at limit (default 12).
* Each candidate dict contains only: item_id, name, aliases, category,
  available_sizes, available_variants. No prices, no raw JSON.
* Never raises into callers — returns empty tuple on any error.
* O(log n) via index when MenuIndexer is available; bounded otherwise.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.menu.indexer import MenuIndexer
    from app.menu.store import MenuStore

_logger = logging.getLogger(__name__)

_DEFAULT_LIMIT: int = 12


class MenuCandidateProvider:
    """Extracts a bounded set of candidate menu items for a user utterance.

    Parameters
    ----------
    indexer:
        ``MenuIndexer`` over the live menu store.  When None the provider
        returns empty candidates (safe for tests / disabled mode).
    """

    def __init__(self, indexer: "MenuIndexer | None" = None) -> None:
        self._indexer = indexer

    def get_candidates(
        self,
        user_text: str,
        *,
        limit: int = _DEFAULT_LIMIT,
    ) -> tuple[dict, ...]:
        """Return up to *limit* menu candidates for *user_text*.

        Parameters
        ----------
        user_text:
            Normalized user utterance — used for index lookup.
        limit:
            Maximum candidates returned (default 12).

        Returns
        -------
        Tuple of candidate dicts safe for GPT consumption.
        Returns empty tuple when no indexer is set or on any error.
        """
        if not self._indexer or not user_text:
            return ()
        try:
            return self._fetch(user_text, limit)
        except Exception as exc:
            _logger.warning(
                "menu_candidate_provider_error",
                extra={"event": "menu_candidate_provider_error", "error": str(exc)[:200]},
            )
            return ()

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch(self, user_text: str, limit: int) -> tuple[dict, ...]:
        """Perform the actual index lookup and build compact candidate dicts."""
        raw_candidates = self._indexer.candidate_items(user_text)
        results: list[dict] = []
        seen_ids: set[str] = set()

        for item in raw_candidates[:limit * 2]:  # over-fetch then trim
            if len(results) >= limit:
                break
            item_id = getattr(item, "item_id", None)
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            candidate = _build_candidate(item)
            if candidate:
                results.append(candidate)

        return tuple(results)


def _build_candidate(item: Any) -> dict | None:
    """Build a compact, safe candidate dict from a MenuItem."""
    try:
        item_id = str(getattr(item, "item_id", "") or "").strip()
        name = str(getattr(item, "name", "") or "").strip()
        if not item_id or not name:
            return None

        aliases = _extract_aliases(item)
        available_sizes = _extract_sizes(item)
        available_variants = _extract_variants(item)
        category = _extract_category(item)

        return {
            "item_id": item_id,
            "name": name,
            "aliases": aliases,
            "category": category,
            "available_sizes": available_sizes,
            "available_variants": available_variants,
        }
    except Exception:
        return None


def _extract_aliases(item: Any) -> list[str]:
    """Extract alias strings from a MenuItem safely."""
    aliases = getattr(item, "aliases", None) or ()
    voice_labels = getattr(item, "voice_labels", None) or ()
    all_names: list[str] = []
    for alias in list(aliases) + list(voice_labels):
        s = str(alias).strip()
        if s:
            all_names.append(s)
    return all_names[:6]  # cap to keep payload small


def _extract_sizes(item: Any) -> list[str]:
    """Extract available size labels from a MenuItem's pricing variants."""
    pricing = getattr(item, "pricing", None)
    if not pricing:
        return []
    variants = getattr(pricing, "variants", None) or []
    sizes: list[str] = []
    for v in variants:
        label = str(getattr(v, "label", "") or "").strip()
        if label:
            sizes.append(label)
    return sizes[:6]


def _extract_variants(item: Any) -> list[str]:
    """Extract variant/size labels from a MenuItem's side groups."""
    # Variants for wings-like items are typically in pricing.variants
    # We reuse _extract_sizes since the structure is the same.
    return _extract_sizes(item)


def _extract_category(item: Any) -> str | None:
    """Extract category name from a MenuItem if available."""
    category = getattr(item, "category", None) or getattr(item, "category_name", None)
    if category:
        return str(category).strip() or None
    return None
