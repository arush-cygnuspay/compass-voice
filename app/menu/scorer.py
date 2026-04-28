# app/menu/scorer.py
"""Item-label scoring for menu matching.

Owns all scoring thresholds (previously magic numbers scattered through
MenuRepository) and the label-scoring function.  Stateless — no store
dependency.  All thresholds are exposed as named class attributes so
tests can inspect or override them.
"""
from __future__ import annotations

from app.menu.models import MenuItem
from app.utils.item_matching import score_item, score_item_normalized

# ---------------------------------------------------------------------------
# Threshold constants (single source of truth; previously inline magic numbers)
# ---------------------------------------------------------------------------

# Within-candidates (CONFIRMING_ITEM path): unambiguous winner required
CANDIDATE_CLEAR_WINNER_THRESHOLD: float = 6.0
CANDIDATE_GAP_REQUIRED: float = 0.9
CANDIDATE_RATIO_REQUIRED: float = 1.15

# Slot-first resolution: NLU slot provides prior evidence → slightly more lenient
SLOT_CLEAR_WINNER_THRESHOLD: float = 5.8
SLOT_GAP_REQUIRED: float = 0.9
SLOT_RATIO_REQUIRED: float = 1.18
SLOT_CLOSE_BAND_RATIO: float = 0.92       # items within 92 % of best are "close"
SLOT_FALLBACK_SINGLE_THRESHOLD: float = 5.4  # promote sole survivor above this

# Free-text / availability resolution
FREE_STRONG_BAND_RATIO: float = 0.85       # items within 85 % of best are "strong"
FREE_CLEAR_WINNER_THRESHOLD: float = 6.0

# Side / modifier group resolution: constrained domain → more generous
GROUP_MATCH_BAND_RATIO: float = 0.90       # items within 90 % of best win

# Similarity / recovery suggestions
SIMILARITY_MINIMUM_THRESHOLD: float = 4.8

# Legacy resolve_item() path
LEGACY_RESOLVE_THRESHOLD: float = 6.5


class MenuScorer:
    """Stateless scorer for menu item labels.

    Scores how well a pre-normalized user query matches an item's name,
    aliases, and voice labels.  Threshold constants are class attributes
    so tests can read them without instantiating MenuRepository.
    """

    # Expose thresholds as class attributes for inspection / override.
    candidate_clear_winner: float = CANDIDATE_CLEAR_WINNER_THRESHOLD
    candidate_gap: float = CANDIDATE_GAP_REQUIRED
    candidate_ratio: float = CANDIDATE_RATIO_REQUIRED

    slot_clear_winner: float = SLOT_CLEAR_WINNER_THRESHOLD
    slot_gap: float = SLOT_GAP_REQUIRED
    slot_ratio: float = SLOT_RATIO_REQUIRED
    slot_close_band: float = SLOT_CLOSE_BAND_RATIO
    slot_fallback_single: float = SLOT_FALLBACK_SINGLE_THRESHOLD

    free_strong_band: float = FREE_STRONG_BAND_RATIO
    free_clear_winner: float = FREE_CLEAR_WINNER_THRESHOLD

    group_match_band: float = GROUP_MATCH_BAND_RATIO

    similarity_minimum: float = SIMILARITY_MINIMUM_THRESHOLD
    legacy_resolve: float = LEGACY_RESOLVE_THRESHOLD

    # ------------------------------------------------------------------

    def score_item_labels(self, normalized_text: str, item: MenuItem) -> float:
        """Score *normalized_text* against all labels of *item* (name, aliases, voice_labels)."""
        return max(
            score_item_normalized(normalized_text, item.normalized_name),
            max(
                (score_item(normalized_text, alias) for alias in item.normalized_aliases),
                default=0.0,
            ),
            max(
                (score_item(normalized_text, label) for label in item.voice_labels),
                default=0.0,
            ),
        )

    def score_labels(self, normalized_text: str, labels: tuple[str, ...]) -> float:
        """Score *normalized_text* against an arbitrary tuple of *labels*."""
        return max(
            (score_item(normalized_text, label) for label in labels),
            default=0.0,
        )
