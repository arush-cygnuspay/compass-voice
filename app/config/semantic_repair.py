# app/config/semantic_repair.py
"""Typed semantic-repair / GPT shadow-mode settings loaded once from env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# Allowed values for COMPASS_GPT_CALL_MODE.
VALID_CALL_MODES: frozenset[str] = frozenset({
    "disabled",       # no GPT call at all
    "eligible_only",  # GPT only when RepairPolicy says eligible (current behavior)
    "all_shadow",     # GPT for every non-terminal customer turn, logging-only, never applied
    "all_apply_safe", # config stub — apply behavior not implemented in this PR
})


@dataclass(frozen=True)
class SemanticRepairConfig:
    """Immutable snapshot of semantic-repair settings.

    phase=0 → eligibility logging only, no GPT calls.
    phase=2 → GPT called in shadow mode; result logged, never applied.

    call_mode overrides phase-based behavior when explicitly set:
      disabled      → no GPT call regardless of phase
      eligible_only → GPT only when RepairPolicy is eligible (phase >= 2 required)
      all_shadow    → GPT for every non-terminal turn, result logged but never applied
      all_apply_safe→ stub only; apply behavior not implemented in Step 1

    call_mode=None uses legacy phase-based behavior (for backward-compat with
    existing tests that pass phase= but not call_mode=).
    """

    phase: int
    model: str
    timeout_seconds: float
    # Granular timeout in ms (takes precedence over timeout_seconds when set)
    timeout_ms: int = 350
    # SLO budget: if turn has already consumed this many ms before GPT call, skip
    slo_budget_ms: int = 1800
    # Daily call budget (0 = unlimited)
    daily_budget: int = 10000
    # Number of top-K sub-intent candidates to extract from NLU model
    top_k_intents: int = 4

    # ── Step 1 rollout additions ─────────────────────────────────────────────
    # None = legacy phase-based behavior (used by existing tests)
    call_mode: str | None = None
    # Timeout for all_shadow background calls (ms); separate from inline timeout
    shadow_timeout_ms: int = 2000
    # Fallback response is applied only when this is True AND call_mode allows it
    apply_fallbacks: bool = False

    # ── Phase 1: GPT ADD_ITEM extractor (shadow mode only) ──────────────────
    # "disabled" → extractor never runs; "shadow" → runs but result is log-only
    add_item_mode: str = "disabled"
    # Per-call timeout for the ADD_ITEM extractor (ms)
    add_item_timeout_ms: int = 350
    # Minimum normalised text length before extractor is eligible
    add_item_min_text_len: int = 3
    # Maximum items[] entries the extractor may return per turn
    add_item_max_items_per_turn: int = 8

    # ── Phase 3: GPT Option Resolver (inline modifier/option matching) ───────
    # "disabled" → option resolver never runs (default — safe)
    # "shadow"   → GPT called, result logged only, never applied to cart
    # "inline"   → GPT called, result applied when validator says safe_to_apply=True
    option_resolver_mode: str = "disabled"
    # Per-call timeout for the option resolver (ms)
    option_resolver_timeout_ms: int = 1200
    # Minimum GPT confidence for safe_to_apply=True (0.0–1.0)
    option_resolver_min_confidence: float = 0.75
    # How many consecutive failed reprompts trigger INLINE_GPT (repeat-loop recovery)
    option_resolver_repeat_threshold: int = 2

    # ── Phase 4: GPT Add-Item Planner (inline multi-item / complex utterance) ─
    # "disabled" → planner never runs (default — safe)
    # "shadow"   → GPT called for complex utterances, result logged only, never applied
    # "inline"   → GPT called; result applied when apply gate approves
    add_item_planner_mode: str = "disabled"
    # Per-call timeout for the add-item planner (ms)
    add_item_planner_timeout_ms: int = 1800
    # Minimum GPT confidence for apply gate approval (0.0–1.0)
    add_item_planner_min_confidence: float = 0.75
    # Maximum candidate menu items sent to GPT (not full menu)
    add_item_planner_max_item_candidates: int = 10
    # Maximum option names per modifier/side group sent to GPT
    add_item_planner_max_option_candidates: int = 20

    def __post_init__(self) -> None:
        if self.call_mode is not None and self.call_mode not in VALID_CALL_MODES:
            raise ValueError(
                f"Invalid call_mode {self.call_mode!r}. "
                f"Allowed: {sorted(VALID_CALL_MODES)}"
            )
        if self.add_item_mode not in {"disabled", "shadow"}:
            raise ValueError(
                f"Invalid add_item_mode {self.add_item_mode!r}. "
                "Allowed: ['disabled', 'shadow']"
            )
        if self.option_resolver_mode not in {"disabled", "shadow", "inline"}:
            raise ValueError(
                f"Invalid option_resolver_mode {self.option_resolver_mode!r}. "
                "Allowed: ['disabled', 'shadow', 'inline']"
            )
        if self.add_item_planner_mode not in {"disabled", "shadow", "inline"}:
            raise ValueError(
                f"Invalid add_item_planner_mode {self.add_item_planner_mode!r}. "
                "Allowed: ['disabled', 'shadow', 'inline']"
            )

    @property
    def effective_call_mode(self) -> str:
        """Resolve the effective call mode, honouring legacy phase= behavior."""
        if self.call_mode is not None:
            return self.call_mode
        # Legacy: derive from phase
        if self.phase < 2:
            return "disabled"
        return "eligible_only"


@lru_cache(maxsize=1)
def get_semantic_repair_config() -> SemanticRepairConfig:
    """Return singleton SemanticRepairConfig, loading env vars on first call.

    The API key is intentionally NOT stored here — it is read directly from
    the environment inside GptRepairService at call time to ensure it is never
    serialised, logged, or included in any diagnostic record.
    """
    timeout_ms = int(os.getenv("COMPASS_GPT_TIMEOUT_MS", "350"))
    # Read with no default: None means unset → legacy phase-based behavior via
    # effective_call_mode.  Explicit "disabled" is honoured as-is.
    call_mode_raw = os.getenv("COMPASS_GPT_CALL_MODE")
    # Silently ignore unrecognised values to avoid crashing misconfigured
    # deployments; an unrecognised value is treated as unset (None → legacy).
    call_mode = call_mode_raw if (call_mode_raw and call_mode_raw in VALID_CALL_MODES) else None
    return SemanticRepairConfig(
        phase=int(os.getenv("COMPASS_GPT_REPAIR_PHASE", "0")),
        model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
        timeout_seconds=float(os.getenv("COMPASS_GPT_REPAIR_TIMEOUT_SECONDS", str(timeout_ms / 1000))),
        timeout_ms=timeout_ms,
        slo_budget_ms=int(os.getenv("COMPASS_GPT_SLO_BUDGET_MS", "1800")),
        daily_budget=int(os.getenv("COMPASS_GPT_DAILY_BUDGET", "10000")),
        top_k_intents=int(os.getenv("COMPASS_GPT_TOP_K_INTENTS", "4")),
        call_mode=call_mode,
        shadow_timeout_ms=int(os.getenv("COMPASS_GPT_SHADOW_TIMEOUT_MS", "2000")),
        apply_fallbacks=os.getenv("COMPASS_GPT_APPLY_FALLBACKS", "false").lower() == "true",
        # Phase 1: ADD_ITEM extractor
        add_item_mode=os.getenv("COMPASS_GPT_ADD_ITEM_MODE", "disabled"),
        add_item_timeout_ms=int(os.getenv("COMPASS_GPT_ADD_ITEM_TIMEOUT_MS", "350")),
        add_item_min_text_len=int(os.getenv("COMPASS_GPT_ADD_ITEM_MIN_TEXT_LEN", "3")),
        add_item_max_items_per_turn=int(
            os.getenv("COMPASS_GPT_ADD_ITEM_MAX_ITEMS_PER_TURN", "8")
        ),
        # Phase 3: Option Resolver
        option_resolver_mode=os.getenv("COMPASS_GPT_OPTION_RESOLVER_MODE", "disabled"),
        option_resolver_timeout_ms=int(
            os.getenv("COMPASS_GPT_OPTION_RESOLVER_TIMEOUT_MS", "1200")
        ),
        option_resolver_min_confidence=float(
            os.getenv("COMPASS_GPT_OPTION_RESOLVER_MIN_CONFIDENCE", "0.75")
        ),
        option_resolver_repeat_threshold=int(
            os.getenv("COMPASS_GPT_OPTION_RESOLVER_REPEAT_THRESHOLD", "2")
        ),
        # Phase 4: Add-Item Planner
        add_item_planner_mode=os.getenv("COMPASS_GPT_ADD_ITEM_PLANNER_MODE", "disabled"),
        add_item_planner_timeout_ms=int(
            os.getenv("COMPASS_GPT_ADD_ITEM_PLANNER_TIMEOUT_MS", "1800")
        ),
        add_item_planner_min_confidence=float(
            os.getenv("COMPASS_GPT_ADD_ITEM_PLANNER_MIN_CONFIDENCE", "0.75")
        ),
        add_item_planner_max_item_candidates=int(
            os.getenv("COMPASS_GPT_ADD_ITEM_PLANNER_MAX_ITEM_CANDIDATES", "10")
        ),
        add_item_planner_max_option_candidates=int(
            os.getenv("COMPASS_GPT_ADD_ITEM_PLANNER_MAX_OPTION_CANDIDATES", "20")
        ),
    )
