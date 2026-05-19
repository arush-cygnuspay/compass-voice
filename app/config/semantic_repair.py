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

    # ── GPT Failure Isolation ────────────────────────────────────────────────
    # Master switch for GptSafeClient async call path (true = enabled).
    gpt_safe_client_enabled: bool = True
    # Default per-call timeout for GptSafeClient (ms). Capped at gpt_max_timeout_ms.
    gpt_timeout_ms: int = 700
    # Hard cap on any single GPT call (enforced by the safe-call wrapper).
    # Per-service timeouts may be lower; this is an absolute ceiling.
    gpt_max_timeout_ms: int = 1200
    # When True, any GPT failure falls back to the local deterministic path.
    gpt_fail_open_to_local: bool = True
    # Maximum characters of raw GPT output to store/log (safety bound).
    gpt_raw_output_log_max_chars: int = 2000

    # ── Elite Flow Stabilization: bucket-based GPT routing ───────────────────
    # Three buckets, each with an independent mode flag:
    #   "disabled" → bucket never triggers (default — safe)
    #   "shadow"   → GPT called, result logged only, never applied
    #   "inline"   → GPT called, result applied when validator approves
    #
    # Bucket 0 — idle_menu_item_resolution
    #   Triggers when state==IDLE, local intent UNKNOWN or low confidence,
    #   and text looks like a menu item query.
    bucket_0_mode: str = "disabled"
    # Bucket 2 — option_resolution
    #   Triggers when waiting state (size/side/modifier) and local matcher
    #   found no selections (option_match_failed=True).
    bucket_2_mode: str = "disabled"
    # Bucket 3 — multi_item_add_planning
    #   Triggers when state==IDLE and multiple ITEM slots or compound markers.
    bucket_3_mode: str = "disabled"
    # Shared timeout for all bucket GPT calls (ms).
    bucket_timeout_ms: int = 1200
    # Bucket 0 — per-call timeout (ms); separate from the shared bucket_timeout_ms.
    bucket_0_timeout_ms: int = 700
    # Bucket 0 — minimum GPT confidence for auto-apply (0.0–1.0).
    bucket_0_min_confidence: float = 0.70
    # Bucket 0 — maximum menu candidates sent to GPT (never the full menu).
    idle_item_menu_candidate_limit: int = 12
    # Bucket 0 — local add_item confidence threshold above which GPT is skipped.
    idle_item_high_conf_threshold: float = 0.85
    # Bucket 2 — per-call timeout (ms); separate from the shared bucket_timeout_ms.
    # Default 700 ms keeps total turn latency well under SLO.
    bucket_2_timeout_ms: int = 700
    # Bucket 2 — minimum GPT confidence for auto-apply (0.0–1.0).
    bucket_2_min_confidence: float = 0.70

    # ── Priority 6: Control-flow buckets (checkout / payment / order-type) ───
    # Bucket 5 — checkout_confirmation_resolution
    #   Triggers in idle/confirming_order when the utterance looks like a
    #   checkout or order-confirmation phrase.
    bucket_5_mode: str = "disabled"
    # Bucket 6 — order_type_change / pickup_delivery_initial
    #   Triggers when utterance contains a pickup/delivery phrase or the
    #   state is waiting_for_order_type.
    bucket_6_mode: str = "disabled"
    # Bucket payment — payment_permission_resolution
    #   Triggers in waiting_for_pickup_sms_permission when the utterance
    #   looks like a payment preference phrase.
    bucket_payment_mode: str = "disabled"
    # Shared timeout for all control-flow GPT calls (ms).
    control_flow_timeout_ms: int = 700
    # Minimum GPT confidence for auto-apply in control-flow buckets (0.0–1.0).
    control_flow_min_confidence: float = 0.70

    # ── SmartTurnPlanner (surgical GPT for risky turns) ──────────────────────
    # Enabled via SMART_TURN_PLANNER_ENABLED env var (default: false).
    # These config fields mirror env vars for test injection convenience;
    # the service also reads env vars directly for feature-flag safety.
    smart_turn_planner_enabled: bool = False
    # Timeout for SmartTurnPlanner GPT call (ms)
    smart_turn_planner_timeout_ms: int = 1200
    # Minimum plan confidence for safe application (0.0–1.0)
    smart_turn_planner_min_confidence: float = 0.75
    # Daily call budget (0 = unlimited)
    smart_turn_planner_daily_budget: int = 5000
    # Low-confidence threshold — triggers planner when local NLU is below this
    smart_turn_planner_low_confidence_threshold: float = 0.55

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
        _bucket_modes = {"disabled", "shadow", "inline"}
        if self.bucket_0_mode not in _bucket_modes:
            raise ValueError(
                f"Invalid bucket_0_mode {self.bucket_0_mode!r}. "
                "Allowed: ['disabled', 'shadow', 'inline']"
            )
        if self.bucket_2_mode not in _bucket_modes:
            raise ValueError(
                f"Invalid bucket_2_mode {self.bucket_2_mode!r}. "
                "Allowed: ['disabled', 'shadow', 'inline']"
            )
        if self.bucket_3_mode not in _bucket_modes:
            raise ValueError(
                f"Invalid bucket_3_mode {self.bucket_3_mode!r}. "
                "Allowed: ['disabled', 'shadow', 'inline']"
            )
        if self.bucket_5_mode not in _bucket_modes:
            raise ValueError(
                f"Invalid bucket_5_mode {self.bucket_5_mode!r}. "
                "Allowed: ['disabled', 'shadow', 'inline']"
            )
        if self.bucket_6_mode not in _bucket_modes:
            raise ValueError(
                f"Invalid bucket_6_mode {self.bucket_6_mode!r}. "
                "Allowed: ['disabled', 'shadow', 'inline']"
            )
        if self.bucket_payment_mode not in _bucket_modes:
            raise ValueError(
                f"Invalid bucket_payment_mode {self.bucket_payment_mode!r}. "
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
        # GPT Failure Isolation
        gpt_safe_client_enabled=os.getenv("GPT_SAFE_CLIENT_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
        gpt_timeout_ms=int(os.getenv("GPT_TIMEOUT_MS", "700")),
        gpt_max_timeout_ms=int(os.getenv("GPT_MAX_TIMEOUT_MS", "1200")),
        gpt_fail_open_to_local=os.getenv("GPT_FAIL_OPEN_TO_LOCAL", "true").lower() in {"1", "true", "yes", "on"},
        gpt_raw_output_log_max_chars=int(os.getenv("GPT_RAW_OUTPUT_LOG_MAX_CHARS", "2000")),
        # Elite Flow Stabilization: bucket modes
        bucket_0_mode=os.getenv("COMPASS_GPT_BUCKET_0_MODE", "disabled"),
        bucket_2_mode=os.getenv("COMPASS_GPT_BUCKET_2_MODE", "disabled"),
        bucket_3_mode=os.getenv("COMPASS_GPT_BUCKET_3_MODE", "disabled"),
        bucket_timeout_ms=int(os.getenv("COMPASS_GPT_BUCKET_TIMEOUT_MS", "1200")),
        bucket_0_timeout_ms=int(os.getenv("COMPASS_GPT_BUCKET_0_TIMEOUT_MS", "700")),
        bucket_0_min_confidence=float(os.getenv("COMPASS_GPT_BUCKET_0_MIN_CONFIDENCE", "0.70")),
        idle_item_menu_candidate_limit=int(os.getenv("COMPASS_IDLE_ITEM_MENU_CANDIDATE_LIMIT", "12")),
        idle_item_high_conf_threshold=float(os.getenv("COMPASS_IDLE_ITEM_HIGH_CONF_LOCAL_THRESHOLD", "0.85")),
        bucket_2_timeout_ms=int(os.getenv("COMPASS_GPT_BUCKET_2_TIMEOUT_MS", "700")),
        bucket_2_min_confidence=float(os.getenv("COMPASS_GPT_BUCKET_2_MIN_CONFIDENCE", "0.70")),
        # Priority 6: Control-flow buckets
        bucket_5_mode=os.getenv("COMPASS_GPT_BUCKET_5_CHECKOUT_MODE", "disabled"),
        bucket_6_mode=os.getenv("COMPASS_GPT_BUCKET_6_ORDER_TYPE_MODE", "disabled"),
        bucket_payment_mode=os.getenv("COMPASS_GPT_BUCKET_PAYMENT_PERMISSION_MODE", "disabled"),
        control_flow_timeout_ms=int(os.getenv("COMPASS_GPT_CONTROL_FLOW_TIMEOUT_MS", "700")),
        control_flow_min_confidence=float(os.getenv("COMPASS_GPT_CONTROL_FLOW_MIN_CONFIDENCE", "0.70")),
        # SmartTurnPlanner
        smart_turn_planner_enabled=os.getenv(
            "SMART_TURN_PLANNER_ENABLED", "false"
        ).lower() in {"1", "true", "yes", "on"},
        smart_turn_planner_timeout_ms=int(
            os.getenv("SMART_TURN_PLANNER_TIMEOUT_MS", "1200")
        ),
        smart_turn_planner_min_confidence=float(
            os.getenv("SMART_TURN_PLANNER_MIN_CONFIDENCE", "0.75")
        ),
        smart_turn_planner_daily_budget=int(
            os.getenv("SMART_TURN_PLANNER_DAILY_BUDGET", "5000")
        ),
        smart_turn_planner_low_confidence_threshold=float(
            os.getenv("SMART_TURN_PLANNER_LOW_CONFIDENCE_THRESHOLD", "0.55")
        ),
    )
