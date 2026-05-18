# app/nlu/semantic_repair/option_resolver_result.py
"""Result dataclass for the GPT option resolver (Phase 3).

Shadow-only contract when route_mode="shadow_gpt":
  safe_to_apply is always False; result is written to JSONL only.

Inline contract when route_mode="inline_gpt":
  safe_to_apply may be True when all selected_names validated and
  confidence >= min_confidence.  Handler applies the result.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptionResolverResult:
    """Result of one GPT option resolution attempt.

    Fields
    ------
    decision : str
        "select_option" — GPT selected at least one option from the list.
        "no_match"      — GPT found no matching option in the choices.
        "skipped"       — GPT was not called (mode disabled or budget exceeded).
        "error"         — GPT call failed or response could not be parsed.
    selected_names : tuple[str, ...]
        Option names as returned by GPT.  Validator maps these to modifier_ids.
        Empty when decision != "select_option".
    control_intent : str | None
        Detected control intent ("skip" | "done" | None) if GPT signalled
        the user wants to exit the modifier group rather than select an option.
    confidence : float
        GPT's self-reported confidence in range [0.0, 1.0].
    reason_code : str
        Short human-readable code: "fuzzy_match" | "phonetic_match" |
        "exact_match" | "no_match" | "not_called" | "parse_error".
    safe_to_apply : bool
        True only when:
          - decision == "select_option"
          - all selected_names resolved to valid modifier_ids in current group
          - confidence >= option_resolver_min_confidence
          - route_mode == "inline_gpt"
        Always False in shadow mode.
    gpt_called : bool
        Whether the OpenAI API was actually invoked.
    latency_ms : float | None
        Wall-clock time of the GPT call in milliseconds.
    parse_error : str | None
        Description of the parse failure if GPT returned invalid JSON.
    skipped_reason : str | None
        Why GPT was skipped:
          "mode_disabled"          — option_resolver_mode == "disabled"
          "missing_api_key"        — OPENAI_API_KEY not set
          "daily_budget_exceeded"  — daily call limit reached
          "no_choices"             — group has no choices to select from
          "not_called"             — never attempted
    route_mode : str
        The routing decision that was made:
          "no_gpt"      — routing policy decided not to call GPT
          "shadow_gpt"  — GPT called but result is never applied
          "inline_gpt"  — GPT called and result may be applied
    prompt_chars : int
        Character count of the prompt sent to GPT (for cost tracking).
    completion_chars : int
        Character count of the GPT response (for cost tracking).
    model : str | None
        The OpenAI model that was used.
    """

    decision: str = "skipped"
    selected_names: tuple[str, ...] = ()
    control_intent: str | None = None
    confidence: float = 0.0
    reason_code: str = "not_called"
    safe_to_apply: bool = False
    gpt_called: bool = False
    latency_ms: float | None = None
    parse_error: str | None = None
    skipped_reason: str | None = None
    route_mode: str = "no_gpt"
    prompt_chars: int = 0
    completion_chars: int = 0
    model: str | None = None


# Sentinel: option resolver was not invoked at all.
OPTION_RESOLVER_NOT_CALLED = OptionResolverResult(
    decision="skipped",
    skipped_reason="not_called",
    route_mode="no_gpt",
)
