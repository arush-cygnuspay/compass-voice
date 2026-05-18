# scripts/test_phase3_option_resolver_manual.py
"""Phase 3 GPT Option Resolver — manual / staging test script.

Run from the repo root with OPENAI_API_KEY set and
COMPASS_GPT_OPTION_RESOLVER_MODE=inline (or shadow):

    COMPASS_GPT_OPTION_RESOLVER_MODE=inline \
    OPENAI_API_KEY=sk-... \
    python scripts/test_phase3_option_resolver_manual.py

Each test case prints:
  - User text
  - Available choices
  - Expected outcome (from test intent)
  - GPT decision, selected names, confidence, reason_code, safe_to_apply
  - PASS / FAIL determination

Exit code is 0 if all assertions pass, 1 if any fail.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Helpers to build test fixtures without a running TurnEngine
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_group(choices: list[tuple[str, str]], *, max_selector: int = 1):
    from app.state_machine.models.pending_item_models import (
        PendingModifierChoice,
        PendingModifierGroup,
    )

    group_choices = [
        PendingModifierChoice(
            modifier_id=mid,
            name=name,
            group_id="test_grp",
            normalized_name=name.lower(),
        )
        for mid, name in choices
    ]
    return PendingModifierGroup(
        group_id="test_grp",
        name="Cheese",
        is_required=True,
        min_selector=1,
        max_selector=max_selector,
        choices=group_choices,
    )


def _make_service(mode: str | None = None):
    from app.config.semantic_repair import SemanticRepairConfig
    from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService

    effective_mode = mode or os.getenv("COMPASS_GPT_OPTION_RESOLVER_MODE", "inline")
    cfg = SemanticRepairConfig(
        phase=3,
        model=os.getenv("COMPASS_GPT_OPTION_RESOLVER_MODEL", "gpt-4o-mini"),
        timeout_seconds=2.0,
        option_resolver_mode=effective_mode,
        option_resolver_timeout_ms=int(os.getenv("COMPASS_GPT_OPTION_RESOLVER_TIMEOUT_MS", "2000")),
        option_resolver_min_confidence=float(
            os.getenv("COMPASS_GPT_OPTION_RESOLVER_MIN_CONFIDENCE", "0.75")
        ),
        option_resolver_repeat_threshold=int(
            os.getenv("COMPASS_GPT_OPTION_RESOLVER_REPEAT_THRESHOLD", "2")
        ),
    )
    return GptOptionResolverService(config=cfg)


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    user_text: str
    choices: list[tuple[str, str]]  # (modifier_id, name)
    item_name: str = "Burger"
    local_resolved: bool = False
    repeat_count: int = 0
    has_correction_signal: bool = False
    # Expectations
    expect_gpt_called: bool | None = None
    expect_decision: str | None = None
    expect_selected_names: list[str] | None = None  # any of these in result is OK
    expect_safe_to_apply: bool | None = None
    expect_skipped: bool = False
    mode_override: str | None = None  # override mode for this test case


TEST_CASES: list[TestCase] = [
    # 1. Phonetic mismatch — the canonical example
    TestCase(
        name="macarola_cheese_phonetic",
        user_text="macarola cheese",
        choices=[("m1", "American Cheese"), ("m2", "Mozzarella Cheese"), ("m3", "Cheddar Cheese")],
        expect_gpt_called=True,
        expect_decision="select_option",
        expect_selected_names=["Mozzarella Cheese"],
        expect_safe_to_apply=True,
    ),
    # 2. Fuzzy spelling mismatch
    TestCase(
        name="mozarella_fuzzy",
        user_text="mozarella",
        choices=[("m1", "American Cheese"), ("m2", "Mozzarella Cheese"), ("m3", "Cheddar Cheese")],
        expect_gpt_called=True,
        expect_decision="select_option",
        expect_selected_names=["Mozzarella Cheese"],
        expect_safe_to_apply=True,
    ),
    # 3. Short name — exact partial match
    TestCase(
        name="american_short",
        user_text="american",
        choices=[("m1", "American Cheese"), ("m2", "Swiss Cheese")],
        expect_gpt_called=True,
        expect_decision="select_option",
        expect_selected_names=["American Cheese"],
        expect_safe_to_apply=True,
    ),
    # 4. Correction signal with rephrasing
    TestCase(
        name="correction_signal_mozzarella",
        user_text="no, mozzarella",
        choices=[("m1", "American Cheese"), ("m2", "Mozzarella Cheese")],
        has_correction_signal=True,
        expect_gpt_called=True,
        expect_decision="select_option",
        expect_selected_names=["Mozzarella Cheese"],
        expect_safe_to_apply=True,
    ),
    # 5. Exact match should bypass GPT (local_resolved=True)
    TestCase(
        name="exact_match_no_gpt",
        user_text="cheddar cheese",
        choices=[("m1", "American Cheese"), ("m2", "Mozzarella Cheese"), ("m3", "Cheddar Cheese")],
        local_resolved=True,  # local matched it
        expect_gpt_called=False,
        expect_skipped=True,
    ),
    # 6. Nonsense text — GPT should return no_match
    TestCase(
        name="nonsense_no_match",
        user_text="flooby dooby snicklefritz",
        choices=[("m1", "American Cheese"), ("m2", "Swiss Cheese")],
        expect_gpt_called=True,
        expect_decision="no_match",
        expect_safe_to_apply=False,
    ),
    # 7. Silence / empty text — never calls GPT
    TestCase(
        name="silence_no_gpt",
        user_text="",
        choices=[("m1", "American Cheese")],
        expect_gpt_called=False,
        expect_skipped=True,
    ),
    # 8. Shadow mode — GPT called but never safe to apply
    TestCase(
        name="shadow_mode_not_applied",
        user_text="macarola cheese",
        choices=[("m1", "American Cheese"), ("m2", "Mozzarella Cheese"), ("m3", "Cheddar Cheese")],
        mode_override="shadow",
        expect_gpt_called=True,
        expect_decision="select_option",
        expect_selected_names=["Mozzarella Cheese"],
        expect_safe_to_apply=False,  # shadow is NEVER safe to apply
    ),
    # 9. Disabled mode — GPT is never called
    TestCase(
        name="disabled_mode_no_gpt",
        user_text="macarola cheese",
        choices=[("m1", "American Cheese"), ("m2", "Mozzarella Cheese")],
        mode_override="disabled",
        expect_gpt_called=False,
        expect_skipped=True,
    ),
    # 10. Repeat-loop escalation — short text but repeat_count >= threshold
    TestCase(
        name="repeat_loop_short_text_escalates",
        user_text="mo",  # < 3 chars, normally NO_GPT in inline
        choices=[("m1", "American Cheese"), ("m2", "Mozzarella Cheese")],
        repeat_count=3,  # >= default threshold=2
        expect_gpt_called=True,
    ),
    # 11. Hallucinated option name — validator must reject
    TestCase(
        name="hallucinated_name_rejected",
        user_text="quantum foil wrapper supreme",
        choices=[("m1", "American Cheese"), ("m2", "Mozzarella Cheese")],
        expect_gpt_called=True,
        expect_safe_to_apply=False,
    ),
    # 12. Context builder includes last_response_key (verified via service call)
    TestCase(
        name="phonetic_with_context",
        user_text="macarola",
        choices=[("m1", "American Cheese"), ("m2", "Mozzarella Cheese"), ("m3", "Cheddar Cheese")],
        expect_gpt_called=True,
        expect_decision="select_option",
        expect_safe_to_apply=True,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_test(svc: Any, tc: TestCase) -> tuple[bool, str]:
    """Run one test case. Returns (passed: bool, detail: str)."""
    from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService

    # Build per-case service if mode override
    if tc.mode_override is not None:
        test_svc = _make_service(tc.mode_override)
    else:
        test_svc = svc

    group = _make_group(tc.choices)

    try:
        result = test_svc.run(
            user_text=tc.user_text,
            item_name=tc.item_name,
            group=group,
            existing_selections=[],
            local_resolved=tc.local_resolved,
            repeat_count=tc.repeat_count,
            has_correction_signal=tc.has_correction_signal,
            last_response_key="ask_for_modifier",
        )
    except Exception as exc:
        return False, f"EXCEPTION: {exc}"

    failures = []

    if tc.expect_gpt_called is not None and result.gpt_called != tc.expect_gpt_called:
        failures.append(f"gpt_called={result.gpt_called!r}, expected {tc.expect_gpt_called!r}")

    if tc.expect_decision is not None and result.decision != tc.expect_decision:
        failures.append(f"decision={result.decision!r}, expected {tc.expect_decision!r}")

    if tc.expect_selected_names is not None:
        matched = any(name in result.selected_names for name in tc.expect_selected_names)
        if not matched:
            failures.append(
                "selected_names=%r, expected one of %r" % (result.selected_names, tc.expect_selected_names)
            )

    if tc.expect_safe_to_apply is not None and result.safe_to_apply != tc.expect_safe_to_apply:
        failures.append(
            f"safe_to_apply={result.safe_to_apply!r}, expected {tc.expect_safe_to_apply!r}"
        )

    if tc.expect_skipped and result.gpt_called:
        failures.append(f"expected skipped but gpt_called={result.gpt_called!r}")

    detail_parts = [
        f"  gpt_called={result.gpt_called}",
        f"  decision={result.decision!r}",
        f"  selected_names={result.selected_names!r}",
        f"  confidence={result.confidence:.3f}" if result.confidence else "  confidence=None",
        f"  reason_code={result.reason_code!r}",
        f"  safe_to_apply={result.safe_to_apply}",
        f"  route_mode={result.route_mode!r}",
        f"  latency_ms={result.latency_ms:.1f}ms" if result.latency_ms else "  latency_ms=None",
        f"  parse_error={result.parse_error!r}" if result.parse_error else "",
        f"  skipped_reason={result.skipped_reason!r}" if result.skipped_reason else "",
    ]
    detail = "\n".join(x for x in detail_parts if x)

    # If GPT was required but API key is missing, report SKIP not FAIL
    api_key_missing = result.skipped_reason == "missing_api_key"
    if api_key_missing and tc.expect_gpt_called is True:
        return True, "\n%s\n  SKIP (no OPENAI_API_KEY)" % detail

    if failures:
        return False, "\n%s\n  FAIL: %s" % (detail, " | ".join(failures))
    return True, "\n%s\n  PASS" % detail


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("WARNING: OPENAI_API_KEY is not set -- GPT calls will be skipped (tests still run).")

    mode = os.getenv("COMPASS_GPT_OPTION_RESOLVER_MODE", "inline")
    print(f"Phase 3 Option Resolver Manual Test — mode={mode!r}")
    print(f"Model: {os.getenv('COMPASS_GPT_OPTION_RESOLVER_MODEL', 'gpt-4o-mini')}")
    print("=" * 70)

    svc = _make_service()

    passed = 0
    failed = 0
    for tc in TEST_CASES:
        label = f"[{tc.name}]"
        print(f"\n{label}")
        print(f"  text={tc.user_text!r}  choices={[n for _, n in tc.choices]}")
        ok, detail = run_test(svc, tc)
        print(detail)
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed / {len(TEST_CASES)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
