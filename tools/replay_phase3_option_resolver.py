#!/usr/bin/env python3
# tools/replay_phase3_option_resolver.py
"""Phase 3.5 Offline Replay + Evaluation Harness CLI.

Replays captured turn logs or fixture turns through the Phase 3 option
resolver decision path and compares behavior across GPT modes.

No production state (cart, session, FSM, order) is mutated.
GPT is never called unless --use-live-gpt is explicitly set.

Usage
-----
# Replay JSONL log in shadow mode (no GPT):
python tools/replay_phase3_option_resolver.py \\
    --input app/logs/gpt_repair_turns.jsonl \\
    --output reports/phase3_replay_report.jsonl \\
    --mode shadow \\
    --max-turns 500 \\
    --filter-state WAITING_FOR_MODIFIER

# Replay built-in fixtures only:
python tools/replay_phase3_option_resolver.py \\
    --fixtures-only \\
    --mode inline \\
    --output reports/fixture_replay.jsonl

# Dry run (no file written):
python tools/replay_phase3_option_resolver.py \\
    --fixtures-only --mode disabled --dry-run

# Live GPT replay (requires OPENAI_API_KEY):
python tools/replay_phase3_option_resolver.py \\
    --input app/logs/gpt_repair_turns.jsonl \\
    --mode inline \\
    --use-live-gpt \\
    --max-turns 50 \\
    --output reports/live_replay.jsonl

Output
------
  <output>.jsonl   — One ReplayResult JSON object per line
  <output summary> — Markdown summary alongside the JSONL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.nlu.semantic_repair.option_resolver_replay import (
    BUILT_IN_FIXTURES,
    Phase3OptionResolverReplayHarness,
    ReplaySummaryBuilder,
)


def _build_config(mode: str) -> "object":
    from app.config.semantic_repair import SemanticRepairConfig

    return SemanticRepairConfig(
        phase=int(os.getenv("COMPASS_GPT_PHASE", "3")),
        model=os.getenv("COMPASS_GPT_MODEL", "gpt-4o-mini"),
        timeout_seconds=float(os.getenv("COMPASS_GPT_REPAIR_TIMEOUT_SECONDS", "2.0")),
        option_resolver_mode=mode,
        option_resolver_timeout_ms=int(
            os.getenv("COMPASS_GPT_OPTION_RESOLVER_TIMEOUT_MS", "2000")
        ),
        option_resolver_min_confidence=float(
            os.getenv("COMPASS_GPT_OPTION_RESOLVER_MIN_CONFIDENCE", "0.75")
        ),
        option_resolver_repeat_threshold=int(
            os.getenv("COMPASS_GPT_OPTION_RESOLVER_REPEAT_THRESHOLD", "2")
        ),
    )


def _resolve_output_path(output_arg: str | None, mode: str) -> Path:
    if output_arg:
        return Path(output_arg)
    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / f"phase3_replay_{mode}.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3.5 Option Resolver Offline Replay Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        default=None,
        help="Path to input JSONL log (gpt_repair_turns.jsonl). "
             "If omitted and --fixtures-only not set, uses app/logs/gpt_repair_turns.jsonl.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Path for output replay JSONL report. "
             "Defaults to reports/phase3_replay_<mode>.jsonl.",
    )
    parser.add_argument(
        "--mode",
        choices=["disabled", "shadow", "inline"],
        default="shadow",
        help="GPT mode for replay (default: shadow).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N turns (0 = no limit).",
    )
    parser.add_argument(
        "--filter-state",
        metavar="STATE",
        default="waiting_for_modifier",
        help="Only replay turns with this state_before value "
             "(default: waiting_for_modifier). Pass '' to replay all states.",
    )
    parser.add_argument(
        "--filter-response-key",
        metavar="KEY",
        default=None,
        help="Only replay turns with this response_key_before value (optional).",
    )
    parser.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Run only the 7 built-in fixture turns, skip JSONL input.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run replay but do not write output files. "
             "Prints summary to stdout.",
    )
    parser.add_argument(
        "--use-live-gpt",
        action="store_true",
        default=False,
        help="Allow real OpenAI calls. Requires OPENAI_API_KEY. "
             "Default is False — GPT is never called without this flag.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-turn results to stdout.",
    )

    args = parser.parse_args(argv)
    mode = args.mode

    if args.use_live_gpt and not os.getenv("OPENAI_API_KEY"):
        print(
            "[ERROR] --use-live-gpt requires OPENAI_API_KEY to be set.",
            file=sys.stderr,
        )
        return 1

    cfg = _build_config(mode)
    harness = Phase3OptionResolverReplayHarness(config=cfg, use_live_gpt=args.use_live_gpt)
    summary = ReplaySummaryBuilder(mode=mode)

    output_path = _resolve_output_path(args.output, mode)
    summary_path = output_path.with_suffix(".summary.md")

    # Collect results
    replay_lines: list[str] = []

    def _process(results_iter):
        for result in results_iter:
            summary.add(result)
            replay_lines.append(json.dumps(result.to_dict(), ensure_ascii=False))
            if args.verbose:
                print(
                    f"  [{result.route_mode:>12}] {result.user_text[:40]!r:42}"
                    f" gpt={result.gpt_called} "
                    f"decision={result.gpt_decision or '-':15} "
                    f"would_apply={result.would_apply}"
                )

    if args.fixtures_only:
        print(f"[replay] Running {len(BUILT_IN_FIXTURES)} built-in fixtures in mode={mode!r}")
        _process(harness.replay_fixtures(BUILT_IN_FIXTURES, mode=mode))
    else:
        input_path = args.input or str(
            _PROJECT_ROOT / "app" / "logs" / "gpt_repair_turns.jsonl"
        )
        if not Path(input_path).exists():
            print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
            print(
                "  Tip: use --fixtures-only to run without a log file.",
                file=sys.stderr,
            )
            return 1

        filter_state = args.filter_state.strip() if args.filter_state else None
        print(
            f"[replay] Input={input_path} | Mode={mode!r} | "
            f"MaxTurns={args.max_turns or 'all'} | "
            f"FilterState={filter_state or 'all'}"
        )
        _process(
            harness.replay_jsonl(
                input_path,
                mode=mode,
                max_turns=args.max_turns,
                filter_state=filter_state,
                filter_response_key=args.filter_response_key,
            )
        )

    # Print summary
    summary_dict = summary.to_dict()
    summary_md = summary.to_markdown()

    print("\n" + "=" * 70)
    print(f"[summary] Total replayed : {summary_dict['total_turns']}")
    print(f"[summary] GPT called     : {summary_dict['gpt_called']}")
    print(f"[summary] Inline cands   : {summary_dict['inline_candidate_count']}")
    print(f"[summary] Validator pass : {summary_dict['validator_pass_count']}")
    print(f"[summary] Would apply    : {summary_dict['would_apply_count']}")
    print(f"[summary] Errors         : {summary_dict['error_count']}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        print(summary_md)
        return 0

    # Write output files
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for line in replay_lines:
            fh.write(line + "\n")

    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(summary_md)

    print(f"\n[output] Replay JSONL   : {output_path}")
    print(f"[output] Summary MD     : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
