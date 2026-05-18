#!/usr/bin/env python3
# tools/gpt_repair_replay.py
"""Replay turns from realtime_turn_latency.csv through the GPT repair verifier.

Usage:
    # Eligibility-only (no GPT calls)
    python tools/gpt_repair_replay.py [--csv PATH] [--limit N]

    # Shadow-mode with live GPT calls (requires OPENAI_API_KEY)
    python tools/gpt_repair_replay.py --phase 2 [--csv PATH] [--limit N]

    # Read pre-computed JSONL instead of replaying live
    python tools/gpt_repair_replay.py --jsonl [PATH]

Reads each row from the CSV (or JSONL), builds a minimal NLUResult and
IntentResult from the logged fields, runs RepairPolicy + GptRepairService in
shadow mode, and prints aggregate statistics.

Output:
    - Per-turn eligibility summary (--verbose mode)
    - Aggregate stats: total rows, eligible %, reason distribution
    - GPT decision distribution (when calls are made or JSONL is read)
    - Timing percentiles (p50/p95/max)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.semantic_repair import SemanticRepairConfig
from app.nlu.intent_resolution.intent import Intent
from app.nlu.intent_resolution.intent_mapping import SUB_INTENT_TO_INTENT
from app.nlu.intent_resolution.intent_result import IntentResult
from app.nlu.nlu_result import NLUResult
from app.nlu.semantic_repair.repair_service import GptRepairService
from app.state_machine.models.conversation_state import ConversationState


_DEFAULT_CSV = _PROJECT_ROOT / "app" / "logs" / "realtime_turn_latency.csv"
_DEFAULT_GPT_CSV = _PROJECT_ROOT / "app" / "logs" / "gpt_repair_turns.csv"


def _parse_intent(intent_value: str) -> Intent:
    if not intent_value:
        return Intent.UNKNOWN
    mapped = SUB_INTENT_TO_INTENT.get(intent_value.strip())
    return mapped if mapped is not None else Intent.UNKNOWN


def _parse_state(state_value: str) -> ConversationState:
    try:
        return ConversationState(state_value.strip())
    except ValueError:
        return ConversationState.IDLE


def _row_to_nlu(row: dict[str, Any]) -> NLUResult:
    normalized = row.get("normalized_text") or row.get("cleaned_text") or ""
    raw = row.get("customer_said") or normalized
    sub_intent = row.get("intent_sub_intent") or ""
    effective = _parse_intent(sub_intent)
    conf_str = row.get("intent_confidence") or "0.0"
    try:
        conf = float(conf_str)
    except (ValueError, TypeError):
        conf = 0.0
    return NLUResult(
        effective_intent=effective,
        intent_confidence=conf,
        raw_text=raw,
        normalized_text=normalized,
    )


def _row_to_intent_result(row: dict[str, Any]) -> IntentResult:
    sub_intent = row.get("intent_sub_intent") or ""
    intent = _parse_intent(sub_intent)
    return IntentResult(intent=intent, raw_text=row.get("cleaned_text") or "")


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(len(values_sorted) * p / 100)
    return values_sorted[min(idx, len(values_sorted) - 1)]


def _run_csv_replay(args: argparse.Namespace, svc: GptRepairService) -> None:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows_processed = 0
    rows_eligible = 0
    reason_counter: Counter[str] = Counter()
    eligible_latencies_ms: list[float] = []
    gpt_decision_counter: Counter[str] = Counter()
    gpt_calls_made = 0
    gpt_request_latencies_ms: list[float] = []

    print(f"[replay] Phase={args.phase} | CSV={csv_path} | Limit={args.limit or 'all'}")
    print("-" * 60)

    t_start = time.perf_counter()

    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if args.limit and rows_processed >= args.limit:
                break

            nlu = _row_to_nlu(row)
            intent_result = _row_to_intent_result(row)
            state_str = row.get("state_before") or "idle"
            state = _parse_state(state_str)

            t0 = time.perf_counter()
            try:
                analysis, result = svc.run(
                    nlu=nlu,
                    intent_result=intent_result,
                    state=state,
                )
            except Exception as exc:
                print(f"  [ERROR] row {rows_processed}: {exc}")
                rows_processed += 1
                continue

            elapsed = (time.perf_counter() - t0) * 1000.0
            rows_processed += 1
            reason_counter[analysis.reason] += 1

            if analysis.gpt_repair_eligible:
                rows_eligible += 1
                eligible_latencies_ms.append(elapsed)

            if result is not None and hasattr(result, "decision") and result.model is not None:
                gpt_calls_made += 1
                gpt_decision_counter[result.decision] += 1
                if result.request_ms is not None:
                    gpt_request_latencies_ms.append(result.request_ms)

            if args.verbose and analysis.gpt_repair_eligible:
                print(
                    f"  [ELIGIBLE] row={rows_processed} "
                    f"state={state_str[:20]} "
                    f"intent={intent_result.intent.value} "
                    f"reason={analysis.reason} "
                    f"elapsed={elapsed:.1f}ms"
                )

    wall_ms = (time.perf_counter() - t_start) * 1000.0
    _print_csv_stats(
        rows_processed=rows_processed,
        rows_eligible=rows_eligible,
        gpt_calls_made=gpt_calls_made,
        wall_ms=wall_ms,
        reason_counter=reason_counter,
        gpt_decision_counter=gpt_decision_counter,
        eligible_latencies_ms=eligible_latencies_ms,
        gpt_request_latencies_ms=gpt_request_latencies_ms,
    )


def _print_csv_stats(
    *,
    rows_processed: int,
    rows_eligible: int,
    gpt_calls_made: int,
    wall_ms: float,
    reason_counter: Counter[str],
    gpt_decision_counter: Counter[str],
    eligible_latencies_ms: list[float],
    gpt_request_latencies_ms: list[float],
) -> None:
    print("-" * 60)
    print(f"[stats] Rows processed   : {rows_processed}")
    print(f"[stats] Eligible turns   : {rows_eligible} ({100*rows_eligible/max(rows_processed,1):.1f}%)")
    print(f"[stats] GPT calls made   : {gpt_calls_made}")
    print(f"[stats] Wall time        : {wall_ms:.0f} ms ({wall_ms/max(rows_processed,1):.2f} ms/row)")

    print("\n[reason distribution]")
    for reason, count in sorted(reason_counter.items(), key=lambda x: -x[1]):
        print(f"  {reason:<35} : {count:>5}  ({100*count/max(rows_processed,1):>5.1f}%)")

    if gpt_calls_made:
        print("\n[gpt decision distribution]")
        for decision, count in sorted(gpt_decision_counter.items(), key=lambda x: -x[1]):
            print(f"  {decision:<35} : {count:>5}  ({100*count/max(gpt_calls_made,1):>5.1f}%)")

        if gpt_request_latencies_ms:
            print("\n[gpt request latency (ms)]")
            print(f"  p50 : {_percentile(gpt_request_latencies_ms, 50):.2f}")
            print(f"  p95 : {_percentile(gpt_request_latencies_ms, 95):.2f}")
            print(f"  max : {max(gpt_request_latencies_ms):.2f}")

    if eligible_latencies_ms:
        print("\n[eligibility check latency (ms)]")
        print(f"  p50 : {_percentile(eligible_latencies_ms, 50):.2f}")
        print(f"  p95 : {_percentile(eligible_latencies_ms, 95):.2f}")
        print(f"  max : {max(eligible_latencies_ms):.2f}")


def _run_gpt_csv_stats(gpt_csv_path: Path, args: argparse.Namespace) -> None:
    """Read pre-computed gpt_repair_turns.csv and print aggregate metrics."""
    if not gpt_csv_path.exists():
        print(f"[ERROR] GPT CSV not found: {gpt_csv_path}", file=sys.stderr)
        sys.exit(1)

    with gpt_csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("[INFO] No records found in GPT CSV.")
        return

    print(f"[gpt-csv] Path={gpt_csv_path} | Rows={len(rows)}")
    print("-" * 60)

    eligible = [r for r in rows if r.get("gpt_repair_eligible") == "1"]
    called = [r for r in rows if r.get("gpt_called") == "1"]
    training = [r for r in rows if r.get("training_candidate") == "1"]

    reason_counter: Counter[str] = Counter(
        r.get("gpt_eligible_reason") or "n/a" for r in rows
    )
    decision_counter: Counter[str] = Counter(
        r.get("gpt_decision") or "n/a" for r in called
    )
    total_ms_vals: list[float] = []
    confidence_vals: list[float] = []
    for r in called:
        try:
            if r.get("gpt_total_ms"):
                total_ms_vals.append(float(r["gpt_total_ms"]))
            if r.get("gpt_confidence"):
                confidence_vals.append(float(r["gpt_confidence"]))
        except ValueError:
            pass

    print(f"[stats] Total rows       : {len(rows)}")
    print(f"[stats] Eligible         : {len(eligible)} ({100*len(eligible)/max(len(rows),1):.1f}%)")
    print(f"[stats] GPT called       : {len(called)} ({100*len(called)/max(len(rows),1):.1f}%)")
    print(f"[stats] Training cands   : {len(training)} ({100*len(training)/max(len(rows),1):.1f}%)")

    print("\n[reason distribution]")
    for reason, count in sorted(reason_counter.items(), key=lambda x: -x[1]):
        print(f"  {reason:<35} : {count:>5}  ({100*count/max(len(rows),1):>5.1f}%)")

    if called:
        print("\n[gpt decision distribution]")
        for decision, count in sorted(decision_counter.items(), key=lambda x: -x[1]):
            print(f"  {decision:<35} : {count:>5}  ({100*count/max(len(called),1):>5.1f}%)")

        if total_ms_vals:
            print("\n[gpt total_ms (ms)]")
            print(f"  p50 : {_percentile(total_ms_vals, 50):.2f}")
            print(f"  p95 : {_percentile(total_ms_vals, 95):.2f}")
            print(f"  max : {max(total_ms_vals):.2f}")

        if confidence_vals:
            avg_conf = sum(confidence_vals) / len(confidence_vals)
            print(f"\n[gpt confidence] avg={avg_conf:.3f} n={len(confidence_vals)}")

    if args.verbose:
        print("\n[verbose] First 10 eligible rows:")
        for r in eligible[:10]:
            print(
                f"  session={r.get('session_id','?')[:8]} "
                f"turn={r.get('turn_index','?')} "
                f"state={r.get('state_before','?')[:20]} "
                f"reason={r.get('gpt_eligible_reason','?')} "
                f"called={r.get('gpt_called')} "
                f"decision={r.get('gpt_decision','?')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay or inspect GPT repair logs")
    parser.add_argument("--csv", default=str(_DEFAULT_CSV), help="Path to CSV log")
    parser.add_argument(
        "--gpt-csv",
        nargs="?",
        const=str(_DEFAULT_GPT_CSV),
        default=None,
        dest="gpt_csv",
        help="Read pre-computed GPT repair CSV instead of live replay (optional path)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0=all)")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 2], help="GPT phase for CSV replay")
    parser.add_argument("--verbose", action="store_true", help="Print per-turn details")
    args = parser.parse_args()

    if args.gpt_csv is not None:
        _run_gpt_csv_stats(Path(args.gpt_csv), args)
        return

    cfg = SemanticRepairConfig(
        phase=args.phase,
        model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
        timeout_seconds=float(os.getenv("COMPASS_GPT_REPAIR_TIMEOUT_SECONDS", "1.0")),
    )
    svc = GptRepairService(config=cfg)
    _run_csv_replay(args, svc)


if __name__ == "__main__":
    main()
