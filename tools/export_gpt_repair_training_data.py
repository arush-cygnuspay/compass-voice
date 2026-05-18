#!/usr/bin/env python3
# tools/export_gpt_repair_training_data.py
"""Export GPT repair training candidates from the gpt_repair_turns.csv log.

Usage:
    python tools/export_gpt_repair_training_data.py [--csv PATH] [--out PATH]
    python tools/export_gpt_repair_training_data.py --dry-run

Reads gpt_repair_turns.csv and writes training-candidate rows to a new CSV.

A row is a training candidate when:
    - training_candidate == "1", OR
    - gpt_called == "1" and gpt_decision is not empty and not "no_repair"

Hard constraints enforced:
    - gpt_applied must always be "0" in phase 2 (invariant check)
    - Records with violations abort the export
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_CSV = _PROJECT_ROOT / "app" / "logs" / "gpt_repair_turns.csv"
_DEFAULT_OUT = _PROJECT_ROOT / "app" / "logs" / "training_candidates.csv"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _is_training_candidate(row: dict[str, str]) -> bool:
    if row.get("training_candidate") == "1":
        return True
    decision = row.get("gpt_decision", "")
    if row.get("gpt_called") == "1" and decision and decision != "no_repair":
        return True
    return False


def _assert_no_gpt_applied(row: dict[str, str]) -> None:
    if row.get("gpt_applied") == "1":
        raise ValueError(
            f"Invariant violation: gpt_applied=1 "
            f"(session={row.get('session_id')}, turn={row.get('turn_index')}). "
            "Phase 2 must never apply GPT suggestions."
        )


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GPT repair training data from CSV")
    parser.add_argument("--csv", default=str(_DEFAULT_CSV), help="Input gpt_repair_turns.csv")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="Output training CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, no file written")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[export] Loading {csv_path} …")
    all_rows = _load_csv(csv_path)
    print(f"[export] Loaded {len(all_rows)} total rows")

    candidates = [r for r in all_rows if _is_training_candidate(r)]
    print(f"[export] Training candidates: {len(candidates)} ({100*len(candidates)/max(len(all_rows),1):.1f}%)")

    violations = 0
    for r in candidates:
        try:
            _assert_no_gpt_applied(r)
        except ValueError as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            violations += 1

    if violations:
        print(f"[ERROR] {violations} invariant violation(s) — aborting export", file=sys.stderr)
        sys.exit(1)

    # Stats
    decision_counter: Counter[str] = Counter(r.get("gpt_decision") or "n/a" for r in candidates)
    reason_counter: Counter[str] = Counter(r.get("gpt_eligible_reason") or "n/a" for r in candidates)
    state_counter: Counter[str] = Counter(r.get("state_before") or "n/a" for r in candidates)
    confidence_vals: list[float] = []
    for r in candidates:
        v = r.get("gpt_confidence", "")
        try:
            if v:
                confidence_vals.append(float(v))
        except ValueError:
            pass

    print("\n[decision distribution]")
    for decision, count in sorted(decision_counter.items(), key=lambda x: -x[1]):
        print(f"  {decision:<35} : {count:>5}  ({100*count/max(len(candidates),1):>5.1f}%)")

    print("\n[eligibility reason distribution]")
    for reason, count in sorted(reason_counter.items(), key=lambda x: -x[1]):
        print(f"  {reason:<35} : {count:>5}  ({100*count/max(len(candidates),1):>5.1f}%)")

    print("\n[state distribution (top 10)]")
    for state, count in state_counter.most_common(10):
        print(f"  {state:<35} : {count:>5}  ({100*count/max(len(candidates),1):>5.1f}%)")

    if confidence_vals:
        avg_conf = sum(confidence_vals) / len(confidence_vals)
        print(
            f"\n[gpt confidence] avg={avg_conf:.3f} "
            f"p50={_percentile(confidence_vals, 50):.3f} "
            f"n={len(confidence_vals)}"
        )

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    if not candidates:
        print("\n[export] No candidates — nothing to write.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(candidates[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    print(f"\n[export] Written {len(candidates)} rows → {out_path}")


if __name__ == "__main__":
    main()
