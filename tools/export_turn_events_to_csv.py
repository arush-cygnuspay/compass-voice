#!/usr/bin/env python3
# tools/export_turn_events_to_csv.py
"""Export canonical turn_events.jsonl to derived CSV files.

Reads ``logs/current/turn_events.jsonl`` (one JSON object per line) and
writes four derived CSVs to ``logs/derived/``:

  nlu_log.csv             — local NLU decision per turn.
  gpt_repair_turns.csv    — GPT repair decision per turn (all turns, not just eligible).
  realtime_turn_latency.csv — timing breakdown per turn.
  training_candidates.csv — turns marked as training candidates.

These CSVs are DERIVED outputs.  The canonical source of truth is
turn_events.jsonl.  Do not edit the CSVs — re-run this script to regenerate.

Usage
-----
    python tools/export_turn_events_to_csv.py [--input PATH] [--output-dir PATH]

Options
-------
--input PATH        Path to turn_events.jsonl.
                    Default: logs/current/turn_events.jsonl
--output-dir PATH   Directory for derived CSVs.
                    Default: logs/derived
--limit N           Process at most N lines (useful for large files).
--since UTC         ISO-8601 timestamp; skip records before this time.
--verbose           Print progress to stderr.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

NLU_COLUMNS: list[str] = [
    "timestamp_utc",
    "session_id",
    "turn_index",
    "state_before",
    "state_after",
    "asr_normalized_text",
    "intent_main",
    "intent_sub_intent",
    "intent_effective",
    "confidence",
    "slot_model_ran",
    "slots_json",
    "route_allowed",
    "route_reject_reason",
    "response_key",
    "fallback_triggered",
    "reprompt_count",
    "user_repeated",
]

GPT_REPAIR_COLUMNS: list[str] = [
    "timestamp_utc",
    "session_id",
    "turn_index",
    "state_before",
    "asr_normalized_text",
    "local_intent",
    "local_confidence",
    "gpt_eligible",
    "gpt_eligible_reason",
    "gpt_called",
    "gpt_phase",
    "gpt_decision",
    "gpt_selected_intent",
    "gpt_selected_control_intent",
    "gpt_confidence",
    "gpt_timeout",
    "gpt_total_ms",
    "gpt_applied",
    "gpt_apply_reason",
    "gpt_fallback_type",
    "final_intent",
    "final_source",
    "repair_type",
    "intent_changed",
    "slots_changed",
    "training_candidate",
    "candidate_reasons",
]

LATENCY_COLUMNS: list[str] = [
    "timestamp_utc",
    "session_id",
    "turn_index",
    "state_before",
    "asr_normalized_text",
    "response_key",
    "preprocess_ms",
    "nlu_ms",
    "flow_ms",
    "route_ms",
    "handler_ms",
    "total_ms",
    "gpt_total_ms",
    "add_item_total_ms",
    "add_item_validator_ms",
]

TRAINING_COLUMNS: list[str] = [
    "timestamp_utc",
    "session_id",
    "turn_index",
    "state_before",
    "state_after",
    "asr_normalized_text",
    "local_intent",
    "local_confidence",
    "final_intent",
    "final_source",
    "repair_type",
    "candidate_reasons",
    "label_status",
    "needs_human_review",
    "response_key",
    "reprompt_count",
]


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _s(value: Any, default: str = "") -> str:
    """Convert value to CSV-safe string."""
    if value is None:
        return default
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_nlu_row(rec: dict[str, Any]) -> dict[str, str]:
    ids = rec.get("ids", {})
    turn = rec.get("turn", {})
    asr = rec.get("asr", {})
    nlu = rec.get("local_nlu", {})
    resp = rec.get("response", {})
    return {
        "timestamp_utc": _s(rec.get("timestamp_utc")),
        "session_id": _s(ids.get("session_id")),
        "turn_index": _s(turn.get("turn_index")),
        "state_before": _s(turn.get("state_before")),
        "state_after": _s(turn.get("state_after")),
        "asr_normalized_text": _s(asr.get("normalized_text")),
        "intent_main": _s(nlu.get("intent_main")),
        "intent_sub_intent": _s(nlu.get("intent_sub_intent")),
        "intent_effective": _s(nlu.get("intent_effective")),
        "confidence": _s(nlu.get("confidence")),
        "slot_model_ran": _s(nlu.get("slot_model_ran")),
        "slots_json": json.dumps(nlu.get("slots", []), ensure_ascii=False),
        "route_allowed": _s(nlu.get("route_allowed")),
        "route_reject_reason": _s(nlu.get("route_reject_reason")),
        "response_key": _s(resp.get("response_key")),
        "fallback_triggered": _s(turn.get("fallback_triggered")),
        "reprompt_count": _s(turn.get("reprompt_count")),
        "user_repeated": _s(turn.get("user_repeated")),
    }


def build_gpt_repair_row(rec: dict[str, Any]) -> dict[str, str]:
    ids = rec.get("ids", {})
    turn = rec.get("turn", {})
    asr = rec.get("asr", {})
    nlu = rec.get("local_nlu", {})
    gpt = rec.get("gpt_repair", {})
    fd = rec.get("final_decision", {})
    tr = rec.get("training", {})
    return {
        "timestamp_utc": _s(rec.get("timestamp_utc")),
        "session_id": _s(ids.get("session_id")),
        "turn_index": _s(turn.get("turn_index")),
        "state_before": _s(turn.get("state_before")),
        "asr_normalized_text": _s(asr.get("normalized_text")),
        "local_intent": _s(nlu.get("intent_effective")),
        "local_confidence": _s(nlu.get("confidence")),
        "gpt_eligible": _s(gpt.get("eligible")),
        "gpt_eligible_reason": _s(gpt.get("eligible_reason")),
        "gpt_called": _s(gpt.get("called")),
        "gpt_phase": _s(gpt.get("phase")),
        "gpt_decision": _s(gpt.get("decision")),
        "gpt_selected_intent": _s(gpt.get("selected_intent")),
        "gpt_selected_control_intent": _s(gpt.get("selected_control_intent")),
        "gpt_confidence": _s(gpt.get("confidence")),
        "gpt_timeout": _s(gpt.get("timeout")),
        "gpt_total_ms": _s(gpt.get("total_ms")),
        "gpt_applied": _s(gpt.get("applied")),
        "gpt_apply_reason": _s(gpt.get("apply_reason")),
        "gpt_fallback_type": _s(gpt.get("fallback_type")),
        "final_intent": _s(fd.get("final_intent")),
        "final_source": _s(fd.get("final_source")),
        "repair_type": _s(fd.get("repair_type")),
        "intent_changed": _s(fd.get("intent_changed")),
        "slots_changed": _s(fd.get("slots_changed")),
        "training_candidate": _s(tr.get("candidate")),
        "candidate_reasons": json.dumps(tr.get("candidate_reason", []), ensure_ascii=False),
    }


def build_latency_row(rec: dict[str, Any]) -> dict[str, str]:
    ids = rec.get("ids", {})
    turn = rec.get("turn", {})
    asr = rec.get("asr", {})
    resp = rec.get("response", {})
    lat = rec.get("latency", {})
    return {
        "timestamp_utc": _s(rec.get("timestamp_utc")),
        "session_id": _s(ids.get("session_id")),
        "turn_index": _s(turn.get("turn_index")),
        "state_before": _s(turn.get("state_before")),
        "asr_normalized_text": _s(asr.get("normalized_text")),
        "response_key": _s(resp.get("response_key")),
        "preprocess_ms": _s(lat.get("preprocess_ms")),
        "nlu_ms": _s(lat.get("nlu_ms")),
        "flow_ms": _s(lat.get("flow_ms")),
        "route_ms": _s(lat.get("route_ms")),
        "handler_ms": _s(lat.get("handler_ms")),
        "total_ms": _s(lat.get("total_ms")),
        "gpt_total_ms": _s(lat.get("gpt_total_ms")),
        "add_item_total_ms": _s(lat.get("add_item_total_ms")),
        "add_item_validator_ms": _s(lat.get("add_item_validator_ms")),
    }


def build_training_row(rec: dict[str, Any]) -> dict[str, str]:
    ids = rec.get("ids", {})
    turn = rec.get("turn", {})
    asr = rec.get("asr", {})
    nlu = rec.get("local_nlu", {})
    fd = rec.get("final_decision", {})
    tr = rec.get("training", {})
    resp = rec.get("response", {})
    return {
        "timestamp_utc": _s(rec.get("timestamp_utc")),
        "session_id": _s(ids.get("session_id")),
        "turn_index": _s(turn.get("turn_index")),
        "state_before": _s(turn.get("state_before")),
        "state_after": _s(turn.get("state_after")),
        "asr_normalized_text": _s(asr.get("normalized_text")),
        "local_intent": _s(nlu.get("intent_effective")),
        "local_confidence": _s(nlu.get("confidence")),
        "final_intent": _s(fd.get("final_intent")),
        "final_source": _s(fd.get("final_source")),
        "repair_type": _s(fd.get("repair_type")),
        "candidate_reasons": json.dumps(tr.get("candidate_reason", []), ensure_ascii=False),
        "label_status": _s(tr.get("label_status")),
        "needs_human_review": _s(tr.get("needs_human_review")),
        "response_key": _s(resp.get("response_key")),
        "reprompt_count": _s(turn.get("reprompt_count")),
    }


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def _iter_records(
    path: Path,
    *,
    limit: int | None = None,
    since: str | None = None,
    verbose: bool = False,
) -> Iterator[dict[str, Any]]:
    count = 0
    skipped = 0
    errors = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors += 1
                if verbose:
                    print(f"[WARN] line {lineno}: JSON parse error: {exc}", file=sys.stderr)
                continue

            if since and rec.get("timestamp_utc", "") < since:
                skipped += 1
                continue

            yield rec
            count += 1
            if limit and count >= limit:
                break

    if verbose:
        print(
            f"[INFO] processed={count} skipped={skipped} errors={errors}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def export(
    input_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    since: str | None = None,
    verbose: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    nlu_path = output_dir / "nlu_log.csv"
    gpt_path = output_dir / "gpt_repair_turns.csv"
    lat_path = output_dir / "realtime_turn_latency.csv"
    train_path = output_dir / "training_candidates.csv"

    with (
        nlu_path.open("w", newline="", encoding="utf-8") as nlu_fh,
        gpt_path.open("w", newline="", encoding="utf-8") as gpt_fh,
        lat_path.open("w", newline="", encoding="utf-8") as lat_fh,
        train_path.open("w", newline="", encoding="utf-8") as train_fh,
    ):
        nlu_w = csv.DictWriter(nlu_fh, fieldnames=NLU_COLUMNS, extrasaction="ignore")
        gpt_w = csv.DictWriter(gpt_fh, fieldnames=GPT_REPAIR_COLUMNS, extrasaction="ignore")
        lat_w = csv.DictWriter(lat_fh, fieldnames=LATENCY_COLUMNS, extrasaction="ignore")
        train_w = csv.DictWriter(train_fh, fieldnames=TRAINING_COLUMNS, extrasaction="ignore")

        nlu_w.writeheader()
        gpt_w.writeheader()
        lat_w.writeheader()
        train_w.writeheader()

        for rec in _iter_records(input_path, limit=limit, since=since, verbose=verbose):
            nlu_w.writerow(build_nlu_row(rec))
            gpt_w.writerow(build_gpt_repair_row(rec))
            lat_w.writerow(build_latency_row(rec))

            tr = rec.get("training", {})
            if tr.get("candidate"):
                train_w.writerow(build_training_row(rec))

    if verbose:
        print(f"[INFO] wrote: {nlu_path}", file=sys.stderr)
        print(f"[INFO] wrote: {gpt_path}", file=sys.stderr)
        print(f"[INFO] wrote: {lat_path}", file=sys.stderr)
        print(f"[INFO] wrote: {train_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export canonical turn_events.jsonl to derived CSV files.",
    )
    parser.add_argument(
        "--input",
        default="logs/current/turn_events.jsonl",
        help="Path to turn_events.jsonl (default: logs/current/turn_events.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/derived",
        help="Directory for derived CSVs (default: logs/derived)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records to process (omit for all)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="ISO-8601 UTC timestamp; skip records before this time",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] input file not found: {input_path}", file=sys.stderr)
        return 1

    export(
        input_path=input_path,
        output_dir=Path(args.output_dir),
        limit=args.limit,
        since=args.since,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
