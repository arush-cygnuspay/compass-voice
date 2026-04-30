# app/scripts/benchmark_nlu_inference.py

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import spacy
import torch

from app.ml.intent.inference_intent import (
    benchmark_predict_intent,
    clear_intent_bundle_cache,
    get_intent_bundle,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INTENT_DIR = _PROJECT_ROOT / "app" / "artifacts" / "models" / "intent" / "distilbert-multihead-intent"

MODEL_DIR = os.getenv(
    "COMPASS_INTENT_MODEL_DIR",
    str(_INTENT_DIR),
)
LABELS_MAIN = os.getenv(
    "COMPASS_INTENT_LABELS_MAIN",
    str(_INTENT_DIR / "labels_main.json"),
)
LABELS_SUB = os.getenv(
    "COMPASS_INTENT_LABELS_SUB",
    str(_INTENT_DIR / "labels_sub.json"),
)

SPACY_MODEL_DIR = os.getenv(
    "COMPASS_SLOT_MODEL_DIR",
    str(_PROJECT_ROOT / "app" / "ml" / "models" / "spacy_slot_trf_out" / "model-best"),
)

REPORT_DIR = Path(
    os.getenv("COMPASS_BENCHMARK_REPORT_DIR", "artifacts/benchmarks")
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_RUNS = int(os.getenv("COMPASS_BENCHMARK_WARMUP_RUNS", "20"))
MEASURED_RUNS = int(os.getenv("COMPASS_BENCHMARK_MEASURED_RUNS", "100"))
MAX_LENGTH = int(os.getenv("COMPASS_INTENT_MAX_LENGTH", "64"))

# Benchmark these batch sizes. You can change later.
BATCH_SIZES = [1, 8, 32]

# Use a realistic small corpus and repeat from it.
BASE_TEXTS = [
    "i want a chicken burger",
    "show me the menu",
    "remove coke from my cart",
    "what is the price of zinger burger",
    "add one large fries",
    "checkout please",
    "show total",
    "i want two pepsis",
    "add one zinger burger and one coke",
    "remove fries from the cart",
    "what drinks do you have",
    "show me burgers",
    "i want a large pizza",
    "add two small coffees",
    "is chicken burger available",
    "show cart",
]


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    component: str
    device: str
    batch_size: int
    warmup_runs: int
    measured_runs: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_measured_seconds: float
    throughput_texts_per_sec: float
    amp_enabled: bool | None = None
    notes: str = ""


def _ensure_texts(batch_size: int) -> list[str]:
    if batch_size <= len(BASE_TEXTS):
        return BASE_TEXTS[:batch_size]

    repeated: list[str] = []
    while len(repeated) < batch_size:
        repeated.extend(BASE_TEXTS)
    return repeated[:batch_size]


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def _torch_sync_if_needed(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _configure_spacy_device(device: str) -> None:
    """
    Must be called before spacy.load(...)
    """
    if device == "cuda":
        ok = spacy.prefer_gpu()
        if not ok:
            raise RuntimeError(
                "spaCy GPU requested, but spaCy could not enable GPU. "
                "Check spacy-transformers / cupy / compatible CUDA install."
            )
    elif device == "cpu":
        spacy.require_cpu()
    else:
        raise ValueError(f"Unsupported spaCy device: {device}")


def _load_spacy_pipeline(device: str):
    _configure_spacy_device(device)
    return spacy.load(SPACY_MODEL_DIR)


def _benchmark_spacy_slot_pipeline(
    *,
    device: str,
    batch_size: int,
    warmup_runs: int,
    measured_runs: int,
) -> BenchmarkRow:
    texts = _ensure_texts(batch_size)
    nlp = _load_spacy_pipeline(device)

    # Warmup
    for _ in range(warmup_runs):
        list(nlp.pipe(texts, batch_size=batch_size))
    _torch_sync_if_needed(device)

    latencies_ms: list[float] = []

    for _ in range(measured_runs):
        start = time.perf_counter()
        list(nlp.pipe(texts, batch_size=batch_size))
        _torch_sync_if_needed(device)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

    latencies_ms.sort()
    total_ms = float(sum(latencies_ms))

    return BenchmarkRow(
        component="slot_spacy",
        device=device,
        batch_size=batch_size,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        avg_latency_ms=round(total_ms / measured_runs, 3),
        p50_latency_ms=round(_percentile(latencies_ms, 50), 3),
        p95_latency_ms=round(_percentile(latencies_ms, 95), 3),
        min_latency_ms=round(latencies_ms[0], 3),
        max_latency_ms=round(latencies_ms[-1], 3),
        total_measured_seconds=round(total_ms / 1000.0, 3),
        throughput_texts_per_sec=round((batch_size * measured_runs) / (total_ms / 1000.0), 3),
        amp_enabled=None,
        notes=f"spaCy pipeline path={SPACY_MODEL_DIR}",
    )


def _benchmark_intent_model(
    *,
    device: str,
    batch_size: int,
    warmup_runs: int,
    measured_runs: int,
) -> BenchmarkRow:
    texts = _ensure_texts(batch_size)

    clear_intent_bundle_cache()
    bundle = get_intent_bundle(
        model_dir=MODEL_DIR,
        labels_main_path=LABELS_MAIN,
        labels_sub_path=LABELS_SUB,
        device=device,
    )

    result = benchmark_predict_intent(
        texts=texts,
        bundle=bundle,
        max_length=MAX_LENGTH,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )

    return BenchmarkRow(
        component="intent_pytorch",
        device=device,
        batch_size=batch_size,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        avg_latency_ms=float(result["avg_latency_ms"]),
        p50_latency_ms=float(result["p50_latency_ms"]),
        p95_latency_ms=float(result["p95_latency_ms"]),
        min_latency_ms=float(result["min_latency_ms"]),
        max_latency_ms=float(result["max_latency_ms"]),
        total_measured_seconds=round(
            (float(result["avg_latency_ms"]) * measured_runs) / 1000.0,
            3,
        ),
        throughput_texts_per_sec=float(result["throughput_texts_per_sec"]),
        amp_enabled=bool(result.get("amp", False)),
        notes=f"intent model path={MODEL_DIR}",
    )


def _safe_run(component: str, device: str, batch_size: int) -> dict[str, Any]:
    try:
        if component == "intent_pytorch":
            row = _benchmark_intent_model(
                device=device,
                batch_size=batch_size,
                warmup_runs=WARMUP_RUNS,
                measured_runs=MEASURED_RUNS,
            )
        elif component == "slot_spacy":
            row = _benchmark_spacy_slot_pipeline(
                device=device,
                batch_size=batch_size,
                warmup_runs=WARMUP_RUNS,
                measured_runs=MEASURED_RUNS,
            )
        else:
            raise ValueError(f"Unknown component: {component}")

        return {
            "ok": True,
            "row": row,
            "error": "",
        }

    except Exception as exc:
        return {
            "ok": False,
            "row": BenchmarkRow(
                component=component,
                device=device,
                batch_size=batch_size,
                warmup_runs=WARMUP_RUNS,
                measured_runs=MEASURED_RUNS,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                total_measured_seconds=0.0,
                throughput_texts_per_sec=0.0,
                amp_enabled=None,
                notes=f"FAILED: {exc}",
            ),
            "error": str(exc),
        }


def _write_csv(rows: Sequence[BenchmarkRow], output_path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "component",
        "device",
        "batch_size",
        "warmup_runs",
        "measured_runs",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "total_measured_seconds",
        "throughput_texts_per_sec",
        "amp_enabled",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _build_summary(rows: Sequence[BenchmarkRow]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[int, BenchmarkRow]]] = {}

    for row in rows:
        grouped.setdefault(row.component, {}).setdefault(row.device, {})[row.batch_size] = row

    summary: dict[str, Any] = {
        "generated_at_epoch_seconds": round(time.time(), 3),
        "report_units": {
            "latency": "milliseconds",
            "total_time": "seconds",
            "throughput": "texts/second",
        },
        "system": {
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "spacy_model_dir": SPACY_MODEL_DIR,
            "intent_model_dir": MODEL_DIR,
        },
        "results": {},
    }

    for component, by_device in grouped.items():
        summary["results"][component] = {}

        cpu_rows = by_device.get("cpu", {})
        gpu_rows = by_device.get("cuda", {})

        for batch_size in sorted(set(cpu_rows.keys()) | set(gpu_rows.keys())):
            cpu_row = cpu_rows.get(batch_size)
            gpu_row = gpu_rows.get(batch_size)

            batch_summary: dict[str, Any] = {}

            if cpu_row is not None:
                batch_summary["cpu"] = asdict(cpu_row)
            if gpu_row is not None:
                batch_summary["gpu"] = asdict(gpu_row)

            if cpu_row is not None and gpu_row is not None:
                if gpu_row.avg_latency_ms > 0:
                    batch_summary["speedup_avg_latency_x"] = round(
                        cpu_row.avg_latency_ms / gpu_row.avg_latency_ms,
                        3,
                    )
                if gpu_row.p50_latency_ms > 0:
                    batch_summary["speedup_p50_x"] = round(
                        cpu_row.p50_latency_ms / gpu_row.p50_latency_ms,
                        3,
                    )
                if cpu_row.throughput_texts_per_sec > 0:
                    batch_summary["throughput_gain_x"] = round(
                        gpu_row.throughput_texts_per_sec / cpu_row.throughput_texts_per_sec,
                        3,
                    )

            summary["results"][component][f"batch_{batch_size}"] = batch_summary

    return summary


def main() -> None:
    rows: list[BenchmarkRow] = []

    print("=" * 80)
    print("Compass NLU Benchmark")
    print("=" * 80)
    print(f"Intent model: {MODEL_DIR}")
    print(f"Slot model:   {SPACY_MODEL_DIR}")
    print(f"Warmup runs:  {WARMUP_RUNS}")
    print(f"Measured:     {MEASURED_RUNS}")
    print(f"Batch sizes:  {BATCH_SIZES}")
    print()

    test_matrix = [
        ("intent_pytorch", "cpu"),
        ("intent_pytorch", "cuda"),
        ("slot_spacy", "cpu"),
        ("slot_spacy", "cuda"),
    ]

    for component, device in test_matrix:
        for batch_size in BATCH_SIZES:
            print(f"[RUN] component={component} device={device} batch_size={batch_size}")
            result = _safe_run(component, device, batch_size)
            row = result["row"]
            rows.append(row)

            if result["ok"]:
                print(
                    f"  avg={row.avg_latency_ms} ms | "
                    f"p50={row.p50_latency_ms} ms | "
                    f"p95={row.p95_latency_ms} ms | "
                    f"throughput={row.throughput_texts_per_sec} texts/sec"
                )
            else:
                print(f"  FAILED: {result['error']}")

    csv_path = REPORT_DIR / "benchmark_report.csv"
    json_path = REPORT_DIR / "benchmark_summary.json"

    _write_csv(rows, csv_path)

    summary = _build_summary(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print("Saved reports")
    print("=" * 80)
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()


# python -c "import spacy; print(spacy.prefer_gpu()); nlp = spacy.load('app/artifacts/models/slot/model-best'); print('loaded')"