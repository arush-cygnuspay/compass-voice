# app/ml/intent/inference_intent.py

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .modeling_multihead import MultiHeadIntentModel


def _ensure_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_device(device: str | None = None) -> torch.device:
    """
    Supported values:
    - None / 'auto' => cuda if available, else cpu
    - 'cpu'
    - 'cuda'
    """
    requested = (device or os.getenv("COMPASS_INTENT_DEVICE", "auto")).strip().lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "COMPASS_INTENT_DEVICE='cuda' but CUDA is not available. "
                "Install a CUDA-enabled PyTorch build and verify GPU visibility."
            )
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device '{requested}'. Expected one of: auto, cpu, cuda"
    )


@dataclass(frozen=True, slots=True)
class IntentBundle:
    tokenizer: PreTrainedTokenizerBase
    model: MultiHeadIntentModel
    id2main: Dict[int, str]
    id2sub: Dict[int, str]
    device: torch.device
    use_amp: bool


_BUNDLES: dict[str, IntentBundle] = {}
_BUNDLE_LOCK = threading.Lock()


def load_intent_bundle(
    model_dir: str,
    labels_main_path: str,
    labels_sub_path: str,
    device: str | None = None,
) -> IntentBundle:
    _ensure_exists(model_dir, "INTENT_MODEL_DIR")
    _ensure_exists(os.path.join(model_dir, "config.json"), "intent config.json")
    _ensure_exists(labels_main_path, "labels_main.json")
    _ensure_exists(labels_sub_path, "labels_sub.json")

    resolved_device = _resolve_device(device)
    use_amp = _env_flag("COMPASS_INTENT_USE_AMP", default=True) and resolved_device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = MultiHeadIntentModel.from_pretrained(model_dir)
    model.to(resolved_device)
    model.eval()

    id2main_raw = _read_json(labels_main_path)
    id2sub_raw = _read_json(labels_sub_path)

    id2main = {int(k): v for k, v in id2main_raw.items()}
    id2sub = {int(k): v for k, v in id2sub_raw.items()}

    if _env_flag("COMPASS_INTENT_DEBUG", default=False):
        debug_payload = {
            "requested_device": device or os.getenv("COMPASS_INTENT_DEVICE", "auto"),
            "resolved_device": str(resolved_device),
            "cuda_available": torch.cuda.is_available(),
            "use_amp": use_amp,
        }
        if resolved_device.type == "cuda":
            gpu_index = torch.cuda.current_device()
            debug_payload.update(
                {
                    "gpu_name": torch.cuda.get_device_name(gpu_index),
                    "gpu_capability": torch.cuda.get_device_capability(gpu_index),
                    "gpu_total_mem_gb": round(
                        torch.cuda.get_device_properties(gpu_index).total_memory / (1024 ** 3),
                        2,
                    ),
                }
            )
        print(f"[IntentBundle] {debug_payload}")

    return IntentBundle(
        tokenizer=tokenizer,
        model=model,
        id2main=id2main,
        id2sub=id2sub,
        device=resolved_device,
        use_amp=use_amp,
    )


def get_intent_bundle(
    model_dir: str,
    labels_main_path: str,
    labels_sub_path: str,
    device: str | None = None,
) -> IntentBundle:
    resolved_device = str(_resolve_device(device))

    cached = _BUNDLES.get(resolved_device)
    if cached is not None:
        return cached

    with _BUNDLE_LOCK:
        cached = _BUNDLES.get(resolved_device)
        if cached is None:
            cached = load_intent_bundle(
                model_dir=model_dir,
                labels_main_path=labels_main_path,
                labels_sub_path=labels_sub_path,
                device=device,
            )
            _BUNDLES[resolved_device] = cached
        return cached


def clear_intent_bundle_cache() -> None:
    with _BUNDLE_LOCK:
        _BUNDLES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def predict_intent(
    texts: str | Sequence[str],
    bundle: IntentBundle,
    max_length: int = 64,
) -> List[dict]:
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)

    if not texts:
        return []

    enc = bundle.tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(bundle.device, non_blocking=(bundle.device.type == "cuda")) for k, v in enc.items()}

    if bundle.device.type == "cuda" and bundle.use_amp:
        with torch.amp.autocast(device_type="cuda"):
            outputs = bundle.model(**enc)
    else:
        outputs = bundle.model(**enc)

    logits_main = outputs["logits_main"]
    logits_sub = outputs["logits_sub"]

    probs_main = F.softmax(logits_main, dim=-1)
    probs_sub = F.softmax(logits_sub, dim=-1)

    conf_main, pred_main_ids = probs_main.max(dim=-1)
    conf_sub, pred_sub_ids = probs_sub.max(dim=-1)

    pred_main_ids = pred_main_ids.tolist()
    pred_sub_ids = pred_sub_ids.tolist()
    conf_main = conf_main.tolist()
    conf_sub = conf_sub.tolist()

    results: List[dict] = []
    for i, text in enumerate(texts):
        main_id = int(pred_main_ids[i])
        sub_id = int(pred_sub_ids[i])

        results.append(
            {
                "text": text,
                "pred_main_intent": bundle.id2main.get(main_id, f"UNKNOWN_MAIN_{main_id}"),
                "pred_sub_intent": bundle.id2sub.get(sub_id, f"UNKNOWN_SUB_{sub_id}"),
                "confidence_main": float(conf_main[i]),
                "confidence_sub": float(conf_sub[i]),
                "device": str(bundle.device),
                "amp": bundle.use_amp,
            }
        )

    return results


def benchmark_predict_intent(
    texts: Sequence[str],
    bundle: IntentBundle,
    *,
    max_length: int = 64,
    warmup_runs: int = 20,
    measured_runs: int = 100,
) -> dict:
    """
    Returns stable latency metrics for CPU/GPU comparison.
    """
    if not texts:
        raise ValueError("texts must not be empty")

    texts = list(texts)

    # Warmup
    for _ in range(warmup_runs):
        predict_intent(texts, bundle, max_length=max_length)
    _synchronize_if_cuda(bundle.device)

    # Timed loop
    latencies_ms: List[float] = []
    for _ in range(measured_runs):
        start = time.perf_counter()
        predict_intent(texts, bundle, max_length=max_length)
        _synchronize_if_cuda(bundle.device)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

    latencies_ms.sort()
    total_ms = sum(latencies_ms)
    batch_size = len(texts)

    def _pct(p: float) -> float:
        idx = min(len(latencies_ms) - 1, max(0, int(round((p / 100.0) * (len(latencies_ms) - 1)))))
        return latencies_ms[idx]

    return {
        "device": str(bundle.device),
        "amp": bundle.use_amp,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "avg_latency_ms": round(total_ms / measured_runs, 3),
        "p50_latency_ms": round(_pct(50), 3),
        "p95_latency_ms": round(_pct(95), 3),
        "min_latency_ms": round(latencies_ms[0], 3),
        "max_latency_ms": round(latencies_ms[-1], 3),
        "throughput_texts_per_sec": round((batch_size * measured_runs) / (total_ms / 1000.0), 3),
    }