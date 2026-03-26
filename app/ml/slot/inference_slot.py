# app/ml/slot/inference_slot.py

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Sequence

import spacy
from spacy.language import Language


def _ensure_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


@dataclass(frozen=True, slots=True)
class SlotBundle:
    nlp: Language


_SLOT_BUNDLE: SlotBundle | None = None
_SLOT_LOCK = threading.Lock()


def load_slot_bundle(spacy_model_dir: str) -> SlotBundle:
    _ensure_exists(spacy_model_dir, "SPACY_SLOT_MODEL_DIR")
    nlp = spacy.load(spacy_model_dir)
    return SlotBundle(nlp=nlp)


def get_slot_bundle(spacy_model_dir: str) -> SlotBundle:
    global _SLOT_BUNDLE

    if _SLOT_BUNDLE is not None:
        return _SLOT_BUNDLE

    with _SLOT_LOCK:
        if _SLOT_BUNDLE is None:
            _SLOT_BUNDLE = load_slot_bundle(spacy_model_dir)

    return _SLOT_BUNDLE


def predict_slots(
    texts: str | Sequence[str],
    bundle: SlotBundle,
    batch_size: int = 32,
) -> List[dict]:
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)

    if not texts:
        return []

    results: List[dict] = []
    docs = bundle.nlp.pipe(texts, batch_size=batch_size)

    for text, doc in zip(texts, docs):
        slots = [
            {
                "slot": ent.label_,
                "value": ent.text,
                "start": int(ent.start_char),
                "end": int(ent.end_char),
            }
            for ent in doc.ents
        ]
        results.append({"text": text, "slots": slots})

    return results