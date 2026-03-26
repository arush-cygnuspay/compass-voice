# app/nlu/slot_resolution/slot_resolver.py
from __future__ import annotations

from typing import Any, Dict, List, Union

from app.ml.slot.inference_slot import SlotBundle, predict_slots as run_slot_inference


def predict_slots(
    texts: Union[str, List[str]],
    bundle: SlotBundle,
) -> List[Dict[str, Any]]:
    """
    Run slot model for one or more texts using an already-loaded bundle.
    """
    return run_slot_inference(
        texts=texts,
        bundle=bundle,
        batch_size=32,
    )