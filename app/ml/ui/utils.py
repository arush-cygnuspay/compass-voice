from __future__ import annotations
import json
import os
from typing import Dict


def read_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)