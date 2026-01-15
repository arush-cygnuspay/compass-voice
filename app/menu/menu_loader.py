# app/menu/menu_loader.py

import json
from typing import Dict
from app.menu.models import MenuItem


class MenuLoader:
    """
    Loads menu data from a static source.
    """

    @staticmethod
    def load_from_file(path: str) -> Dict[str, MenuItem]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Parsing logic will be implemented incrementally
        return {}
