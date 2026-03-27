# app/bootstrap/runtime.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.core.response_builder import ResponseBuilder
from app.core.turn_engine import TurnEngine
from app.logging.nlu_csv_logger import NluCsvLogger
from app.menu.exceptions import MenuLoadError
from app.menu.repository import MenuRepository
from app.menu.store import MenuStore
from app.ml.intent.inference_intent import IntentBundle, load_intent_bundle
from app.ml.slot.inference_slot import SlotBundle, load_slot_bundle
from app.state_machine.state_router import StateRouter


@dataclass(frozen=True, slots=True)
class AppRuntime:
    menu_store: MenuStore
    menu_repo: MenuRepository
    router: StateRouter
    intent_bundle: IntentBundle
    slot_bundle: SlotBundle
    engine: TurnEngine
    responder: ResponseBuilder


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _restaurant_data_root(restaurant_id: str) -> Path:
    return _project_root() / "app" / "data" / "restaurants" / restaurant_id


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _project_root() / path


def _get_env_path(name: str, default_relative_path: str) -> Path:
    value = os.getenv(name, default_relative_path).strip()
    return _resolve_project_path(value)


def build_runtime(restaurant_id: str = "demo") -> AppRuntime:
    data_root = _restaurant_data_root(restaurant_id)

    menu_path = data_root / "menu.json"
    entity_index_path = data_root / "entity_index.json"

    print(f"[runtime] project_root={_project_root()}")
    print(f"[runtime] data_root={data_root}")
    print(f"[runtime] menu_path={menu_path}")
    print(f"[runtime] entity_index_path={entity_index_path}")
    print(f"[runtime] menu_exists={menu_path.exists()}")
    print(f"[runtime] entity_exists={entity_index_path.exists()}")