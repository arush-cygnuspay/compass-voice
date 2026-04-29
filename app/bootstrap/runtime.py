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
from app.services.sms_service import SmsService
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
    sms_service: SmsService


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


def _validate_model_paths(
    intent_model_dir: Path,
    intent_labels_main_path: Path,
    intent_labels_sub_path: Path,
    slot_model_dir: Path,
) -> None:
    """Fail fast with actionable messages if any model path is missing."""
    checks: list[tuple[Path, str, list[str]]] = [
        (intent_model_dir, "COMPASS_INTENT_MODEL_DIR", ["config.json", "model.safetensors"]),
        (intent_labels_main_path, "COMPASS_INTENT_LABELS_MAIN", []),
        (intent_labels_sub_path, "COMPASS_INTENT_LABELS_SUB", []),
        (slot_model_dir, "COMPASS_SLOT_MODEL_DIR", ["config.cfg", "meta.json"]),
    ]
    errors: list[str] = []
    for path, env_var, required_children in checks:
        if not path.exists():
            errors.append(
                f"  ✗ {env_var}\n"
                f"      resolved : {path}\n"
                f"      fix      : set env var or place model at that path"
            )
            continue
        for child in required_children:
            if not (path / child).exists():
                errors.append(
                    f"  ✗ {env_var} — missing required file: {child}\n"
                    f"      resolved dir : {path}"
                )

    if errors:
        raise RuntimeError("\n".join(["[runtime] Model path validation failed:"] + errors))

    print(f"[runtime] intent_model_dir={intent_model_dir}")
    print(f"[runtime] slot_model_dir={slot_model_dir}")


def build_runtime(restaurant_id: str = "demo") -> AppRuntime:
    data_root = _restaurant_data_root(restaurant_id)

    menu_path = data_root / "menu.json"
    entity_index_path = data_root / "entity_index.json"

    print(f"[runtime] project_root={_project_root()}")
    print(f"[runtime] data_root={data_root}")
    print(f"[runtime] menu_exists={menu_path.exists()}")
    print(f"[runtime] entity_exists={entity_index_path.exists()}")

    # -----------------------------------------------------------------------
    # Model paths — canonical location: app/artifacts/models/
    # Relative paths are resolved from _project_root() so they work both
    # in local development and inside Docker (WORKDIR /app, COPY . .).
    #
    # Override any path via env var (absolute or project-root-relative):
    #   COMPASS_INTENT_MODEL_DIR   — intent model bundle directory
    #   COMPASS_INTENT_LABELS_MAIN — path to labels_main.json
    #   COMPASS_INTENT_LABELS_SUB  — path to labels_sub.json
    #   COMPASS_SLOT_MODEL_DIR     — spaCy slot model directory
    # -----------------------------------------------------------------------
    _INTENT_BASE = "app/artifacts/models/intent/distilbert-multihead-intent"

    intent_model_dir = _get_env_path(
        "COMPASS_INTENT_MODEL_DIR",
        _INTENT_BASE,
    )
    intent_labels_main_path = _get_env_path(
        "COMPASS_INTENT_LABELS_MAIN",
        f"{_INTENT_BASE}/labels_main.json",
    )
    intent_labels_sub_path = _get_env_path(
        "COMPASS_INTENT_LABELS_SUB",
        f"{_INTENT_BASE}/labels_sub.json",
    )
    intent_device = os.getenv("COMPASS_INTENT_DEVICE", "auto").strip()

    slot_model_dir = _get_env_path(
        "COMPASS_SLOT_MODEL_DIR",
        "app/artifacts/models/slot/model-best",
    )

    _validate_model_paths(
        intent_model_dir=intent_model_dir,
        intent_labels_main_path=intent_labels_main_path,
        intent_labels_sub_path=intent_labels_sub_path,
        slot_model_dir=slot_model_dir,
    )

    try:
        menu_store = MenuStore(menu_path, entity_index_path)
    except MenuLoadError as exc:
        raise RuntimeError(f"Failed to load menu: {exc}") from exc

    menu_repo = MenuRepository(menu_store)
    router = StateRouter()
    responder = ResponseBuilder(menu_repo)
    sms_service = SmsService()

    intent_bundle = load_intent_bundle(
        model_dir=str(intent_model_dir),
        labels_main_path=str(intent_labels_main_path),
        labels_sub_path=str(intent_labels_sub_path),
        device=intent_device,
    )
    slot_bundle = load_slot_bundle(str(slot_model_dir))

    engine = TurnEngine(
        router=router,
        menu_repo=menu_repo,
        intent_bundle=intent_bundle,
        slot_bundle=slot_bundle,
        responder=responder,
        sms_service=sms_service,
        nlu_logger=NluCsvLogger(),
    )

    return AppRuntime(
        menu_store=menu_store,
        menu_repo=menu_repo,
        router=router,
        intent_bundle=intent_bundle,
        slot_bundle=slot_bundle,
        engine=engine,
        responder=responder,
        sms_service=sms_service,
    )
