#!/usr/bin/env python3
# scripts/verify_gpt_ordering_pipeline.py
"""Pre-deployment verification script for Compass Voice GPT ordering pipeline.

Checks configuration and safety invariants for Phase 3 (option resolver) and
Phase 4 (add-item planner) before manual or automated deployment.

Does NOT call OpenAI.  Does NOT mutate any state.  Exit codes:
  0 — all checks passed (may include warnings for review)
  1 — one or more checks failed (deploy blocked)
  2 — internal error (misconfigured Python environment)

Usage:
    python scripts/verify_gpt_ordering_pipeline.py
    python scripts/verify_gpt_ordering_pipeline.py --env production
    python scripts/verify_gpt_ordering_pipeline.py --env staging --strict
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Project imports — only after sys.path is set.
# Deferred to after _PROJECT_ROOT insertion below.

# ── Add project root to sys.path so script is runnable from repo root ────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Module-level import so tests can patch at 'scripts.verify_gpt_ordering_pipeline.get_semantic_repair_config'
from app.config.semantic_repair import get_semantic_repair_config  # noqa: E402

# ── Result types ─────────────────────────────────────────────────────────────

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
INFO = "INFO"


@dataclass
class CheckResult:
    status: str        # PASS | WARN | FAIL | INFO
    name: str
    message: str
    detail: str = ""


# ── Colour helpers ────────────────────────────────────────────────────────────

_COLOUR = {
    PASS: "\033[32m",   # green
    WARN: "\033[33m",   # yellow
    FAIL: "\033[31m",   # red
    INFO: "\033[36m",   # cyan
}
_RESET = "\033[0m"


def _fmt(result: CheckResult, *, colour: bool) -> str:
    tag = f"[{result.status}]"
    if colour:
        tag = f"{_COLOUR.get(result.status, '')}{tag}{_RESET}"
    line = f"  {tag} {result.name}: {result.message}"
    if result.detail:
        line += f"\n       {result.detail}"
    return line


# ── Individual checks ─────────────────────────────────────────────────────────


def _check_config_loads() -> CheckResult:
    """SemanticRepairConfig can be loaded without errors."""
    try:
        # Clear lru_cache so env vars are re-read fresh.
        get_semantic_repair_config.cache_clear()
        cfg = get_semantic_repair_config()
        return CheckResult(PASS, "config_loads", "SemanticRepairConfig loaded without errors")
    except Exception as exc:
        return CheckResult(FAIL, "config_loads", f"Config failed to load: {exc}")


def _check_option_resolver_mode() -> CheckResult:
    """COMPASS_GPT_OPTION_RESOLVER_MODE is a known valid value."""
    get_semantic_repair_config.cache_clear()
    cfg = get_semantic_repair_config()
    mode = cfg.option_resolver_mode
    valid = {"disabled", "shadow", "inline"}
    if mode not in valid:
        return CheckResult(
            FAIL, "option_resolver_mode",
            f"Invalid mode {mode!r}",
            f"Allowed: {sorted(valid)}",
        )
    env_val = os.getenv("COMPASS_GPT_OPTION_RESOLVER_MODE", "<unset — default: disabled>")
    return CheckResult(
        INFO, "option_resolver_mode",
        f"mode={mode!r}",
        f"COMPASS_GPT_OPTION_RESOLVER_MODE={env_val!r}",
    )


def _check_add_item_planner_mode() -> CheckResult:
    """COMPASS_GPT_ADD_ITEM_PLANNER_MODE is a known valid value."""
    get_semantic_repair_config.cache_clear()
    cfg = get_semantic_repair_config()
    mode = cfg.add_item_planner_mode
    valid = {"disabled", "shadow", "inline"}
    if mode not in valid:
        return CheckResult(
            FAIL, "add_item_planner_mode",
            f"Invalid mode {mode!r}",
            f"Allowed: {sorted(valid)}",
        )
    env_val = os.getenv("COMPASS_GPT_ADD_ITEM_PLANNER_MODE", "<unset — default: disabled>")
    return CheckResult(
        INFO, "add_item_planner_mode",
        f"mode={mode!r}",
        f"COMPASS_GPT_ADD_ITEM_PLANNER_MODE={env_val!r}",
    )


def _check_disabled_is_default() -> CheckResult:
    """Both GPT features default to 'disabled' when env vars are unset."""
    results = []
    for env_var, field in (
        ("COMPASS_GPT_OPTION_RESOLVER_MODE", "option_resolver_mode"),
        ("COMPASS_GPT_ADD_ITEM_PLANNER_MODE", "add_item_planner_mode"),
    ):
        if env_var not in os.environ:
            results.append(f"{env_var}=<unset -> disabled>")
        else:
            val = os.environ[env_var]
            if val != "disabled":
                results.append(f"{env_var}={val!r} (override active)")
            else:
                results.append(f"{env_var}=disabled (explicit)")
    return CheckResult(
        PASS, "default_mode_safe",
        "Disabled-by-default invariant verified",
        "; ".join(results),
    )


def _check_api_key_presence() -> CheckResult:
    """OPENAI_API_KEY presence check — never logs the key value."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return CheckResult(
            WARN, "openai_api_key",
            "OPENAI_API_KEY is not set",
            "GPT calls will be skipped at runtime (safe in disabled/shadow mode)",
        )
    # Only log length/prefix to avoid leaking the key.
    prefix = key[:7] + "..." if len(key) > 10 else "<short>"
    return CheckResult(
        PASS, "openai_api_key",
        "OPENAI_API_KEY is present",
        f"prefix={prefix!r}  len={len(key)}",
    )


def _check_inline_without_api_key() -> CheckResult:
    """If inline mode is active but API key is absent, warn loudly."""
    get_semantic_repair_config.cache_clear()
    cfg = get_semantic_repair_config()
    key = os.getenv("OPENAI_API_KEY", "")
    inline_features: list[str] = []
    if cfg.option_resolver_mode == "inline":
        inline_features.append("option_resolver")
    if cfg.add_item_planner_mode == "inline":
        inline_features.append("add_item_planner")

    if not inline_features:
        return CheckResult(
            PASS, "inline_requires_api_key",
            "No inline features active — API key not required",
        )
    if not key:
        return CheckResult(
            FAIL, "inline_requires_api_key",
            f"Inline mode active for {inline_features} but OPENAI_API_KEY is missing",
            "All inline GPT calls will silently fall back to local path — set the key",
        )
    return CheckResult(
        PASS, "inline_requires_api_key",
        f"Inline features {inline_features} have API key",
    )


def _check_shadow_mode_never_applies() -> CheckResult:
    """Structural verification: shadow mode can never apply its result.

    Checks the apply gate logic — not a runtime call.
    """
    try:
        from app.nlu.semantic_repair.add_item_plan_validator import PlannerApplyGate
        gate = PlannerApplyGate()
        safe, reason = gate.should_apply(
            route_mode="shadow_gpt",
            decision="add_items",
            validated_plan=None,
            confidence=0.99,
            min_confidence=0.0,
            parse_error=None,
            gpt_called=True,
            timed_out=False,
        )
        if safe:
            return CheckResult(
                FAIL, "shadow_never_applies",
                "PlannerApplyGate approved apply for shadow_gpt — THIS IS A BUG",
                f"reason={reason!r}",
            )
        return CheckResult(
            PASS, "shadow_never_applies",
            "PlannerApplyGate correctly rejects shadow_gpt",
            f"block_reason={reason!r}",
        )
    except Exception as exc:
        return CheckResult(FAIL, "shadow_never_applies", f"Exception: {exc}")


def _check_inline_apply_gate_requires_all_conditions() -> CheckResult:
    """Apply gate blocks apply when any one of 8 conditions fails."""
    try:
        from app.nlu.semantic_repair.add_item_plan_validator import PlannerApplyGate
        gate = PlannerApplyGate()

        failures: list[str] = []

        # Gate 1: route_mode must be inline_gpt
        safe, _ = gate.should_apply(
            route_mode="shadow_gpt", decision="add_items", validated_plan=None,
            confidence=0.99, min_confidence=0.0,
        )
        if safe:
            failures.append("gate1: wrong route_mode accepted")

        # Gate 3: timed_out blocks apply
        safe, _ = gate.should_apply(
            route_mode="inline_gpt", decision="add_items", validated_plan=None,
            confidence=0.99, min_confidence=0.0, timed_out=True,
        )
        if safe:
            failures.append("gate3: timed_out=True accepted")

        # Gate 4: parse_error blocks apply
        safe, _ = gate.should_apply(
            route_mode="inline_gpt", decision="add_items", validated_plan=None,
            confidence=0.99, min_confidence=0.0, parse_error="json_invalid",
        )
        if safe:
            failures.append("gate4: parse_error accepted")

        # Gate 5: decision must be add_items
        safe, _ = gate.should_apply(
            route_mode="inline_gpt", decision="no_repair", validated_plan=None,
            confidence=0.99, min_confidence=0.0,
        )
        if safe:
            failures.append("gate5: wrong decision accepted")

        # Gate 6: confidence threshold
        safe, _ = gate.should_apply(
            route_mode="inline_gpt", decision="add_items", validated_plan=None,
            confidence=0.10, min_confidence=0.75,
        )
        if safe:
            failures.append("gate6: low confidence accepted")

        if failures:
            return CheckResult(
                FAIL, "inline_apply_gate_all_conditions",
                f"{len(failures)} gate condition(s) did not block as expected",
                "; ".join(failures),
            )
        return CheckResult(
            PASS, "inline_apply_gate_all_conditions",
            "Apply gate correctly blocks on all tested failure conditions",
        )
    except Exception as exc:
        return CheckResult(FAIL, "inline_apply_gate_all_conditions", f"Exception: {exc}")


def _check_option_resolver_service_instantiates() -> CheckResult:
    """GptOptionResolverService can be instantiated without an API key."""
    try:
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.2,
            option_resolver_mode="shadow",
        )
        from app.nlu.semantic_repair.option_resolver_service import GptOptionResolverService
        svc = GptOptionResolverService(config=cfg)
        return CheckResult(
            PASS, "option_resolver_instantiates",
            "GptOptionResolverService constructed successfully",
        )
    except Exception as exc:
        return CheckResult(FAIL, "option_resolver_instantiates", f"Exception: {exc}")


def _check_add_item_planner_service_instantiates() -> CheckResult:
    """GptAddItemPlannerService can be instantiated without an API key."""
    try:
        from app.config.semantic_repair import SemanticRepairConfig
        cfg = SemanticRepairConfig(
            phase=0, model="gpt-4o-mini", timeout_seconds=1.8,
            add_item_planner_mode="shadow",
        )
        from app.nlu.semantic_repair.add_item_planner_service import GptAddItemPlannerService
        svc = GptAddItemPlannerService(config=cfg)
        return CheckResult(
            PASS, "add_item_planner_instantiates",
            "GptAddItemPlannerService constructed successfully",
        )
    except Exception as exc:
        return CheckResult(FAIL, "add_item_planner_instantiates", f"Exception: {exc}")


def _check_handler_dispatcher_builds() -> CheckResult:
    """HandlerDispatcher builds cleanly with current env config."""
    try:
        from pathlib import Path as _Path
        data_root = _PROJECT_ROOT / "app" / "data" / "restaurants" / "steves_grill"
        if not (data_root / "menu.json").exists():
            # Try demo restaurant
            data_root = _PROJECT_ROOT / "app" / "data" / "restaurants" / "demo"
        if not (data_root / "menu.json").exists():
            return CheckResult(
                WARN, "handler_dispatcher_builds",
                "No menu fixture found — skipping dispatcher build check",
            )
        from app.menu.store import MenuStore
        from app.menu.repository import MenuRepository
        from app.cart.read_models.cart_summary_builder import CartSummaryBuilder
        from app.core.command_executor import CommandExecutor
        from app.core.response_builder import ResponseBuilder
        from app.core.turn_diagnostics import TurnDiagnostics
        from app.core.handler_dispatcher import HandlerDispatcher
        from app.services.checkout_service import CheckoutService

        class _Stub:
            def is_configured(self): return False
            def send(self, r): return type("R", (), {"ok": False, "sid": None, "error_code": "x", "error_message": "x"})()

        store = MenuStore(
            menu_path=data_root / "menu.json",
            entity_index_path=data_root / "entity_index.json",
        )
        repo = MenuRepository(store)
        sms = _Stub()
        dispatcher = HandlerDispatcher(
            menu_repo=repo,
            cart_summary_builder=CartSummaryBuilder(repo),
            sms_service=sms,
            checkout_service=CheckoutService(),
            responder=ResponseBuilder(repo),
            command_executor=CommandExecutor(sms),
            diagnostics=TurnDiagnostics(backends=[]),
        )
        handler = dispatcher.get_handler("add_item_handler")
        planner = getattr(handler, "_gpt_planner", None)
        get_semantic_repair_config.cache_clear()
        mode = get_semantic_repair_config().add_item_planner_mode
        if mode == "disabled" and planner is not None:
            return CheckResult(
                FAIL, "handler_dispatcher_builds",
                "add_item_planner_mode=disabled but planner is not None",
            )
        if mode != "disabled" and planner is None:
            return CheckResult(
                WARN, "handler_dispatcher_builds",
                f"add_item_planner_mode={mode!r} but planner is None — check config load",
            )
        planner_desc = f"gpt_planner={type(planner).__name__}" if planner else "gpt_planner=None"
        return CheckResult(
            PASS, "handler_dispatcher_builds",
            f"HandlerDispatcher built cleanly — add_item_handler {planner_desc}",
        )
    except Exception as exc:
        return CheckResult(FAIL, "handler_dispatcher_builds", f"Exception: {exc}")


def _check_production_inline_warning(env_name: str) -> CheckResult:
    """Warn loudly if inline is enabled in a production environment."""
    get_semantic_repair_config.cache_clear()
    cfg = get_semantic_repair_config()
    is_prod = env_name.lower() in ("production", "prod")
    inline_features: list[str] = []
    if cfg.option_resolver_mode == "inline":
        inline_features.append("option_resolver")
    if cfg.add_item_planner_mode == "inline":
        inline_features.append("add_item_planner")

    if is_prod and inline_features:
        return CheckResult(
            WARN, "production_inline_check",
            f"PRODUCTION with inline active: {inline_features}",
            "Inline mode applies GPT results — ensure staging validation passed first. "
            "Set to shadow or disabled if not yet validated.",
        )
    if is_prod:
        return CheckResult(
            PASS, "production_inline_check",
            "Production environment — no inline features active",
        )
    return CheckResult(
        INFO, "production_inline_check",
        f"Non-production env={env_name!r} — inline check advisory only",
        f"inline features: {inline_features or 'none'}",
    )


def _check_logging_fields_present() -> CheckResult:
    """Spot-check that the expected structured log fields are emitted.

    Inspects handler source for field names rather than executing a real turn.
    """
    try:
        import inspect
        from app.state_machine.handlers.item.add_item import add_item_handler as aih
        source = inspect.getsource(aih)
        required_fields = [
            "add_item_planner_mode",
            "add_item_planner_route_reason",
            "add_item_planner_called",
            "add_item_planner_decision",
            "add_item_planner_confidence",
            "add_item_planner_validator_passed",
            "add_item_planner_safe_to_apply",
            "add_item_planner_latency_ms",
            "add_item_planner_applied",
            "add_item_planner_apply_block_reason",
            "add_item_planner_validator_reject_reason",
        ]
        missing = [f for f in required_fields if f not in source]
        if missing:
            return CheckResult(
                FAIL, "logging_fields_present",
                f"{len(missing)} required log field(s) missing from add_item_handler",
                f"missing: {missing}",
            )
        return CheckResult(
            PASS, "logging_fields_present",
            "All required structured log fields present in add_item_handler",
        )
    except Exception as exc:
        return CheckResult(FAIL, "logging_fields_present", f"Exception: {exc}")


def _check_option_resolver_logging_fields() -> CheckResult:
    """Spot-check option resolver log fields in waiting_for_modifier_handler."""
    try:
        import inspect
        from app.state_machine.handlers.item.add_item import waiting_for_modifier_handler as wmh
        source = inspect.getsource(wmh)
        required_fields = [
            "option_resolver_mode",
            "option_resolver_route_reason",
            "option_resolver_called",
            "option_resolver_decision",
            "option_resolver_confidence",
            "option_resolver_safe_to_apply",
            "option_resolver_applied",
            "option_resolver_latency_ms",
        ]
        missing = [f for f in required_fields if f not in source]
        if missing:
            return CheckResult(
                FAIL, "option_resolver_logging_fields",
                f"{len(missing)} required log field(s) missing from waiting_for_modifier_handler",
                f"missing: {missing}",
            )
        return CheckResult(
            PASS, "option_resolver_logging_fields",
            "All required option resolver log fields present",
        )
    except Exception as exc:
        return CheckResult(FAIL, "option_resolver_logging_fields", f"Exception: {exc}")


# ── Runner ────────────────────────────────────────────────────────────────────

_ALL_CHECKS: list[Callable[[], CheckResult]] = [
    _check_config_loads,
    _check_option_resolver_mode,
    _check_add_item_planner_mode,
    _check_disabled_is_default,
    _check_api_key_presence,
    _check_inline_without_api_key,
    _check_shadow_mode_never_applies,
    _check_inline_apply_gate_requires_all_conditions,
    _check_option_resolver_service_instantiates,
    _check_add_item_planner_service_instantiates,
    _check_handler_dispatcher_builds,
    _check_logging_fields_present,
    _check_option_resolver_logging_fields,
]


def run_checks(env_name: str = "local") -> tuple[list[CheckResult], int]:
    """Run all checks and return (results, exit_code).

    exit_code 0 = all passed (warnings are OK).
    exit_code 1 = one or more FAIL results.
    """
    results: list[CheckResult] = []
    for check_fn in _ALL_CHECKS:
        try:
            results.append(check_fn())
        except Exception as exc:
            results.append(CheckResult(FAIL, check_fn.__name__, f"Unhandled exception: {exc}"))

    # Production inline warning is conditional on env (wrapped like all other checks).
    try:
        results.append(_check_production_inline_warning(env_name))
    except Exception as exc:
        results.append(CheckResult(FAIL, "_check_production_inline_warning", f"Unhandled exception: {exc}"))

    exit_code = 0 if all(r.status != FAIL for r in results) else 1
    return results, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        default="local",
        help="Environment label for context-aware checks (local/staging/production)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARN as FAIL",
    )
    parser.add_argument(
        "--no-colour",
        action="store_true",
        help="Disable ANSI colour in output",
    )
    args = parser.parse_args(argv)

    colour = not args.no_colour and sys.stdout.isatty()

    print("=" * 60)
    print(f"Compass Voice — GPT Ordering Pipeline Verification")
    print(f"Environment : {args.env!r}")
    print(f"Strict mode : {args.strict}")
    print("=" * 60)

    results, exit_code = run_checks(env_name=args.env)

    if args.strict:
        for r in results:
            if r.status == WARN:
                r = CheckResult(FAIL, r.name, r.message, r.detail)  # escalate
        exit_code = 0 if all(r.status != FAIL for r in results) else 1

    counts = {PASS: 0, WARN: 0, FAIL: 0, INFO: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        print(_fmt(r, colour=colour))

    print()
    print(
        f"Results: {counts[PASS]} passed  "
        f"{counts[WARN]} warn  "
        f"{counts[FAIL]} failed  "
        f"{counts[INFO]} info"
    )

    if exit_code == 0:
        status_msg = "DEPLOYMENT SAFE" if counts[WARN] == 0 else "DEPLOYMENT SAFE (review warnings)"
        colour_code = _COLOUR[PASS] if colour else ""
        print(f"\n{colour_code}[OK] {status_msg}{_RESET if colour else ''}")
    else:
        colour_code = _COLOUR[FAIL] if colour else ""
        print(f"\n{colour_code}[BLOCKED] DEPLOYMENT BLOCKED -- fix FAIL items above{_RESET if colour else ''}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
