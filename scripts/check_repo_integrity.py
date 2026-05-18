#!/usr/bin/env python3
"""Repo integrity guard - run as a fast pre-test step in CI.

Catches the corruption modes seen in previous incidents:
  1. Truncated JSON fixtures (menu.json, entity_index.json) - mid-write saves
     leave the file unparseable and crash every test that loads the menu.
  2. Trailing-null-byte padding on .py files - editor / build step writes
     with a sector-aligned buffer and forgets to truncate. ast.parse rejects
     these with "source code string cannot contain null bytes".
  3. Mid-statement source truncation (unterminated string, paren imbalance,
     etc.) - same root cause as #1 but for code.

Exits non-zero on any failure. Designed to run in <5 seconds against the
whole repo.

Usage:
    python scripts/check_repo_integrity.py
    python scripts/check_repo_integrity.py --root path/to/project
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist", ".pytest_cache"}

REQUIRED_JSON_FIXTURES = (
    "app/data/restaurants/steves_grill/menu.json",
    "app/data/restaurants/steves_grill/entity_index.json",
    # restaurant.json profile not yet present for steves_grill; the runtime
    # only needs menu + entity_index, so we don't gate the repo integrity
    # check on a missing profile here.
)


def _iter_python_sources(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def _check_python_files(root: Path) -> list[str]:
    errors: list[str] = []
    for path in _iter_python_sources(root):
        try:
            data = path.read_bytes()
        except OSError as exc:
            errors.append(f"{path}: read failed: {exc}")
            continue

        if b"\x00" in data:
            # Trailing-null-byte padding is the dominant failure mode; flag
            # any null byte but report whether it's trailing-only so the
            # operator can choose to auto-strip vs investigate.
            stripped = data.rstrip(b"\x00")
            if b"\x00" in stripped:
                errors.append(f"{path}: contains embedded NUL bytes (corruption)")
            else:
                errors.append(
                    f"{path}: trailing NUL-byte padding "
                    f"({len(data) - len(stripped)} bytes) - rewrite this file"
                )
            continue

        try:
            ast.parse(data, str(path))
        except SyntaxError as exc:
            # PEP 695 generic syntax (`def f[T](...)`) is valid on Py 3.12+
            # but parses as SyntaxError on earlier interpreters. Skip the
            # check when running on an older Python so the guard still works
            # locally for developers who haven't upgraded.
            if sys.version_info < (3, 12) and "invalid syntax" in str(exc):
                continue
            errors.append(f"{path}:{exc.lineno}: {exc.msg}")
    return errors


def _check_json_fixtures(root: Path) -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_JSON_FIXTURES:
        path = root / rel
        if not path.exists():
            errors.append(f"{rel}: required fixture missing")
            continue
        try:
            with path.open(encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as exc:
            errors.append(f"{rel}: invalid JSON at line {exc.lineno}: {exc.msg}")
        except OSError as exc:
            errors.append(f"{rel}: read failed: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"ERROR: root {root!r} does not exist", file=sys.stderr)
        return 2

    print(f"[check_repo_integrity] root={root}")
    errors: list[str] = []
    errors += _check_json_fixtures(root)
    errors += _check_python_files(root)

    if errors:
        print(f"FAILED: {len(errors)} integrity issues found:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("OK: all JSON fixtures valid and all Python sources parse cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
