# tests/menu/test_category_coverage.py
"""Phase 6 — at least one add-item smoke test per active menu category.

Goal: detect the day a category gets added to menu.json with broken
references or unresolvable items, and ensure the resolver can name
*something* in every public-facing category.

Implementation notes
--------------------
* Test cases are **generated at collection time** from the live menu
  fixture, so when the menu changes, the parametrization changes too.
  No hardcoding of category lists.
* We pick a representative item per category by preferring ones with
  fewer required side / modifier groups (cheaper happy path).
* This is a *smoke* test — we assert that the resolver does NOT bounce
  to `item_not_found`. We do NOT assert the full state machine walk
  per category (that would be 26 long flow tests).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest

from app.nlu.intent_resolution.intent import Intent
from app.state_machine.models.conversation_state import ConversationState
from tests.support.voice_test_harness import (
    ScriptedTurn,
    build_engine,
    build_menu_repo,
    make_slot,
    new_session,
    simulate_turn,
)


# ─── menu introspection (collection-time, no engine dependency) ──────────────

_MENU_PATH = (
    Path(__file__).resolve().parents[2]
    / "app" / "data" / "restaurants" / "steves_grill" / "menu.json"
)


def _load_menu() -> dict:
    with _MENU_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _representative_items() -> list[tuple[str, str, str]]:
    """Return [(category_name, item_name, item_id), ...] — one per category.

    Skips:
      * categories whose only item is a gift card or other non-orderable.
      * categories with zero resolvable items (which is a P1 menu integrity bug).
    """
    menu = _load_menu()
    items = menu.get("items") or {}
    cats = menu.get("categories") or {}

    skip_cats = {"Gift Card", "EBT"}  # not part of food-ordering happy path

    chosen: list[tuple[str, str, str]] = []
    for cid, cat in cats.items():
        cname = cat.get("name", "?")
        if cname in skip_cats:
            continue
        candidates = []
        for iid in cat.get("item_ids", []) or []:
            it = items.get(iid)
            if not it:
                continue
            sg = len(it.get("side_groups") or [])
            mg = len(it.get("modifier_groups") or [])
            # Prefer items with the fewest required groups so the smoke
            # test only verifies resolution, not multi-turn flow.
            candidates.append((sg + mg, it.get("name", "?"), iid))
        if not candidates:
            continue
        candidates.sort()
        _, name, iid = candidates[0]
        chosen.append((cname, name, iid))
    return chosen


_REPRESENTATIVE_ITEMS = _representative_items()


# ─── menu integrity ─────────────────────────────────────────────────────────

def test_menu_has_resolvable_items_in_every_category() -> None:
    menu = _load_menu()
    items = menu.get("items") or {}
    cats = menu.get("categories") or {}
    empty = []
    for cid, cat in cats.items():
        if not any(items.get(iid) for iid in (cat.get("item_ids") or [])):
            empty.append(cat.get("name"))
    assert not empty, f"categories with zero resolvable items: {empty!r}"


def test_menu_has_no_null_item_entries() -> None:
    menu = _load_menu()
    items = menu.get("items") or {}
    nulls = [iid for iid, it in items.items() if it is None]
    assert not nulls, f"null item entries in menu.json: {nulls[:5]!r}"


def test_menu_has_no_broken_category_item_refs() -> None:
    menu = _load_menu()
    items = menu.get("items") or {}
    cats = menu.get("categories") or {}
    broken = []
    for cid, cat in cats.items():
        missing = [
            iid for iid in (cat.get("item_ids") or [])
            if iid not in items or items[iid] is None
        ]
        if missing:
            broken.append((cat.get("name"), missing[:2]))
    assert not broken, f"broken category->item refs: {broken!r}"


# ─── per-category smoke test ────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("category_name", "item_name", "item_id"),
    _REPRESENTATIVE_ITEMS,
    ids=[c for c, *_ in _REPRESENTATIVE_ITEMS],
)
def test_each_category_resolves_its_representative_item(
    category_name: str,
    item_name: str,
    item_id: str,
) -> None:
    """Pick the simplest item per category and confirm the engine resolves
    it (not item_not_found / item_clarification_limit_reached)."""
    engine = build_engine(menu_repo=build_menu_repo())
    session = new_session()
    simulate_turn(engine, session, ScriptedTurn("pickup"))

    result = simulate_turn(
        engine,
        session,
        ScriptedTurn(
            item_name,
            intent=Intent.ADD_ITEM,
            slots=(make_slot("ITEM", item_name),),
        ),
    )

    assert result.response_key not in {
        "item_not_found",
        "item_not_found_near_miss",
        "item_clarification_limit_reached",
        "intent_not_allowed",
    }, (
        f"category {category_name!r} representative {item_name!r} did not "
        f"resolve cleanly: response_key={result.response_key!r} "
        f"text={result.response_text!r}"
    )
