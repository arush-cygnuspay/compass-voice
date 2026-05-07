# tests/nlu/test_modifier_instructions.py
"""Coverage for the canonical modifier-instruction parser, priority, and speech.

These tests pin the contract that every resolver and response site relies on.
If you add a new alias to app/nlu/modifier_instructions.py, add a test here
too — the lexicon is the only place a regression can hide silently.
"""
from __future__ import annotations

import pytest

from app.nlu.modifier_instructions import (
    Action,
    Instruction,
    ModifierIntent,
    merge,
    parse_phrase,
    priority,
    speak,
    speak_join,
)


# ── parse_phrase: REMOVE family ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, target",
    [
        ("no onions",          "onions"),
        ("no the onions",      "onions"),          # rare STT artefact
        ("without mushrooms",  "mushrooms"),
        ("without the salt",   "salt"),
        ("without any cheese", "cheese"),
        ("hold the salt",      "salt"),
        ("hold salt",          "salt"),
        ("remove the pickles", "pickles"),
        ("remove pickles",     "pickles"),
        ("skip the lettuce",   "lettuce"),
        ("skip lettuce",       "lettuce"),
        ("drop the bacon",     "bacon"),
        ("kill the onions",    "onions"),
    ],
)
def test_parse_remove_family(raw, target):
    intent = parse_phrase(raw)
    assert intent is not None, raw
    assert intent.action is Action.REMOVE, raw
    assert intent.instruction is Instruction.NONE, raw
    assert intent.target == target, raw


# ── parse_phrase: EXTRA family ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, target",
    [
        ("extra cheese",       "cheese"),
        ("extra extra cheese", "cheese"),
        ("more cheese",        "cheese"),
        ("double bacon",       "bacon"),
        ("double the bacon",   "bacon"),
        ("triple bacon",       "bacon"),
        ("triple the bacon",   "bacon"),
        ("lots of mayo",       "mayo"),
        ("loads of mayo",      "mayo"),
        ("plenty of mayo",     "mayo"),
        ("a lot of mayo",      "mayo"),
        ("tons of mayo",       "mayo"),
    ],
)
def test_parse_extra_family(raw, target):
    intent = parse_phrase(raw)
    assert intent is not None, raw
    assert intent.action is Action.ADD, raw
    assert intent.instruction is Instruction.EXTRA, raw
    assert intent.target == target, raw


# ── parse_phrase: LESS family ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, target",
    [
        ("less mayo",            "mayo"),
        ("light mayo",           "mayo"),
        ("light on the mayo",    "mayo"),
        ("easy on the salt",     "salt"),
        ("easy on salt",         "salt"),
        ("go light on mayo",     "mayo"),
        ("go light on the mayo", "mayo"),
        ("go easy on the salt",  "salt"),
        ("half cheese",          "cheese"),
        ("half the cheese",      "cheese"),
    ],
)
def test_parse_less_family(raw, target):
    intent = parse_phrase(raw)
    assert intent is not None, raw
    assert intent.action is Action.ADD, raw
    assert intent.instruction is Instruction.LESS, raw
    assert intent.target == target, raw


# ── parse_phrase: ON_SIDE suffix ────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, target",
    [
        ("ranch on the side",            "ranch"),
        ("ranch on the side please",     "ranch"),
        ("dressing on side",             "dressing"),
    ],
)
def test_parse_on_side(raw, target):
    intent = parse_phrase(raw)
    assert intent is not None, raw
    assert intent.action is Action.ADD, raw
    assert intent.instruction is Instruction.ON_SIDE, raw
    assert intent.target == target, raw


def test_parse_remove_with_on_side_suffix_remove_wins_action_on_side_for_instruction_dropped():
    """REMOVE prefix wins on action; ON_SIDE suffix is conceptually orthogonal
    but a 'no X on the side' is rare and ambiguous — we treat it as plain REMOVE."""
    intent = parse_phrase("no ranch on the side")
    assert intent is not None
    assert intent.action is Action.REMOVE
    assert intent.target == "ranch"


# ── parse_phrase: ADD prefix is filler-only ─────────────────────────────────

@pytest.mark.parametrize(
    "raw, target",
    [
        ("add bacon",        "bacon"),
        ("add the bacon",    "bacon"),
        ("add some bacon",   "bacon"),
        ("with avocado",     "avocado"),
        ("with the avocado", "avocado"),
        ("with some avocado","avocado"),
        ("can i get bacon",  "bacon"),
        ("get me bacon",     "bacon"),
        ("bacon",            "bacon"),
    ],
)
def test_parse_add_prefix_is_filler(raw, target):
    intent = parse_phrase(raw)
    assert intent is not None, raw
    assert intent.action is Action.ADD, raw
    assert intent.instruction is Instruction.NONE, raw
    assert intent.target == target, raw


# ── parse_phrase: edge / null cases ─────────────────────────────────────────

@pytest.mark.parametrize("raw", ["", "  ", None, "no", "extra", "without"])
def test_parse_returns_none_for_empty_or_target_less(raw):
    assert parse_phrase(raw) is None


# ── priority + merge ────────────────────────────────────────────────────────

def test_priority_remove_beats_extra_beats_less_beats_on_side_beats_add():
    assert priority(_intent(Action.REMOVE, Instruction.NONE,    "x")) > \
           priority(_intent(Action.ADD,    Instruction.EXTRA,   "x"))
    assert priority(_intent(Action.ADD,    Instruction.EXTRA,   "x")) > \
           priority(_intent(Action.ADD,    Instruction.LESS,    "x"))
    assert priority(_intent(Action.ADD,    Instruction.LESS,    "x")) > \
           priority(_intent(Action.ADD,    Instruction.ON_SIDE, "x"))
    assert priority(_intent(Action.ADD,    Instruction.ON_SIDE, "x")) > \
           priority(_intent(Action.ADD,    Instruction.NONE,    "x"))


def test_merge_keeps_higher_priority_intent():
    bare  = _intent(Action.ADD,    Instruction.NONE,    "cheese")
    extra = _intent(Action.ADD,    Instruction.EXTRA,   "cheese")
    rmv   = _intent(Action.REMOVE, Instruction.NONE,    "cheese")
    assert merge(bare, extra) is extra
    assert merge(extra, bare) is extra
    assert merge(extra, rmv)  is rmv
    assert merge(rmv, extra)  is rmv


# ── speak ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "name, action, instruction, expected",
    [
        ("cheese",  "add",    None,      "cheese"),
        ("onions",  "remove", None,      "no onions"),
        ("cheese",  "add",    "extra",   "extra cheese"),
        ("mayo",    "add",    "less",    "light mayo"),
        ("ranch",   "add",    "on_side", "ranch on the side"),
        # Enum inputs work too
        ("cheese",  Action.REMOVE, Instruction.NONE,  "no cheese"),
        ("bacon",   Action.ADD,    Instruction.EXTRA, "extra bacon"),
        # Empty / missing name → empty
        ("",        "add",    None,      ""),
        ("   ",     "add",    None,      ""),
    ],
)
def test_speak(name, action, instruction, expected):
    assert speak(name, action, instruction) == expected


def test_speak_join():
    assert speak_join([]) == ""
    assert speak_join(["no onions"]) == "no onions"
    assert speak_join(["no onions", "extra cheese"]) == "no onions and extra cheese"
    assert (
        speak_join(["no onions", "extra cheese", "light mayo"])
        == "no onions, extra cheese, and light mayo"
    )
    # Empties are filtered
    assert speak_join(["", "no onions", None]) == "no onions"


# ── helpers ─────────────────────────────────────────────────────────────────

def _intent(action: Action, instruction: Instruction, target: str) -> ModifierIntent:
    return ModifierIntent(action=action, instruction=instruction, target=target, raw=target)
