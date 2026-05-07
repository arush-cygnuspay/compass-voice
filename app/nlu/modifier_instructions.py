# app/nlu/modifier_instructions.py
"""Single source of truth for modifier instructions (no / extra / less / on the side / add).

A "modifier instruction" is the verb-or-quantifier wrapper a customer puts
around a modifier name:

    "no onions"            → action=REMOVE, instruction=NONE,    target="onions"
    "extra cheese"         → action=ADD,    instruction=EXTRA,   target="cheese"
    "double bacon"         → action=ADD,    instruction=EXTRA,   target="bacon"
    "light mayo"           → action=ADD,    instruction=LESS,    target="mayo"
    "easy on the salt"     → action=ADD,    instruction=LESS,    target="salt"
    "go light on mayo"     → action=ADD,    instruction=LESS,    target="mayo"
    "half cheese"          → action=ADD,    instruction=LESS,    target="cheese"
    "cheese on the side"   → action=ADD,    instruction=ON_SIDE, target="cheese"
    "ranch on the side please" → action=ADD,instruction=ON_SIDE, target="ranch"
    "skip the lettuce"     → action=REMOVE, instruction=NONE,    target="lettuce"
    "drop the pickles"     → action=REMOVE, instruction=NONE,    target="pickles"
    "kill the onions"      → action=REMOVE, instruction=NONE,    target="onions"
    "hold the salt"        → action=REMOVE, instruction=NONE,    target="salt"
    "without mushrooms"    → action=REMOVE, instruction=NONE,    target="mushrooms"
    "add bacon"            → action=ADD,    instruction=NONE,    target="bacon"
    "with avocado"         → action=ADD,    instruction=NONE,    target="avocado"

Why this module exists
----------------------
Modifier instruction parsing previously lived in three different places:
- `multi_group_prefill._clean_phrase`
- `modifier_group_resolver._parse` / `_extract_remove_targets`
- the response layer (`format_utils._current_modifier_payload`,
  `prefill_orchestrator._build_prefilled_summary`)

Each had its own copy of the prefix/suffix tables, and each table had drifted.
That is why "easy on the salt" and "skip the lettuce" never worked, why "extra
cheese" sometimes lost its instruction (the slot path added "cheese" with no
instruction first, then deduped the instruction-bearing split candidate), and
why "X on the side" did not echo back in the success message.

This module owns:
1. The lexicon (one table per instruction class).
2. `parse_phrase(text) → ModifierIntent` — the single parser.
3. `priority(intent)` — used to merge two parses for the same target.
4. `speak(name, action, instruction)` — natural speech rendering, used by
   every response site so "no onions" and "extra cheese" surface the same
   way everywhere.

Add new aliases here ("kill the X", "hold the X", "go light on X", ...).
Resolvers and response builders pick them up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


# ── Public types ─────────────────────────────────────────────────────────────

class Action(str, Enum):
    ADD = "add"
    REMOVE = "remove"


class Instruction(str, Enum):
    NONE = ""
    EXTRA = "extra"      # "extra X", "double X", "more X", "lots of X"
    LESS = "less"        # "less X", "light on X", "easy on X", "half X"
    ON_SIDE = "on_side"  # "X on the side"


@dataclass(frozen=True, slots=True)
class ModifierIntent:
    """Structured representation of one customer-spoken modifier phrase."""
    action: Action
    instruction: Instruction
    target: str           # the modifier-name candidate ("cheese", "onions", ...)
    raw: str              # original phrase as spoken (for audit / debug)


# ── Lexicon ──────────────────────────────────────────────────────────────────
# IMPORTANT: longest-first within each tuple so the greedy prefix loop hits
# the most specific match (e.g. "hold the X" wins over "hold X").

_REMOVE_PREFIXES: tuple[str, ...] = (
    # multi-word with article
    "hold the ",
    "remove the ",
    "skip the ",
    "drop the ",
    "kill the ",
    "without the ",
    "without any ",
    "no the ",         # rare STT artefact ("no the onions")
    # bare verbs
    "hold ",
    "remove ",
    "skip ",
    "drop ",
    "kill ",
    "without ",
    "no ",
)

_EXTRA_PREFIXES: tuple[str, ...] = (
    "extra extra ",     # idempotent ("extra extra cheese")
    "double the ",
    "triple the ",
    "tons of ",
    "lots of ",
    "lot of ",
    "loads of ",
    "plenty of ",
    "a lot of ",
    "extra ",
    "double ",
    "triple ",
    "more ",
)

_LESS_PREFIXES: tuple[str, ...] = (
    "go light on the ",
    "go easy on the ",
    "easy on the ",
    "light on the ",
    "go light on ",
    "go easy on ",
    "easy on ",
    "light on ",
    "half the ",
    "half ",
    "less ",
    "light ",
)

_ON_SIDE_SUFFIXES: tuple[str, ...] = (
    " on the side please",
    " on the side",
    " on side",
)

# ADD prefixes are filler-strippers only — they do not change action/instruction,
# they just clean leading verbs so the target is matchable.
_ADD_PREFIXES: tuple[str, ...] = (
    "add the ",
    "add some ",
    "add ",
    "with some ",
    "with the ",
    "with ",
    "can i get ",
    "can i have ",
    "get me ",
)

# Conflict-resolution priority. REMOVE always wins over any ADD; a specific
# instruction always wins over a bare ADD.  This is what `merge()` uses so
# "extra cheese" beats a concurrent slot-only "cheese", and "no cheese" beats
# both "extra cheese" and bare "cheese".
_INSTRUCTION_PRIORITY: dict[tuple[str, str], int] = {
    (Action.REMOVE.value, Instruction.NONE.value):    100,
    (Action.ADD.value,    Instruction.EXTRA.value):    50,
    (Action.ADD.value,    Instruction.LESS.value):     40,
    (Action.ADD.value,    Instruction.ON_SIDE.value):  30,
    (Action.ADD.value,    Instruction.NONE.value):     10,
}


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_phrase(raw: str) -> Optional[ModifierIntent]:
    """Parse a single raw phrase into a ModifierIntent.

    Returns None for empty / unparseable input.

    Resolution order:
      1. Strip trailing ON_SIDE suffix (it co-exists with any action).
      2. REMOVE prefix wins next (longest first).
      3. EXTRA prefix.
      4. LESS prefix.
      5. ADD prefix is filler-only, peeled and ignored.
    """
    text = (raw or "").strip().lower()
    if not text:
        return None

    # 1. ON_SIDE suffix — strip first so subsequent prefix parsers see
    #    only the target portion.
    on_side = False
    for suffix in _ON_SIDE_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            on_side = True
            break

    if not text:
        return None

    # 2. REMOVE prefixes.
    for prefix in _REMOVE_PREFIXES:
        if text.startswith(prefix):
            target = text[len(prefix):].strip()
            target = _peel_leading_article(target)
            if not target:
                return None
            return ModifierIntent(
                action=Action.REMOVE,
                instruction=Instruction.NONE,
                target=target,
                raw=raw,
            )

    # 3. EXTRA prefixes.
    for prefix in _EXTRA_PREFIXES:
        if text.startswith(prefix):
            target = text[len(prefix):].strip()
            target = _peel_leading_article(target)
            if not target:
                return None
            return ModifierIntent(
                action=Action.ADD,
                instruction=Instruction.EXTRA,
                target=target,
                raw=raw,
            )

    # 4. LESS prefixes.
    for prefix in _LESS_PREFIXES:
        if text.startswith(prefix):
            target = text[len(prefix):].strip()
            target = _peel_leading_article(target)
            if not target:
                return None
            return ModifierIntent(
                action=Action.ADD,
                instruction=Instruction.LESS,
                target=target,
                raw=raw,
            )

    # 5. ADD prefixes are filler-strippers only.
    for prefix in _ADD_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    text = _peel_leading_article(text)
    if not text:
        return None

    # Reject a bare instruction keyword as a target — "no" / "extra" / "without"
    # spoken alone is a control phrase, not a modifier.  This guards the deny
    # / affirm intent classifier upstream from being shadowed by an empty
    # ModifierIntent that no resolver can act on.
    if text in _STANDALONE_NON_TARGETS:
        return None

    return ModifierIntent(
        action=Action.ADD,
        instruction=Instruction.ON_SIDE if on_side else Instruction.NONE,
        target=text,
        raw=raw,
    )


# Tokens that may *prefix* a modifier but are never themselves a modifier name.
# Built once at import time from the lexicon so the set stays in sync.
def _build_standalone_non_targets() -> frozenset[str]:
    bare: set[str] = set()
    for prefix in (*_REMOVE_PREFIXES, *_EXTRA_PREFIXES, *_LESS_PREFIXES, *_ADD_PREFIXES):
        for token in prefix.strip().split():
            if token:
                bare.add(token)
    return frozenset(bare)


_STANDALONE_NON_TARGETS: frozenset[str] = _build_standalone_non_targets()


def _peel_leading_article(text: str) -> str:
    for article in ("the ", "a ", "an ", "some "):
        if text.startswith(article):
            return text[len(article):].strip()
    return text


# ── Conflict resolution ──────────────────────────────────────────────────────

def priority(intent: ModifierIntent) -> int:
    """Higher = wins when two parses target the same modifier."""
    return _INSTRUCTION_PRIORITY.get(
        (intent.action.value, intent.instruction.value),
        _INSTRUCTION_PRIORITY[(Action.ADD.value, Instruction.NONE.value)],
    )


def merge(existing: ModifierIntent, candidate: ModifierIntent) -> ModifierIntent:
    """Return whichever intent should prevail for the same target."""
    return candidate if priority(candidate) > priority(existing) else existing


# ── Speech rendering ─────────────────────────────────────────────────────────
# Used by every response site that voices a modifier back to the caller.
# Keeps the bot's voice consistent with the customer's intent.

def speak(
    modifier_name: str,
    action: str = "add",
    instruction: Optional[str] = None,
) -> str:
    """Render a single modifier as natural speech.

        speak("cheese", "remove")             → "no cheese"
        speak("bacon",  "add", "extra")       → "extra bacon"
        speak("mayo",   "add", "less")        → "light mayo"
        speak("ranch",  "add", "on_side")     → "ranch on the side"
        speak("pickles")                      → "pickles"

    Accepts string OR enum input for action/instruction so callers that
    persist plain strings (e.g. ``ModifierSelection``) work without
    conversion.
    """
    name = (modifier_name or "").strip()
    if not name:
        return ""

    act = action.value if isinstance(action, Action) else (action or "add")
    inst = (
        instruction.value
        if isinstance(instruction, Instruction)
        else (instruction or "")
    )

    if act == Action.REMOVE.value:
        return f"no {name}"
    if inst == Instruction.EXTRA.value:
        return f"extra {name}"
    if inst == Instruction.LESS.value:
        # "light X" sounds more natural than "less X" in voice; the LESS
        # instruction covers both inputs.
        return f"light {name}"
    if inst == Instruction.ON_SIDE.value:
        return f"{name} on the side"
    return name


def speak_join(spoken_modifiers: Iterable[str]) -> str:
    """Format already-spoken modifier strings into a comma/and clause.

        []                                     → ""
        ["no onions"]                          → "no onions"
        ["no onions","extra cheese"]           → "no onions and extra cheese"
        ["no onions","extra cheese","light mayo"]
                                               → "no onions, extra cheese, and light mayo"
    """
    parts = [
        str(x).strip()
        for x in spoken_modifiers
        if x is not None and str(x).strip()
    ]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


# ── Backward-compatible exports for the existing call sites ──────────────────
# So legacy imports (multi_group_prefill, modifier_group_resolver) keep
# working as we migrate them.

REMOVE_PREFIXES = _REMOVE_PREFIXES
EXTRA_PREFIXES = _EXTRA_PREFIXES
LESS_PREFIXES = _LESS_PREFIXES
ON_SIDE_SUFFIXES = _ON_SIDE_SUFFIXES
ADD_PREFIXES = _ADD_PREFIXES

__all__ = [
    "Action",
    "Instruction",
    "ModifierIntent",
    "parse_phrase",
    "priority",
    "merge",
    "speak",
    "speak_join",
    "REMOVE_PREFIXES",
    "EXTRA_PREFIXES",
    "LESS_PREFIXES",
    "ON_SIDE_SUFFIXES",
    "ADD_PREFIXES",
]
