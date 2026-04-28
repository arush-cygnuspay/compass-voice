# app/nlu/control_phrase_lexicon.py
"""YAML-backed loader and validator for control phrases.

``ControlPhraseLexicon`` is the single source of truth for affirm/deny/control
phrase sets.  ``ControlIntentResolver`` is the primary runtime consumer;
``linguistic_rules`` uses the same singleton for NLU-level affirm/deny detection.

Public API
----------
ControlPhraseLexicon.load(path) -> ControlPhraseLexicon
ControlPhraseLexicon.is_affirm(text) -> bool
ControlPhraseLexicon.is_deny(text) -> bool
ControlPhraseLexicon.get_phrases(category) -> frozenset[str]
ControlPhraseLexicon.get_substring_phrases(category) -> tuple[str, ...]

Module-level singleton
----------------------
``DEFAULT_LEXICON`` is loaded once at import time from ``control_phrases.yaml``
sitting next to this file.  Import it rather than calling ``load()`` at every
call site.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.nlu.query_normalization.text_preprocessor import normalize_text

# ---------------------------------------------------------------------------
# YAML duplicate-key detection
# ---------------------------------------------------------------------------

class _StrictLoader(yaml.SafeLoader):
    pass


# PyYAML 1.1 treats "yes"/"no"/"on"/"off" as booleans by default.
# Disable all implicit bool resolution so phrase values stay as strings.
for _ch, _resolvers in list(_StrictLoader.yaml_implicit_resolvers.items()):
    _StrictLoader.yaml_implicit_resolvers[_ch] = [
        (tag, regexp)
        for tag, regexp in _resolvers
        if tag != "tag:yaml.org,2002:bool"
    ]


def _construct_strict_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    pairs = loader.construct_pairs(node)
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"Duplicate key in control_phrases.yaml: {key!r}")
        seen.add(key)
    return dict(pairs)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_strict_mapping,
)

# ---------------------------------------------------------------------------
# Candidate-generation (mirrors control_intent_resolver._signal_candidates)
# ---------------------------------------------------------------------------

_LEADING_FILLERS: tuple[str, ...] = (
    "well",
    "so",
    "just",
    "please",
    "uh",
    "um",
    "hmm",
    "okay",
    "ok",
    "yeah",
    "yep",
    "yup",
    "yes",
)
_TRAILING_FILLERS: tuple[str, ...] = (
    "please",
    "thanks",
    "thank you",
)


def _signal_candidates(text: str) -> set[str]:
    """Return the set of stripped candidates for phrase matching."""
    normalized = normalize_text(text)
    if not normalized:
        return set()

    candidates: set[str] = {normalized}
    queue: list[str] = [normalized]
    seen: set[str] = set()

    while queue:
        value = queue.pop()
        if not value or value in seen:
            continue
        seen.add(value)
        candidates.add(value)

        for filler in _LEADING_FILLERS:
            prefix = f"{filler} "
            if value.startswith(prefix):
                queue.append(value[len(prefix):].strip())

        for filler in _TRAILING_FILLERS:
            suffix = f" {filler}"
            if value.endswith(suffix):
                queue.append(value[: -len(suffix)].strip())

        if " and " in value:
            queue.extend(part.strip() for part in value.split(" and ") if part.strip())

    return candidates


# ---------------------------------------------------------------------------
# ControlPhraseLexicon
# ---------------------------------------------------------------------------

_CANDIDATE = "candidate"
_SUBSTRING = "substring"
_VALID_MATCH_TYPES = frozenset({_CANDIDATE, _SUBSTRING})


class ControlPhraseLexicon:
    """Loaded and validated control-phrase lexicon backed by a YAML file.

    Immutable after construction — all phrase sets are frozen at load time.
    """

    def __init__(
        self,
        candidate_phrases: dict[str, frozenset[str]],
        substring_phrases: dict[str, tuple[str, ...]],
    ) -> None:
        self._candidate = candidate_phrases
        self._substring = substring_phrases

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "ControlPhraseLexicon":
        """Load and validate a control_phrases.yaml file.

        Parameters
        ----------
        path:
            Absolute or relative path to the YAML file.

        Raises
        ------
        FileNotFoundError
            When *path* does not exist.
        ValueError
            When the YAML structure is invalid (missing keys, bad match_type,
            empty phrase lists, duplicate category keys).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"control_phrases.yaml not found: {path}")

        with path.open(encoding="utf-8") as fh:
            raw = yaml.load(fh, Loader=_StrictLoader)  # noqa: S506

        if not isinstance(raw, dict) or "categories" not in raw:
            raise ValueError("control_phrases.yaml must have a top-level 'categories' key")

        categories = raw["categories"]
        if not isinstance(categories, dict):
            raise ValueError("'categories' must be a YAML mapping")

        candidate_phrases: dict[str, frozenset[str]] = {}
        substring_phrases: dict[str, tuple[str, ...]] = {}

        for category, entry in categories.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Category {category!r} must be a mapping")

            match_type = entry.get("match_type")
            if match_type not in _VALID_MATCH_TYPES:
                raise ValueError(
                    f"Category {category!r} has invalid match_type {match_type!r}. "
                    f"Must be one of: {sorted(_VALID_MATCH_TYPES)}"
                )

            phrases = entry.get("phrases")
            if not phrases or not isinstance(phrases, list):
                raise ValueError(
                    f"Category {category!r} must have a non-empty 'phrases' list"
                )

            normalized: list[str] = []
            for phrase in phrases:
                if not isinstance(phrase, str) or not phrase.strip():
                    raise ValueError(
                        f"Category {category!r} contains an empty or non-string phrase"
                    )
                n = normalize_text(phrase)
                if not n:
                    raise ValueError(
                        f"Category {category!r}: phrase {phrase!r} normalizes to empty"
                    )
                normalized.append(n)

            if match_type == _CANDIDATE:
                candidate_phrases[category] = frozenset(normalized)
            else:
                substring_phrases[category] = tuple(dict.fromkeys(normalized))

        return cls(candidate_phrases, substring_phrases)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def is_affirm(self, text: str) -> bool:
        """Return True when *text* candidate-matches any affirm phrase."""
        phrases = self._candidate.get("affirm", frozenset())
        if not phrases:
            return False
        return any(c in phrases for c in _signal_candidates(text) if c)

    def is_deny(self, text: str) -> bool:
        """Return True when *text* candidate-matches any deny phrase."""
        phrases = self._candidate.get("deny", frozenset())
        if not phrases:
            return False
        return any(c in phrases for c in _signal_candidates(text) if c)

    def get_phrases(self, category: str) -> frozenset[str]:
        """Return the candidate-matched phrase set for *category*.

        Returns an empty frozenset when *category* is unknown or has
        match_type ``substring``.
        """
        return self._candidate.get(category, frozenset())

    def get_substring_phrases(self, category: str) -> tuple[str, ...]:
        """Return the substring-matched phrase tuple for *category*.

        Returns an empty tuple when *category* is unknown or has
        match_type ``candidate``.
        """
        return self._substring.get(category, ())

    def categories(self) -> frozenset[str]:
        """Return all known category names."""
        return frozenset(self._candidate) | frozenset(self._substring)


# ---------------------------------------------------------------------------
# Module-level singleton — loaded once at import time
# ---------------------------------------------------------------------------

DEFAULT_LEXICON: ControlPhraseLexicon = ControlPhraseLexicon.load(
    Path(__file__).with_name("control_phrases.yaml")
)
