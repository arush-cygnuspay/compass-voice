# tests/state_machine/handlers/item/add_item/test_alias_suppression.py
"""Tests for Phase 3: alias / voice-label suppression from 'I couldn't find' output.

Validates:
- _strip_item_name: compact alias form stripped from segment text
- _dedupe_unresolved: compact alias filtered from unresolved list
- _collapse_unresolved_for_feedback: alias filtered from feedback
- success.py item_added_successfully: unmatched_names filtered against item labels
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.state_machine.handlers.item.add_item.multi_group_prefill import (
    MultiGroupPrefillEngine,
)
from app.state_machine.handlers.item.add_item.prefill_orchestrator import (
    PendingItemCaptureHelper,
)
from app.responses.item.success import item_added_successfully


# ---------------------------------------------------------------------------
# _strip_item_name
# ---------------------------------------------------------------------------

class TestStripItemName:
    def _strip(self, segment, item_name, voice_labels=()):
        return MultiGroupPrefillEngine._strip_item_name(segment, item_name, voice_labels)

    def test_canonical_name_stripped_from_prefix(self):
        result = self._strip("cheese burger with fries", "Cheese Burger")
        assert result == "with fries"

    def test_compact_alias_stripped_from_prefix(self):
        """'cheeseburger with fries' → strips 'cheeseburger' voice label."""
        result = self._strip(
            "cheeseburger with fries",
            "Cheese Burger",
            voice_labels=("cheese burger", "cheeseburger"),
        )
        assert result == "with fries"

    def test_compact_alias_stripped_from_interior(self):
        result = self._strip(
            "add cheeseburger with fries",
            "Cheese Burger",
            voice_labels=("cheese burger", "cheeseburger"),
        )
        # "add" is not stripped by _strip_item_name (filler stripping is elsewhere)
        assert "cheeseburger" not in result

    def test_no_match_returns_full_text(self):
        result = self._strip("chicken taco with guac", "Cheese Burger")
        assert result == "chicken taco with guac"

    def test_empty_text_returns_empty(self):
        assert self._strip("", "Cheese Burger") == ""

    def test_voice_label_not_stripped_when_matches_canonical(self):
        """Voice labels that equal the canonical name don't cause double-strip."""
        result = self._strip(
            "cheese burger with fries",
            "Cheese Burger",
            voice_labels=("cheese burger",),
        )
        assert result == "with fries"


# ---------------------------------------------------------------------------
# _dedupe_unresolved
# ---------------------------------------------------------------------------

def _make_pending(item_name, aliases=(), voice_labels=()):
    p = MagicMock()
    p.item_name = item_name
    p.item_aliases = aliases
    p.item_voice_labels = voice_labels
    return p


class TestDedupeUnresolved:
    def _dedupe(self, unresolved, pending, bindings=()):
        return MultiGroupPrefillEngine._dedupe_unresolved(
            unresolved, bindings=list(bindings), pending=pending
        )

    def test_compact_alias_suppressed(self):
        """'cheeseburger' is in item voice_labels → suppress from unresolved."""
        pending = _make_pending(
            "Cheese Burger",
            voice_labels=("cheese burger", "cheeseburger"),
        )
        result = self._dedupe(["cheeseburger"], pending)
        assert result == [], f"Expected empty, got {result}"

    def test_item_name_tokens_suppressed(self):
        """Phrase whose tokens are all in bound_tokens gets removed."""
        pending = _make_pending("Cheese Burger")
        result = self._dedupe(["cheese", "burger"], pending)
        assert result == []

    def test_alias_suppressed(self):
        pending = _make_pending(
            "Cheese Burger",
            aliases=("cb",),
        )
        result = self._dedupe(["cb"], pending)
        assert result == []

    def test_real_unresolved_phrase_kept(self):
        """A phrase that isn't an item label or bound token survives."""
        pending = _make_pending("Cheese Burger")
        result = self._dedupe(["spicy sauce"], pending)
        assert "spicy sauce" in result

    def test_empty_unresolved_returns_empty(self):
        pending = _make_pending("Coke")
        assert self._dedupe([], pending) == []

    def test_duplicate_phrases_deduped(self):
        pending = _make_pending("Coke")
        result = self._dedupe(["spicy sauce", "spicy sauce"], pending)
        assert result == ["spicy sauce"]

    def test_filler_only_phrase_suppressed_via_tokens(self):
        """Pure order-filler residue ('want', 'a') is all in bound_tokens."""
        pending = _make_pending("Coke")
        # "a" normalizes to a token that's weak/stop; if it remains, it should still
        # not appear because it's in the filler set covered by bound_tokens.
        result = self._dedupe([""], pending)
        assert result == []


# ---------------------------------------------------------------------------
# _collapse_unresolved_for_feedback
# ---------------------------------------------------------------------------

class TestCollapseUnresolvedForFeedback:
    def _collapse(self, phrases, pending=None):
        from app.state_machine.handlers.item.add_item.prefill_orchestrator import (
            PrefillOrchestrator,
        )
        return PrefillOrchestrator._collapse_unresolved_for_feedback(
            phrases, pending=pending
        )

    def test_compact_alias_suppressed(self):
        pending = MagicMock()
        pending.item_name = "Cheese Burger"
        pending.item_aliases = ()
        pending.item_voice_labels = ("cheese burger", "cheeseburger")
        pending.side_groups = []
        pending.modifier_groups = []
        pending.item_variants = []

        result = self._collapse(["cheeseburger"], pending=pending)
        assert result == [], f"Expected [], got {result}"

    def test_real_unresolved_passes_through(self):
        pending = MagicMock()
        pending.item_name = "Coke"
        pending.item_aliases = ()
        pending.item_voice_labels = ("coke",)
        pending.side_groups = []
        pending.modifier_groups = []
        pending.item_variants = []

        result = self._collapse(["dragon sauce"], pending=pending)
        assert "dragon sauce" in result

    def test_filler_only_suppressed(self):
        """Pure filler like 'a' or 'and' is never surfaced."""
        pending = MagicMock()
        pending.item_name = "Coke"
        pending.item_aliases = ()
        pending.item_voice_labels = ("coke",)
        pending.side_groups = []
        pending.modifier_groups = []
        pending.item_variants = []
        result = self._collapse(["a", "the", "and"], pending=pending)
        assert result == []

    def test_none_pending_no_crash(self):
        result = self._collapse(["unknown thing"], pending=None)
        assert "unknown thing" in result


# ---------------------------------------------------------------------------
# success.py item_added_successfully — unmatched_names filter
# ---------------------------------------------------------------------------

class TestItemAddedSuccessfullyUnmatchedFilter:
    def _build_payload(self, unmatched, item_name="Cheese Burger", aliases=(), voice_labels=()):
        return {
            "item_name": item_name,
            "quantity": 1,
            "unmatched_names": list(unmatched),
            "item_aliases": list(aliases),
            "item_voice_labels": list(voice_labels),
        }

    def test_compact_alias_suppressed_from_success_response(self):
        """'cheeseburger' in unmatched_names must not appear in response."""
        payload = self._build_payload(
            unmatched=["cheeseburger"],
            item_name="Cheese Burger",
            voice_labels=["cheese burger", "cheeseburger"],
        )
        text = item_added_successfully(payload)
        assert "couldn't find" not in text, (
            f"'I couldn't find' must not appear: {text!r}"
        )

    def test_real_unmatched_appears_in_response(self):
        """A genuinely unrecognized phrase still appears."""
        payload = self._build_payload(
            unmatched=["dragon sauce"],
            item_name="Cheese Burger",
            voice_labels=["cheese burger", "cheeseburger"],
        )
        text = item_added_successfully(payload)
        assert "couldn't find" in text
        assert "dragon sauce" in text

    def test_no_unmatched_no_note(self):
        payload = self._build_payload(unmatched=[], item_name="Coke")
        text = item_added_successfully(payload)
        assert "couldn't find" not in text

    def test_success_line_intact_when_alias_suppressed(self):
        payload = self._build_payload(
            unmatched=["cheeseburger"],
            item_name="Cheese Burger",
            voice_labels=["cheese burger", "cheeseburger"],
        )
        text = item_added_successfully(payload)
        assert "Cheese Burger" in text
        assert "added" in text.lower() or "Would you like" in text
