# tests/nlu/test_control_phrase_lexicon.py
"""Tests for ControlPhraseLexicon — YAML loading, validation, and phrase matching."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.nlu.control_phrase_lexicon import ControlPhraseLexicon, DEFAULT_LEXICON


# ===========================================================================
# Helpers
# ===========================================================================


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "phrases.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _minimal_yaml(tmp_path: Path) -> Path:
    return _write_yaml(
        tmp_path,
        """
        categories:
          affirm:
            match_type: candidate
            phrases:
              - yes
              - okay
          deny:
            match_type: candidate
            phrases:
              - no
              - nope
        """,
    )


# ===========================================================================
# YAML loading
# ===========================================================================


class TestLoad:
    def test_loads_default_lexicon(self):
        assert isinstance(DEFAULT_LEXICON, ControlPhraseLexicon)

    def test_all_expected_categories_present(self):
        cats = DEFAULT_LEXICON.categories()
        for expected in ("affirm", "deny", "done", "cancel", "options", "meta_clarify",
                         "skip", "stay_on_call", "after_call", "cannot_open_link"):
            assert expected in cats, f"Missing category: {expected!r}"

    def test_candidate_categories_return_frozenset(self):
        for cat in ("affirm", "deny", "done", "cancel", "options", "meta_clarify", "skip"):
            result = DEFAULT_LEXICON.get_phrases(cat)
            assert isinstance(result, frozenset), f"{cat!r} should return frozenset"
            assert len(result) > 0, f"{cat!r} should be non-empty"

    def test_substring_categories_return_tuple(self):
        for cat in ("stay_on_call", "after_call", "cannot_open_link"):
            result = DEFAULT_LEXICON.get_substring_phrases(cat)
            assert isinstance(result, tuple), f"{cat!r} should return tuple"
            assert len(result) > 0, f"{cat!r} should be non-empty"

    def test_unknown_category_returns_empty_frozenset(self):
        assert DEFAULT_LEXICON.get_phrases("nonexistent") == frozenset()

    def test_unknown_substring_category_returns_empty_tuple(self):
        assert DEFAULT_LEXICON.get_substring_phrases("nonexistent") == ()

    def test_candidate_category_via_substring_accessor_returns_empty(self):
        assert DEFAULT_LEXICON.get_substring_phrases("affirm") == ()

    def test_substring_category_via_candidate_accessor_returns_empty(self):
        assert DEFAULT_LEXICON.get_phrases("stay_on_call") == frozenset()

    def test_phrases_are_normalized(self):
        phrases = DEFAULT_LEXICON.get_phrases("affirm")
        assert "yes" in phrases
        assert "no" not in phrases

    def test_load_from_minimal_yaml(self, tmp_path):
        path = _minimal_yaml(tmp_path)
        lexicon = ControlPhraseLexicon.load(path)
        assert "yes" in lexicon.get_phrases("affirm")
        assert "no" in lexicon.get_phrases("deny")

    def test_phrases_normalized_at_load_time(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
            categories:
              affirm:
                match_type: candidate
                phrases:
                  - "YES"
                  - "  Okay  "
            """,
        )
        lexicon = ControlPhraseLexicon.load(path)
        phrases = lexicon.get_phrases("affirm")
        assert "yes" in phrases
        assert "okay" in phrases


# ===========================================================================
# Validation — malformed YAML
# ===========================================================================


class TestValidation:
    def test_missing_categories_key_raises(self, tmp_path):
        path = _write_yaml(tmp_path, "something: else\n")
        with pytest.raises(ValueError, match="'categories'"):
            ControlPhraseLexicon.load(path)

    def test_invalid_match_type_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
            categories:
              affirm:
                match_type: fuzzy
                phrases: ["yes"]
            """,
        )
        with pytest.raises(ValueError, match="match_type"):
            ControlPhraseLexicon.load(path)

    def test_empty_phrase_list_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
            categories:
              affirm:
                match_type: candidate
                phrases: []
            """,
        )
        with pytest.raises(ValueError, match="non-empty"):
            ControlPhraseLexicon.load(path)

    def test_missing_phrases_key_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
            categories:
              affirm:
                match_type: candidate
            """,
        )
        with pytest.raises(ValueError, match="non-empty"):
            ControlPhraseLexicon.load(path)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ControlPhraseLexicon.load(tmp_path / "does_not_exist.yaml")

    def test_duplicate_yaml_category_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
            categories:
              affirm:
                match_type: candidate
                phrases: ["yes"]
              affirm:
                match_type: candidate
                phrases: ["okay"]
            """,
        )
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            ControlPhraseLexicon.load(path)


# ===========================================================================
# is_affirm
# ===========================================================================


class TestIsAffirm:
    @pytest.mark.parametrize("text", [
        "yes", "yeah", "yep", "yup", "correct", "ok", "okay",
        "sure", "sounds good", "go ahead", "confirm", "proceed",
        "continue", "place it", "checkout", "all good", "looks good",
    ])
    def test_core_affirm_phrases(self, text: str):
        assert DEFAULT_LEXICON.is_affirm(text), f"Expected is_affirm({text!r}) to be True"

    @pytest.mark.parametrize("text", [
        "well yes", "uh yeah please", "yes thanks", "okay thank you",
        "so sure", "um correct", "hmm proceed",
    ])
    def test_affirm_with_leading_trailing_fillers(self, text: str):
        assert DEFAULT_LEXICON.is_affirm(text)

    def test_empty_string_returns_false(self):
        assert not DEFAULT_LEXICON.is_affirm("")

    def test_unknown_phrase_returns_false(self):
        assert not DEFAULT_LEXICON.is_affirm("a large coffee please")

    def test_deny_phrase_is_not_affirm(self):
        assert not DEFAULT_LEXICON.is_affirm("no")
        assert not DEFAULT_LEXICON.is_affirm("nope")
        assert not DEFAULT_LEXICON.is_affirm("nah")

    def test_casing_normalized(self):
        assert DEFAULT_LEXICON.is_affirm("YES")
        assert DEFAULT_LEXICON.is_affirm("Yes")


# ===========================================================================
# is_deny
# ===========================================================================


class TestIsDeny:
    @pytest.mark.parametrize("text", [
        "no", "nope", "nah", "no thanks", "incorrect", "not correct",
        "thats wrong", "change it", "leave it", "none", "nothing",
        "wrong", "go back",
    ])
    def test_core_deny_phrases(self, text: str):
        assert DEFAULT_LEXICON.is_deny(text), f"Expected is_deny({text!r}) to be True"

    @pytest.mark.parametrize("text", [
        "well no", "uh nope please", "no thanks",
    ])
    def test_deny_with_fillers(self, text: str):
        assert DEFAULT_LEXICON.is_deny(text)

    def test_empty_string_returns_false(self):
        assert not DEFAULT_LEXICON.is_deny("")

    def test_unknown_phrase_returns_false(self):
        assert not DEFAULT_LEXICON.is_deny("a large coffee please")

    def test_affirm_phrase_is_not_deny(self):
        assert not DEFAULT_LEXICON.is_deny("yes")
        assert not DEFAULT_LEXICON.is_deny("okay")
        assert not DEFAULT_LEXICON.is_deny("sure")

    def test_casing_normalized(self):
        assert DEFAULT_LEXICON.is_deny("NO")
        assert DEFAULT_LEXICON.is_deny("Nope")


# ===========================================================================
# Mutual exclusivity and edge cases
# ===========================================================================


class TestEdgeCases:
    def test_affirm_and_deny_mutually_exclusive_for_clear_inputs(self):
        assert DEFAULT_LEXICON.is_affirm("yes")
        assert not DEFAULT_LEXICON.is_deny("yes")

        assert DEFAULT_LEXICON.is_deny("no")
        assert not DEFAULT_LEXICON.is_affirm("no")

    def test_empty_input_returns_false_for_both(self):
        assert not DEFAULT_LEXICON.is_affirm("")
        assert not DEFAULT_LEXICON.is_deny("")

    def test_whitespace_only_returns_false(self):
        assert not DEFAULT_LEXICON.is_affirm("   ")
        assert not DEFAULT_LEXICON.is_deny("   ")

    def test_get_phrases_returns_frozen(self):
        phrases = DEFAULT_LEXICON.get_phrases("affirm")
        assert isinstance(phrases, frozenset)

    def test_done_phrases_non_empty(self):
        assert len(DEFAULT_LEXICON.get_phrases("done")) > 0

    def test_cancel_phrases_non_empty(self):
        assert len(DEFAULT_LEXICON.get_phrases("cancel")) > 0

    def test_skip_phrases_non_empty(self):
        assert len(DEFAULT_LEXICON.get_phrases("skip")) > 0

    def test_stay_on_call_substring_match(self):
        phrases = DEFAULT_LEXICON.get_substring_phrases("stay_on_call")
        assert "stay on the line" in phrases

    def test_after_call_substring_match(self):
        phrases = DEFAULT_LEXICON.get_substring_phrases("after_call")
        assert "later" in phrases

    def test_cannot_open_link_substring_match(self):
        phrases = DEFAULT_LEXICON.get_substring_phrases("cannot_open_link")
        assert "cannot open the link" in phrases


# ===========================================================================
# Structural assertion: no inline frozensets in control_intent_resolver
# ===========================================================================


class TestNoInlinePhrasesets:
    """Guard against re-introduction of inline phrase frozensets."""

    def test_control_intent_resolver_phrase_vars_loaded_from_lexicon(self):
        """Phrase frozenset variables must be assigned from _LEXICON, not literals."""
        from pathlib import Path
        import re
        src = (
            Path(__file__).resolve().parents[2]
            / "app" / "state_machine" / "control_intent_resolver.py"
        ).read_text(encoding="utf-8")
        phrase_var_names = (
            "_AFFIRM_PHRASES", "_DENY_PHRASES", "_DONE_PHRASES",
            "_OPTIONS_PHRASES", "_CANCEL_PHRASES", "_META_CLARIFY_PHRASES",
            "_SKIP_PHRASES",
        )
        for var in phrase_var_names:
            # Each should be assigned from _LEXICON, not frozenset({...})
            inline = re.findall(
                rf"{re.escape(var)}\s*[:=][^=].*?frozenset\s*\(\s*\{{", src
            )
            assert not inline, (
                f"{var} appears to be assigned an inline frozenset literal"
            )

    def test_linguistic_rules_has_no_affirm_deny_word_sets(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[2]
            / "app" / "nlu" / "linguistic_rules.py"
        ).read_text(encoding="utf-8")
        assert "AFFIRM_WORDS" not in src, "AFFIRM_WORDS still defined in linguistic_rules"
        assert "DENY_WORDS" not in src, "DENY_WORDS still defined in linguistic_rules"

    def test_semantic_signals_does_not_import_affirm_deny_words(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[2]
            / "app" / "state_machine" / "semantic_signals.py"
        ).read_text(encoding="utf-8")
        assert "AFFIRM_WORDS" not in src
        assert "DENY_WORDS" not in src
