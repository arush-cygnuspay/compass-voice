from __future__ import annotations

from dataclasses import dataclass
import re

from app.nlu.modifier_instructions import (
    Action as _MIAction,
    Instruction as _MIInstruction,
    ModifierIntent,
    parse_phrase as _parse_modifier_phrase,
    priority as _modifier_intent_priority,
)
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.option_matching import (
    OptionCandidate,
    build_match_debug_payload,
    build_scoped_phrase_candidates,
    build_slot_first_option_candidates,
    extract_slot_candidate_texts,
    score_scoped_choice,
    strip_common_option_fillers,
)
from app.state_machine.models.pending_item_models import ModifierSelection
from app.utils.candidate_texts import build_candidate_texts_normalized
from app.utils.token_matcher import (
    tokenize,
)

AUTO_ACCEPT_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.60
MIN_CONFIRM_GAP = 0.08

# Re-exported so any external import (tests, plugins) keeps working.
# New code should import from app.nlu.modifier_instructions instead.
from app.nlu.modifier_instructions import (
    REMOVE_PREFIXES,
    EXTRA_PREFIXES as _EXTRA_PREFIXES,
    LESS_PREFIXES as _LESS_PREFIXES,
    ON_SIDE_SUFFIXES as ON_SIDE_SUFFIXES_FULL,
)

# Legacy single-word sets used by the greedy phrase-context scanner that
# inspects the immediate previous token (not a prefix string). Keep them as
# token sets, but derive them from the canonical lexicon so an alias added
# in modifier_instructions.py automatically lights up here too.
EXTRA_WORDS = {p.strip().split()[0] for p in _EXTRA_PREFIXES if " " not in p.strip()} | {"more"}
LESS_WORDS = {p.strip().split()[0] for p in _LESS_PREFIXES if " " not in p.strip()}
# The "on the side" suffix in this file historically had no leading space,
# so map back to the unprefixed forms for the existing call sites.
ON_SIDE_SUFFIXES = tuple(s.strip() for s in ON_SIDE_SUFFIXES_FULL)
GREEDY_CONTEXT_WORDS = {
    "with",
    "and",
    "plus",
    "also",
    "extra",
    "more",
    "double",
    "no",
    "without",
    "light",
    "less",
    "on",
    "the",
    "side",
}

# IMPORTANT: generic words that should NEVER become modifiers
GENERIC_MODIFIER_WORDS = {
    "toppings",
    "stuff",
    "things",
}


@dataclass(frozen=True, slots=True)
class ModifierGroupMatch:
    selections: list[ModifierSelection]
    unmatched_values: list[str]
    match_debug: dict[str, object] | None = None


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        value = (value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_modifier_slot_values_normalized(context) -> list[str]:
    return extract_slot_candidate_texts(
        slots=context.last_slots or (),
        allowed_slot_labels={"MODIFIER", "VARIANT"},
        cleaner=_normalize_modifier_candidate,
    )


def build_modifier_option_candidates(context, normalized_user_text: str) -> list[OptionCandidate]:
    return build_slot_first_option_candidates(
        raw_utterance=normalized_user_text,
        slots=context.last_slots or (),
        allowed_slot_labels={"MODIFIER", "VARIANT"},
        cleaner=_normalize_modifier_candidate,
        allow_split=True,
    )


def _normalize_modifier_candidate(text: str) -> str:
    return strip_common_option_fillers(text)


class ModifierGroupResolver:
    def resolve(
        self,
        *,
        group,
        normalized_user_text: str,
        normalized_slot_values: list[str] | None = None,
        option_candidates: list[OptionCandidate] | None = None,
        already_selected_ids: list[str] | None = None,
        ignored_values: list[str] | None = None,
        known_choice_phrases: list[str] | None = None,
    ) -> ModifierGroupMatch:
        already_selected_ids = already_selected_ids or []
        ignored_values = dedupe_keep_order(list(ignored_values or []))
        ignored_token_set: set[str] = set()
        for value in ignored_values:
            ignored_token_set.update(tokenize(value))

        candidates = option_candidates or self._build_candidates(
            normalized_user_text,
            normalized_slot_values or [],
            ignored_values=ignored_values,
        )
        stripped_user_text = self._strip_ignored_values(
            normalized_user_text,
            ignored_values,
        )
        candidates = self._apply_ignored_values_to_candidates(
            candidates,
            ignored_values=ignored_values,
        )
        candidates = self._augment_candidates_with_group_phrases(
            candidates=candidates,
            group=group,
            normalized_user_text=stripped_user_text,
        )

        selections_by_id: dict[str, ModifierSelection] = {}
        unmatched: list[str] = []
        matched_candidate_texts: list[str] = []
        slot_candidates_present = any(
            candidate.source in {"slot_value", "slot_raw"}
            for candidate in candidates
        )
        debug_candidate_text: str | None = None
        debug_match_source: str | None = None
        debug_matched_option: str | None = None
        debug_match_score: float | None = None

        for candidate in candidates:
            parsed = self._parse(candidate.text)

            if not parsed["target"]:
                continue

            if parsed["target"] in GENERIC_MODIFIER_WORDS:
                continue

            scored = self._match(group, parsed["target"], action=parsed["action"])
            if not scored:
                if debug_candidate_text is None:
                    debug_candidate_text = candidate.text
                    debug_match_source = candidate.source
                if candidate.source != "raw_utterance" or not slot_candidates_present:
                    unmatched.append(candidate.text)
                continue

            mod_id, name, confidence = scored
            if debug_candidate_text is None:
                debug_candidate_text = candidate.text
                debug_match_source = candidate.source
                debug_matched_option = name
                debug_match_score = confidence
            if confidence < AUTO_ACCEPT_THRESHOLD:
                if candidate.source != "raw_utterance" or not slot_candidates_present:
                    unmatched.append(candidate.text)
                continue

            if mod_id in already_selected_ids:
                continue

            new_selection = ModifierSelection(
                modifier_id=mod_id,
                name=name,
                action=parsed["action"],
                instruction=parsed["instruction"],
            )

            previous = selections_by_id.get(mod_id)
            if previous is None or self._priority(new_selection) > self._priority(previous):
                selections_by_id[mod_id] = new_selection
                matched_candidate_texts.append(parsed["target"])
                if debug_match_score is None or confidence >= debug_match_score:
                    debug_candidate_text = candidate.text
                    debug_match_source = candidate.source
                    debug_matched_option = name
                    debug_match_score = confidence

        self._greedy_scan_for_embedded_modifiers(
            group=group,
            text=normalized_user_text,
            selections_by_id=selections_by_id,
            already_selected_ids=already_selected_ids,
            ignored_tokens=ignored_token_set,
            matched_candidate_texts=matched_candidate_texts,
            known_choice_phrases=known_choice_phrases or [],
        )
        for target in self._extract_remove_targets(normalized_user_text):
            scored = self._match_unique_remove_family(group, target)
            if not scored:
                continue
            modifier_id, name, _ = scored
            if modifier_id in selections_by_id or modifier_id in already_selected_ids:
                continue
            selections_by_id[modifier_id] = ModifierSelection(
                modifier_id=modifier_id,
                name=name,
                action="remove",
                instruction=None,
            )
            matched_candidate_texts.append(target)

        ordered_ids: list[str] = []
        for candidate in candidates:
            parsed = self._parse(candidate.text)
            scored = self._match(group, parsed["target"], action=parsed["action"])
            if not scored:
                continue
            modifier_id = scored[0]
            if modifier_id in selections_by_id and modifier_id not in ordered_ids:
                ordered_ids.append(modifier_id)

        for modifier_id in selections_by_id:
            if modifier_id not in ordered_ids:
                ordered_ids.append(modifier_id)

        matched_tokens: set[str] = set()
        for selection in selections_by_id.values():
            matched_tokens.update(tokenize(normalize_text(selection.name)))
        for candidate_text in matched_candidate_texts:
            matched_tokens.update(tokenize(candidate_text))

        first_pass: list[str] = []
        for value in unmatched:
            value_tokens = set(tokenize(value))
            if value_tokens and not value_tokens.issubset(matched_tokens | ignored_token_set):
                first_pass.append(value)

        cleaned_unmatched: list[str] = []
        for value in first_pass:
            value_tokens = set(tokenize(value))
            novel_tokens = value_tokens - matched_tokens - ignored_token_set
            other_unmatched_tokens: set[str] = set()
            for other in first_pass:
                if other != value and len(other) < len(value):
                    other_unmatched_tokens.update(set(tokenize(other)) - ignored_token_set)
            if novel_tokens and novel_tokens.issubset(other_unmatched_tokens):
                continue
            cleaned_unmatched.append(value)

        return ModifierGroupMatch(
            selections=[selections_by_id[modifier_id] for modifier_id in ordered_ids],
            unmatched_values=dedupe_keep_order(cleaned_unmatched),
            match_debug=build_match_debug_payload(
                raw_utterance=normalized_user_text,
                candidates=candidates,
                selected_candidate=debug_candidate_text,
                matched_option=debug_matched_option,
                match_source=debug_match_source,
                match_score=debug_match_score,
            ),
        )

    def _augment_candidates_with_group_phrases(
        self,
        *,
        candidates: list[OptionCandidate],
        group,
        normalized_user_text: str,
    ) -> list[OptionCandidate]:
        slot_first = [candidate for candidate in candidates if candidate.source != "raw_utterance"]
        raw_fallback = [candidate for candidate in candidates if candidate.source == "raw_utterance"]
        scoped_candidates = build_scoped_phrase_candidates(
            raw_utterance=normalized_user_text,
            phrases=self._group_choice_phrases(group),
            cleaner=_normalize_modifier_candidate,
            source="raw_utterance",
        )
        return dedupe_option_candidates(slot_first + scoped_candidates + raw_fallback)

    def _greedy_scan_for_embedded_modifiers(
        self,
        *,
        group,
        text: str,
        selections_by_id: dict[str, ModifierSelection],
        already_selected_ids: list[str],
        ignored_tokens: set[str] | None = None,
        matched_candidate_texts: list[str] | None = None,
        known_choice_phrases: list[str] | None = None,
    ) -> None:
        if not text:
            return

        ignored_tokens = ignored_tokens or set()
        matched_candidate_texts = matched_candidate_texts if matched_candidate_texts is not None else []
        known_choice_phrases = [phrase for phrase in (known_choice_phrases or []) if phrase]
        text_tokens = set(tokenize(text))
        if not text_tokens:
            return

        for choice in group.choices:
            modifier_id = choice.modifier_id
            if modifier_id in selections_by_id or modifier_id in already_selected_ids:
                continue

            labels = sorted(
                set(getattr(choice, "match_texts", ()) or (choice.normalized_name,)),
                key=len,
                reverse=True,
            )
            for label in labels:
                if not label:
                    continue

                selection = self._selection_from_phrase_context(
                    text=text,
                    choice=choice,
                    label=label,
                )
                if selection is None:
                    continue

                choice_tokens = set(tokenize(label))
                if not choice_tokens:
                    continue

                if ignored_tokens and choice_tokens.issubset(ignored_tokens):
                    continue

                if not choice_tokens.issubset(text_tokens):
                    continue

                if label not in text:
                    continue

                if not self._has_supported_phrase_context(
                    text,
                    label,
                    known_choice_phrases=known_choice_phrases,
                ):
                    continue

                selections_by_id[modifier_id] = selection
                matched_candidate_texts.append(label)
                break

    def _priority(self, selection: ModifierSelection) -> int:
        """Conflict-resolution priority for two parses against the same modifier.

        Routes through the canonical ``modifier_instructions.priority`` so the
        ranking stays consistent with the prefill engine.  REMOVE wins over
        any ADD; a specific instruction (extra > less > on_side) beats bare ADD.
        """
        try:
            action = _MIAction(selection.action) if selection.action else _MIAction.ADD
        except ValueError:
            action = _MIAction.ADD
        try:
            instruction = (
                _MIInstruction(selection.instruction)
                if selection.instruction
                else _MIInstruction.NONE
            )
        except ValueError:
            instruction = _MIInstruction.NONE
        return _modifier_intent_priority(
            ModifierIntent(
                action=action,
                instruction=instruction,
                target=selection.name,
                raw=selection.name,
            )
        )

    def _selection_from_phrase_context(self, *, text: str, choice, label: str) -> ModifierSelection | None:
        pattern = re.compile(rf"\b{re.escape(label)}\b")
        for match in pattern.finditer(text):
            prefix = text[:match.start()].rstrip()
            suffix = text[match.end():].lstrip()
            trailing_words = prefix.split()
            leading_words = suffix.split()

            if len(trailing_words) >= 2 and " ".join(trailing_words[-2:]) == "on the":
                return ModifierSelection(
                    modifier_id=choice.modifier_id,
                    name=choice.name,
                    action="add",
                    instruction="on_side",
                )

            if trailing_words:
                last_word = trailing_words[-1]
                if last_word in {"no", "without", "hold", "remove"}:
                    return ModifierSelection(
                        modifier_id=choice.modifier_id,
                        name=choice.name,
                        action="remove",
                        instruction=None,
                    )
                if last_word in EXTRA_WORDS:
                    return ModifierSelection(
                        modifier_id=choice.modifier_id,
                        name=choice.name,
                        action="add",
                        instruction="extra",
                    )
                if last_word in LESS_WORDS:
                    return ModifierSelection(
                        modifier_id=choice.modifier_id,
                        name=choice.name,
                        action="add",
                        instruction="less",
                    )

            if len(leading_words) >= 3 and " ".join(leading_words[:3]) == "on the side":
                return ModifierSelection(
                    modifier_id=choice.modifier_id,
                    name=choice.name,
                    action="add",
                    instruction="on_side",
                )

            return ModifierSelection(
                modifier_id=choice.modifier_id,
                name=choice.name,
                action="add",
                instruction=None,
            )

        return None

    def _build_candidates(
        self,
        text: str,
        slot_values: list[str],
        *,
        ignored_values: list[str] | None = None,
    ) -> list[OptionCandidate]:
        candidates: list[OptionCandidate] = []
        ignored_values = ignored_values or []
        cleaned_text = self._strip_ignored_values(text, ignored_values)

        for value in slot_values:
            if value and value not in ignored_values:
                normalized = _normalize_modifier_candidate(value)
                if normalized:
                    candidates.append(OptionCandidate(text=normalized, source="slot_value"))

        raw_values = build_candidate_texts_normalized(
            normalized_user_text=_normalize_modifier_candidate(cleaned_text),
            normalized_slot_values=[],
            allow_split=True,
        )
        if raw_values:
            raw_values = raw_values[1:] + raw_values[:1]

        for value in raw_values:
            if value:
                candidates.append(OptionCandidate(text=value, source="raw_utterance"))

        return dedupe_option_candidates(candidates)

    @staticmethod
    def _strip_ignored_values(text: str, ignored_values: list[str]) -> str:
        cleaned = (text or "").strip()
        if not cleaned or not ignored_values:
            return cleaned

        for value in sorted((v for v in ignored_values if v), key=len, reverse=True):
            cleaned = re.sub(rf"\b{re.escape(value)}\b", " ", cleaned)

        return re.sub(r"\s+", " ", cleaned).strip()

    def _apply_ignored_values_to_candidates(
        self,
        candidates: list[OptionCandidate],
        *,
        ignored_values: list[str] | None,
    ) -> list[OptionCandidate]:
        if not ignored_values:
            return dedupe_option_candidates(candidates)

        cleaned_candidates: list[OptionCandidate] = []
        for candidate in candidates:
            cleaned_text = self._strip_ignored_values(candidate.text, list(ignored_values))
            if not cleaned_text:
                continue
            cleaned_candidates.append(
                OptionCandidate(
                    text=cleaned_text,
                    source=candidate.source,
                    slot_label=candidate.slot_label,
                )
            )

        return dedupe_option_candidates(cleaned_candidates)

    def _parse(self, text: str):
        """Delegate to the canonical modifier-instruction parser.

        The historical inline parser supported only ``no/without/hold/remove``,
        ``extra/more/double``, ``less/light`` and the ``on the side`` suffix.
        It missed ``skip / drop / kill / easy on / go light on / half /
        double the / lots of / triple`` etc.  Routing through the canonical
        module gives this resolver the same coverage as the prefill engine
        and the response layer in one shot — and means future aliases land
        in exactly one file.
        """
        intent = _parse_modifier_phrase(text)
        if intent is None:
            return {"action": "add", "instruction": None, "target": ""}
        return {
            "action": intent.action.value,
            "instruction": intent.instruction.value or None,
            "target": intent.target,
        }

    @staticmethod
    def _group_choice_phrases(group) -> list[str]:
        phrases: list[str] = []
        seen: set[str] = set()
        for choice in group.choices:
            for label in getattr(choice, "match_texts", ()) or (choice.normalized_name,):
                normalized = normalize_text(label or "")
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                phrases.append(normalized)
        return phrases

    def _match(self, group, candidate, *, action: str = "add"):
        best = None
        best_confidence = 0.0
        second_confidence = 0.0

        for choice in group.choices:
            labels = getattr(choice, "match_texts", ()) or (choice.normalized_name,)
            confidence = max(
                (self._confidence(candidate, label) for label in labels if label),
                default=0.0,
            )
            if confidence > best_confidence:
                second_confidence = best_confidence
                best_confidence = confidence
                best = choice
            elif confidence > second_confidence:
                second_confidence = confidence

        if not best:
            if action == "remove":
                return self._match_unique_remove_family(group, candidate)
            return None

        if best_confidence < CONFIRM_THRESHOLD:
            return None

        if (
            best_confidence < AUTO_ACCEPT_THRESHOLD
            and (best_confidence - second_confidence) < MIN_CONFIRM_GAP
        ):
            return None

        return best.modifier_id, best.name, best_confidence

    def _match_unique_remove_family(self, group, candidate: str):
        candidate_tokens = set(tokenize(candidate))
        if not candidate_tokens:
            return None

        matches = []
        for choice in group.choices:
            choice_tokens = set(tokenize(choice.normalized_name))
            if candidate_tokens.issubset(choice_tokens):
                matches.append(choice)

        if len(matches) != 1:
            return None

        choice = matches[0]
        return choice.modifier_id, choice.name, AUTO_ACCEPT_THRESHOLD

    @staticmethod
    def _extract_remove_targets(text: str) -> list[str]:
        if not text:
            return []

        pattern = re.compile(
            r"\b(?:no|without|hold(?:\s+the)?|remove(?:\s+the)?)\s+([a-z][a-z\s]+?)(?=\b(?:and|plus|also|extra|more|double|less|light|with)\b|$|,)",
        )
        targets: list[str] = []
        for match in pattern.finditer(text):
            target = " ".join(match.group(1).split()).strip()
            if target:
                targets.append(target)
        return dedupe_keep_order(targets)

    def _confidence(self, candidate: str, choice_name: str) -> float:
        return score_scoped_choice(
            candidate,
            choice_name,
            reject_candidate_superset=True,
        )

    @staticmethod
    def _has_supported_phrase_context(
        text: str,
        phrase: str,
        *,
        known_choice_phrases: list[str] | None = None,
    ) -> bool:
        known_choice_phrases = [value for value in (known_choice_phrases or []) if value and value != phrase]
        pattern = re.compile(rf"\b{re.escape(phrase)}\b")
        for match in pattern.finditer(text):
            prefix = text[:match.start()].rstrip()
            suffix = text[match.end():].lstrip()

            previous_word = re.search(r"([a-z0-9]+)$", prefix)
            next_word = re.match(r"([a-z0-9]+)", suffix)

            prev_ok = previous_word is None or previous_word.group(1) in GREEDY_CONTEXT_WORDS
            next_ok = next_word is None or next_word.group(1) in GREEDY_CONTEXT_WORDS

            if not prev_ok and known_choice_phrases:
                prev_ok = any(prefix.endswith(other) for other in known_choice_phrases)

            if not next_ok and known_choice_phrases:
                next_ok = any(suffix.startswith(other) for other in known_choice_phrases)

            if not prev_ok or not next_ok:
                continue

            return True

        return False


def dedupe_option_candidates(values: list[OptionCandidate]) -> list[OptionCandidate]:
    seen: set[str] = set()
    result: list[OptionCandidate] = []
    for value in values:
        if not value.text or value.text in seen:
            continue
        seen.add(value.text)
        result.append(value)
    return result
