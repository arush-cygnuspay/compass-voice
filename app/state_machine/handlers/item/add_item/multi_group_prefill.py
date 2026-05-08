# app/state_machine/handlers/item/add_item/multi_group_prefill.py
"""
MultiGroupPrefillEngine
=======================

Unified, segment-scoped option prefill for a single PendingAddItem.

Why this exists
---------------
The historical prefill iterated each side/modifier group independently and
asked it to fish its own choices out of the segment text. That works for
single-item utterances, but in multi-item utterances slots have already
been segmented and labels are noisy:

  "chicken taco with coke steak and chicken
   and a chicken burger with american cheese ..."

Inside the chicken-taco segment we have phrases ("coke", "steak", "chicken")
whose slot labels may be ITEM, MODIFIER, or missing. A per-group resolver
that gates on slot label, or that only re-mines the text against its own
choice phrases, easily drops "coke" while still picking up "chicken".

This engine flips the model:

  1. Mine ALL candidate option phrases from the segment in one pass:
        - slot values (any label — labels are advisory only)
        - longest-first phrase mining against every group's choice phrases
        - conservative connector splits ("and", ",", "+", "/", "&")
  2. For each phrase, score against EVERY valid target on this item
     (variants + side choices + modifier choices) and bind to the highest
     scoring target, regardless of slot label.
  3. Apply per-group cap rules at the end.

The engine returns a structured PrefillResult that the handler applies to
ConversationContext. No business logic about prompts/states lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Iterable, Sequence

from app.nlu.modifier_instructions import (
    Action as _MIAction,
    Instruction as _MIInstruction,
    ModifierIntent,
    parse_phrase as _parse_modifier_phrase,
    priority as _modifier_priority,
)
from app.nlu.nlu_result import SlotValue
from app.nlu.order_scaffolding import ORDER_FILLER_TOKENS
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.handlers.item.add_item.option_matching import (
    score_scoped_choice,
)
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)
from app.state_machine.models.pending_item_models import (
    ModifierSelection,
    PendingAddItem,
    PendingModifierGroup,
    PendingSideGroup,
    PendingVariantChoice,
)
from app.utils.token_matcher import tokenize

logger = logging.getLogger(__name__)


# ─── Tunables ────────────────────────────────────────────────────────────────
AUTO_ACCEPT_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.60
MIN_CONFIRM_GAP = 0.08

# Connectors used both for splitting compound phrases AND as bridge tokens
# in the longest-first phrase miner so that "coke and steak" still yields
# both "coke" and "steak".
_CONNECTOR_RE = re.compile(r"\s*(?:,| and | & |\+|/| or | plus | also )\s*", re.IGNORECASE)
_BRIDGE_TOKENS: frozenset[str] = frozenset(
    {
        "a", "an", "the",
        "and", "plus", "also", "or",
        "with", "of",
        "extra", "more", "double",
        "less", "light",
        "no", "without", "hold", "remove",
        "on", "side",
        "small", "medium", "large", "regular", "mini", "xl",
    }
)

# Pure structural fillers that should never become standalone candidates.
# Derived from the shared ORDER_FILLER_TOKENS set plus a few connector/bridge
# tokens that are specific to the phrase-mining context.
_IGNORED_LEADING_WORDS: frozenset[str] = ORDER_FILLER_TOKENS | frozenset(
    {
        "id", "ill", "im",
        "with", "plus", "also",
        "also", "bring", "make",
        "um", "uh",
    }
)

# Modifier instruction prefixes/suffixes are owned by app.nlu.modifier_instructions.
# Re-exported here for back-compat with any external import; new code should
# import from the canonical module.
from app.nlu.modifier_instructions import (
    REMOVE_PREFIXES as _REMOVE_PREFIXES,
    EXTRA_PREFIXES as _EXTRA_PREFIXES,
    LESS_PREFIXES as _LESS_PREFIXES,
    ON_SIDE_SUFFIXES as _ON_SIDE_SUFFIXES,
)


# ─── Public types ────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CandidatePhrase:
    text: str          # cleaned, normalized phrase (no instruction prefix)
    raw: str           # original phrase as mined (with prefix)
    action: str        # "add" | "remove"
    instruction: str | None  # "extra" | "less" | "on_side" | None
    source: str        # "slot" | "phrase_mine" | "split"


@dataclass(frozen=True, slots=True)
class GroupBinding:
    """One resolved binding from a candidate phrase to a target on the item."""
    target_kind: str       # "variant" | "side" | "modifier"
    group_id: str | None   # None for variants
    group_name: str
    option_id: str         # variant_id | side_item_id | modifier_id
    option_name: str
    confidence: float
    candidate: CandidatePhrase


@dataclass(slots=True)
class PrefillResult:
    variant_id: str | None = None
    variant_name: str | None = None
    side_selections: dict[str, list[str]] = field(default_factory=dict)
    modifier_selections: dict[str, list[ModifierSelection]] = field(default_factory=dict)
    # Per-group feedback for the user-facing acknowledgement / error message.
    feedback: list[dict] = field(default_factory=list)
    # Phrases the user said that didn't bind to anything on this item.
    unresolved_phrases: list[str] = field(default_factory=list)
    # Structured debug snapshot for logging.
    debug: dict = field(default_factory=dict)


# ─── Engine ──────────────────────────────────────────────────────────────────
class MultiGroupPrefillEngine:
    """Resolve segment phrases against every group on a PendingAddItem."""

    def prefill(
        self,
        *,
        pending: PendingAddItem,
        segment_text: str,
        slots: Sequence[SlotValue] = (),
    ) -> PrefillResult:
        result = PrefillResult()

        # 1. Strip item name from the segment so its own tokens don't
        #    contaminate option matching.
        text_after_item = self._strip_item_name(
            segment_text,
            pending.item_name,
            getattr(pending, "item_voice_labels", ()),
        )
        result.debug["segment_text"] = segment_text
        result.debug["segment_text_after_item"] = text_after_item

        # 2. Mine candidate phrases from text + slots.
        candidates = self._build_candidate_phrases(
            text=text_after_item,
            slots=slots,
            pending=pending,
        )
        result.debug["candidate_phrases"] = [c.text for c in candidates]

        # 3. Bind each candidate to its best-scoring target across all groups.
        all_bindings: list[GroupBinding] = []
        unresolved: list[str] = []
        for cand in candidates:
            binding = self._best_binding_across_groups(cand, pending)
            if binding is None:
                # Use the raw phrase (with action prefix preserved, e.g.
                # "no sauce") so the user-facing feedback reads naturally.
                phrase_for_feedback = cand.raw or cand.text
                if phrase_for_feedback:
                    unresolved.append(phrase_for_feedback)
                continue
            all_bindings.append(binding)

        # 4. Apply per-group caps and build the final result.
        self._apply_bindings(all_bindings, pending, result)
        result.unresolved_phrases = self._dedupe_unresolved(
            unresolved,
            bindings=all_bindings,
            pending=pending,
        )

        result.debug["resolved_group_values"] = self._resolved_group_values_debug(
            result, pending
        )
        result.debug["bindings"] = [
            {
                "phrase": b.candidate.text,
                "target_kind": b.target_kind,
                "group_name": b.group_name,
                "option_name": b.option_name,
                "confidence": round(b.confidence, 3),
                "source": b.candidate.source,
            }
            for b in all_bindings
        ]
        return result

    # ── Step 1: item-name strip ────────────────────────────────────────────
    @staticmethod
    def _strip_item_name(
        segment_text: str,
        item_name: str,
        item_voice_labels: tuple[str, ...] = (),
    ) -> str:
        """Strip the item name (or a matching voice-label alias) from segment_text.

        Tries the canonical name first, then each voice label in order.
        Returns the segment with the first matching label removed, or the
        full normalized text if nothing matches.
        """
        normalized_text = normalize_text(segment_text or "")
        if not normalized_text:
            return normalized_text
        normalized_item = normalize_text(item_name or "")

        def _try_strip(label: str) -> str | None:
            """Attempt to remove *label* from *normalized_text*.

            Returns the stripped result if the label was found, or None.
            """
            if not label:
                return None
            if normalized_text.startswith(label):
                tail = normalized_text[len(label):].strip()
                return tail or ""
            try:
                pattern = re.compile(rf"\b{re.escape(label)}\b")
            except re.error:
                return None
            result = pattern.sub(" ", normalized_text, count=1).strip()
            return result if result != normalized_text else None

        # Try canonical name first.
        if normalized_item:
            stripped = _try_strip(normalized_item)
            if stripped is not None:
                return stripped

        # Try voice labels (skip duplicates of the canonical name).
        for label in item_voice_labels:
            norm_label = normalize_text(label)
            if not norm_label or norm_label == normalized_item:
                continue
            stripped = _try_strip(norm_label)
            if stripped is not None:
                return stripped

        return normalized_text

    # ── Step 2: candidate phrase mining ────────────────────────────────────
    def _build_candidate_phrases(
        self,
        *,
        text: str,
        slots: Sequence[SlotValue],
        pending: PendingAddItem,
    ) -> list[CandidatePhrase]:
        # Index by target text so we can UPGRADE an earlier-emitted bare
        # candidate ("cheese") with a later instruction-bearing one
        # ("extra cheese", "no cheese"). REMOVE wins over EXTRA wins over
        # LESS wins over ON_SIDE wins over plain ADD — see
        # `modifier_instructions.priority`.
        target_to_index: dict[str, int] = {}
        phrases: list[CandidatePhrase] = []

        def _phrase_priority(phrase: CandidatePhrase) -> int:
            try:
                action = _MIAction(phrase.action) if phrase.action else _MIAction.ADD
            except ValueError:
                action = _MIAction.ADD
            try:
                inst = (
                    _MIInstruction(phrase.instruction)
                    if phrase.instruction
                    else _MIInstruction.NONE
                )
            except ValueError:
                inst = _MIInstruction.NONE
            return _modifier_priority(
                ModifierIntent(
                    action=action,
                    instruction=inst,
                    target=phrase.text,
                    raw=phrase.raw,
                )
            )

        def add(raw: str, source: str) -> None:
            cleaned = self._clean_phrase(raw)
            if cleaned is None:
                return
            target_text = cleaned["target"]
            if not target_text:
                return
            new_phrase = CandidatePhrase(
                text=target_text,
                raw=raw,
                action=cleaned["action"],
                instruction=cleaned["instruction"],
                source=source,
            )
            existing_idx = target_to_index.get(target_text)
            if existing_idx is None:
                target_to_index[target_text] = len(phrases)
                phrases.append(new_phrase)
                return
            # Same target seen before — keep the higher-priority instruction
            # so "extra cheese" beats a prior bare "cheese" slot mention,
            # and a later "no cheese" beats an "extra cheese" emission.
            if _phrase_priority(new_phrase) > _phrase_priority(phrases[existing_idx]):
                phrases[existing_idx] = new_phrase

        # 2a. Slot values — labels are advisory only; we accept them all
        #     except the slot that anchors *this* item.
        item_name_normalized = normalize_text(pending.item_name)
        for slot in slots or ():
            slot_value = getattr(slot, "value", None)
            if not isinstance(slot_value, str) or not slot_value.strip():
                continue
            normalized_value = normalize_text(slot_value)
            if not normalized_value or normalized_value == item_name_normalized:
                continue
            add(normalized_value, source="slot")

        # 2b. Longest-first phrase mining against every option phrase on this item.
        all_phrases = self._collect_all_option_phrases(pending)
        for phrase in self._mine_phrases(text=text, phrases=all_phrases):
            add(phrase, source="phrase_mine")

        # 2c. Conservative connector splits as a fallback so words like
        #     "american cheese red onions and fresh mushrooms" still produce
        #     ("american cheese", "red onions", "fresh mushrooms"). The
        #     phrase miner above will already have caught these when they
        #     match a known group label, but the split is the safety net.
        for chunk in _CONNECTOR_RE.split(text):
            chunk = chunk.strip()
            if not chunk:
                continue
            chunk = self._strip_leading_filler(chunk)
            if chunk:
                add(chunk, source="split")

        return phrases

    @staticmethod
    def _collect_all_option_phrases(pending: PendingAddItem) -> list[str]:
        """Every label/alias/voice form for every option on the item."""
        seen: set[str] = set()
        phrases: list[str] = []

        def add(value: str | None) -> None:
            normalized = normalize_text(value or "")
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            phrases.append(normalized)

        for variant in pending.item_variants:
            add(variant.normalized_name)
            add(variant.name)

        for group in pending.side_groups:
            for choice in group.choices:
                for label in (getattr(choice, "match_texts", ()) or (choice.normalized_name,)):
                    add(label)
                add(choice.name)

        for group in pending.modifier_groups:
            for choice in group.choices:
                for label in (getattr(choice, "match_texts", ()) or (choice.normalized_name,)):
                    add(label)
                add(choice.name)

        return phrases

    @staticmethod
    def _mine_phrases(*, text: str, phrases: list[str]) -> list[str]:
        """
        Walk `text` token by token and emit the longest known phrase that
        matches at each cursor position. Bridge tokens ("with", "and",
        "extra", etc.) advance the allowed-start set so consecutive phrases
        can be mined without colliding.
        """
        if not text or not phrases:
            return []

        phrase_map: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        for phrase in phrases:
            tokens = tuple(token for token in phrase.split() if token)
            if not tokens:
                continue
            phrase_map.setdefault(tokens[0], []).append((tokens, phrase))
        for bucket in phrase_map.values():
            bucket.sort(key=lambda item: len(item[0]), reverse=True)

        text_tokens = [t for t in text.split() if t]
        emitted: list[str] = []
        seen: set[str] = set()
        allowed_starts: set[int] = {0}
        idx = 0
        while idx < len(text_tokens):
            matched = False
            if idx in allowed_starts:
                for phrase_tokens, original in phrase_map.get(text_tokens[idx], ()):
                    end = idx + len(phrase_tokens)
                    if tuple(text_tokens[idx:end]) != phrase_tokens:
                        continue
                    if original not in seen:
                        seen.add(original)
                        emitted.append(original)
                    allowed_starts.add(end)
                    idx = end
                    matched = True
                    break

            if matched:
                continue

            if text_tokens[idx] in _BRIDGE_TOKENS:
                allowed_starts.add(idx + 1)
            idx += 1

        return emitted

    @staticmethod
    def _strip_leading_filler(text: str) -> str:
        tokens = text.split()
        while tokens and tokens[0] in _IGNORED_LEADING_WORDS:
            tokens.pop(0)
        return " ".join(tokens).strip()

    @classmethod
    def _clean_phrase(cls, raw: str) -> dict | None:
        """
        Parse a raw candidate into {action, instruction, target}.

        Delegates instruction parsing (no/extra/less/on_side/add) to the
        canonical module ``app.nlu.modifier_instructions`` so that every
        resolver and response site uses the same lexicon and the same
        precedence rules.

        Returns None for empty / pure-filler input.
        """
        intent = _parse_modifier_phrase(raw)
        if intent is None:
            return None

        target = cls._strip_leading_filler(intent.target)
        if not target or all(t in _IGNORED_LEADING_WORDS for t in target.split()):
            return None

        return {
            "action": intent.action.value,
            "instruction": intent.instruction.value or None,
            "target": target,
        }

    # ── Step 3: best binding across all groups ─────────────────────────────
    def _best_binding_across_groups(
        self,
        cand: CandidatePhrase,
        pending: PendingAddItem,
    ) -> GroupBinding | None:
        """
        Score the candidate against every variant/side/modifier choice on
        the item and return the highest-scoring binding above threshold.

        Cross-group ambiguity guard: when two different groups both match
        with similar scores we accept only when the gap is wide enough,
        which prevents "chicken" from being silently bound to a side
        group when it's actually a modifier (or vice versa).
        """
        best: GroupBinding | None = None
        second_score = 0.0

        # Variants
        if pending.item_variants:
            scored = self._score_variants(cand.text, pending.item_variants)
            if scored is not None:
                variant, score = scored
                best = GroupBinding(
                    target_kind="variant",
                    group_id=None,
                    group_name="Size",
                    option_id=variant.variant_id,
                    option_name=variant.name,
                    confidence=score,
                    candidate=cand,
                )

        # Sides
        for group in pending.side_groups:
            scored = self._score_group_choices(
                cand.text, group, attr="item_id", name_attr="name"
            )
            if scored is None:
                continue
            choice, score = scored
            if best is None or score > best.confidence:
                if best is not None:
                    second_score = max(second_score, best.confidence)
                best = GroupBinding(
                    target_kind="side",
                    group_id=group.group_id,
                    group_name=group.name,
                    option_id=choice.item_id,
                    option_name=choice.name,
                    confidence=score,
                    candidate=cand,
                )
            else:
                second_score = max(second_score, score)

        # Modifiers
        for group in pending.modifier_groups:
            scored = self._score_group_choices(
                cand.text, group, attr="modifier_id", name_attr="name"
            )
            if scored is None:
                continue
            choice, score = scored
            if best is None or score > best.confidence:
                if best is not None:
                    second_score = max(second_score, best.confidence)
                best = GroupBinding(
                    target_kind="modifier",
                    group_id=group.group_id,
                    group_name=group.name,
                    option_id=choice.modifier_id,
                    option_name=choice.name,
                    confidence=score,
                    candidate=cand,
                )
            else:
                second_score = max(second_score, score)

        if best is None:
            return None
        if best.confidence < CONFIRM_THRESHOLD:
            return None
        if (
            best.confidence < AUTO_ACCEPT_THRESHOLD
            and (best.confidence - second_score) < MIN_CONFIRM_GAP
        ):
            return None
        return best

    @staticmethod
    def _score_variants(
        target: str, variants: Iterable[PendingVariantChoice]
    ) -> tuple[PendingVariantChoice, float] | None:
        best = None
        best_score = 0.0
        for variant in variants:
            score = score_scoped_choice(
                target, variant.normalized_name, reject_candidate_superset=True
            )
            if score > best_score:
                best_score = score
                best = variant
        if best is None or best_score < CONFIRM_THRESHOLD:
            return None
        return best, best_score

    @staticmethod
    def _score_group_choices(
        target: str,
        group: PendingSideGroup | PendingModifierGroup,
        *,
        attr: str,
        name_attr: str,
    ) -> tuple[object, float] | None:
        best = None
        best_score = 0.0
        for choice in group.choices:
            labels = getattr(choice, "match_texts", ()) or (choice.normalized_name,)
            score = max(
                (
                    score_scoped_choice(target, label, reject_candidate_superset=True)
                    for label in labels
                    if label
                ),
                default=0.0,
            )
            if score > best_score:
                best_score = score
                best = choice
        if best is None or best_score < CONFIRM_THRESHOLD:
            return None
        return best, best_score

    # ── Step 4: apply bindings + caps ─────────────────────────────────────
    def _apply_bindings(
        self,
        bindings: list[GroupBinding],
        pending: PendingAddItem,
        result: PrefillResult,
    ) -> None:
        # Group bindings by their target group/variant.
        # Bindings are already ordered by candidate emission order so the
        # spoken order is preserved.
        per_group: dict[tuple[str, str | None], list[GroupBinding]] = {}
        for binding in bindings:
            key = (binding.target_kind, binding.group_id)
            per_group.setdefault(key, []).append(binding)

        # Variants: take the first (highest-scored, first-spoken) one only.
        variant_bindings = per_group.pop(("variant", None), [])
        if variant_bindings:
            top = variant_bindings[0]
            result.variant_id = top.option_id
            result.variant_name = top.option_name

        # Sides
        for group in pending.side_groups:
            key = ("side", group.group_id)
            group_bindings = self._dedupe_by_option(per_group.pop(key, []))
            if not group_bindings:
                continue
            min_sel, max_sel = effective_group_selector_bounds(group)
            cap = max_sel if max_sel > 0 else len(group_bindings)
            accepted = group_bindings[:cap]
            dropped = group_bindings[cap:]
            over_max = bool(dropped)
            if accepted and not over_max:
                result.side_selections[group.group_id] = [
                    b.option_id for b in accepted
                ]
            result.feedback.append(
                {
                    "kind": "side",
                    "group_id": group.group_id,
                    "group_name": group.name,
                    "accepted_names": [] if over_max else [b.option_name for b in accepted],
                    "requested_names": [b.option_name for b in group_bindings],
                    "dropped_names": [b.option_name for b in dropped],
                    "min_selector": min_sel,
                    "max_selector": max_sel,
                    "over_max": over_max,
                }
            )

        # Modifiers
        for group in pending.modifier_groups:
            key = ("modifier", group.group_id)
            group_bindings = self._dedupe_by_option(per_group.pop(key, []))
            if not group_bindings:
                continue
            min_sel, max_sel = effective_group_selector_bounds(group)
            cap = max_sel if max_sel > 0 else len(group_bindings)
            accepted = group_bindings[:cap]
            dropped = group_bindings[cap:]
            over_max = bool(dropped)
            if accepted and not over_max:
                result.modifier_selections[group.group_id] = [
                    ModifierSelection(
                        modifier_id=b.option_id,
                        name=b.option_name,
                        action=b.candidate.action,
                        instruction=b.candidate.instruction,
                    )
                    for b in accepted
                ]
            result.feedback.append(
                {
                    "kind": "modifier",
                    "group_id": group.group_id,
                    "group_name": group.name,
                    "accepted_names": [] if over_max else [b.option_name for b in accepted],
                    "requested_names": [b.option_name for b in group_bindings],
                    "dropped_names": [b.option_name for b in dropped],
                    "min_selector": min_sel,
                    "max_selector": max_sel,
                    "over_max": over_max,
                }
            )

    @staticmethod
    def _dedupe_by_option(bindings: list[GroupBinding]) -> list[GroupBinding]:
        seen: set[str] = set()
        out: list[GroupBinding] = []
        for binding in bindings:
            if binding.option_id in seen:
                continue
            seen.add(binding.option_id)
            out.append(binding)
        return out

    # ── Step 5: unresolved cleanup + debug ────────────────────────────────
    @staticmethod
    def _dedupe_unresolved(
        unresolved: list[str],
        *,
        bindings: list[GroupBinding],
        pending: PendingAddItem,
    ) -> list[str]:
        if not unresolved:
            return []

        bound_tokens: set[str] = set()
        for binding in bindings:
            bound_tokens.update(tokenize(normalize_text(binding.option_name)))
            bound_tokens.update(tokenize(binding.candidate.text))
        bound_tokens.update(tokenize(normalize_text(getattr(pending, "item_name", "") or "")))

        # Build item label norms (canonical name + aliases + voice labels) for
        # exact-match suppression.  Both spaced and compact forms are indexed so
        # that "cheeseburger" (compact ASR form) is suppressed when the item
        # name is "Cheese Burger" and "cheeseburger" is a voice label.
        item_label_norms: set[str] = set()
        _item_norm = normalize_text(getattr(pending, "item_name", "") or "")
        if _item_norm:
            item_label_norms.add(_item_norm)
            item_label_norms.add(_item_norm.replace(" ", ""))
        for alias in (getattr(pending, "item_aliases", ()) or ()):
            norm = normalize_text(alias)
            if norm:
                item_label_norms.add(norm)
                item_label_norms.add(norm.replace(" ", ""))
        for vl in (getattr(pending, "item_voice_labels", ()) or ()):
            norm = normalize_text(vl)
            if norm:
                item_label_norms.add(norm)
                item_label_norms.add(norm.replace(" ", ""))

        cleaned: list[str] = []
        seen: set[str] = set()
        for value in unresolved:
            normalized = normalize_text(value).strip()
            if not normalized or normalized in seen:
                continue
            # Suppress if it exactly matches any item label form.
            if normalized in item_label_norms:
                continue
            compact = normalized.replace(" ", "")
            if compact and compact in item_label_norms:
                continue
            value_tokens = set(tokenize(normalized))
            if value_tokens and value_tokens.issubset(bound_tokens):
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned

    @staticmethod
    def _resolved_group_values_debug(
        result: PrefillResult, pending: PendingAddItem
    ) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if result.variant_id and result.variant_name:
            out["Size"] = [result.variant_name]
        for group in pending.side_groups:
            ids = result.side_selections.get(group.group_id) or []
            if not ids:
                continue
            out[group.name] = [
                group.choices_by_item_id[i].name
                for i in ids
                if i in group.choices_by_item_id
            ]
        for group in pending.modifier_groups:
            sels = result.modifier_selections.get(group.group_id) or []
            if not sels:
                continue
            out[group.name] = [s.name for s in sels]
        return out
