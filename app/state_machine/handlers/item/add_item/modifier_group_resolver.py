# app/state_machine/handlers/item/add_item/modifier_group_resolver.py
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.pending_item_models import ModifierSelection
from app.utils.candidate_texts import build_candidate_texts_normalized
from app.utils.token_matcher import is_controlled_partial_match, is_strong_token_match, tokenize

AUTO_ACCEPT_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.60
MIN_CONFIRM_GAP = 0.08

REMOVE_PREFIXES = ("no ", "without ")
EXTRA_WORDS = {"extra", "more", "double"}
LESS_WORDS = {"less", "light"}
ON_SIDE_SUFFIXES = ("on the side", "on side")

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


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for v in values:
        v = (v or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        result.append(v)
    return result


def extract_modifier_slot_values_normalized(context) -> list[str]:
    slots = context.last_slots or ()
    values = []
    seen = set()

    for slot in slots:
        if str(slot.name).upper() not in {"MODIFIER", "ITEM", "MENU_ITEM"}:
            continue

        val = slot.value
        if not isinstance(val, str):
            continue

        norm = normalize_text(val)
        if not norm or norm in seen:
            continue

        seen.add(norm)
        values.append(norm)

    return values


class ModifierGroupResolver:

    def resolve(
        self,
        *,
        group,
        normalized_user_text: str,
        normalized_slot_values: list[str],
        already_selected_ids: list[str] | None = None,
    ) -> ModifierGroupMatch:

        already_selected_ids = already_selected_ids or []

        candidates = self._build_candidates(
            normalized_user_text,
            normalized_slot_values,
        )

        selections_by_id = {}
        unmatched = []

        for candidate in candidates:
            parsed = self._parse(candidate)

            if not parsed["target"]:
                continue

            # 🚫 block generic words
            if parsed["target"] in GENERIC_MODIFIER_WORDS:
                continue

            scored = self._match(group, parsed["target"])
            if not scored:
                unmatched.append(candidate)
                continue

            mod_id, name, conf = scored
            if conf < AUTO_ACCEPT_THRESHOLD:
                unmatched.append(candidate)
                continue

            if mod_id in already_selected_ids:
                continue

            new_sel = ModifierSelection(
                modifier_id=mod_id,
                name=name,
                action=parsed["action"],
                instruction=parsed["instruction"],
            )

            prev = selections_by_id.get(mod_id)
            if prev is None or self._priority(new_sel) > self._priority(prev):
                selections_by_id[mod_id] = new_sel

        ordered_ids = []
        for candidate in candidates:
            parsed = self._parse(candidate)
            scored = self._match(group, parsed["target"])
            if not scored:
                continue
            mid = scored[0]
            if mid in selections_by_id and mid not in ordered_ids:
                ordered_ids.append(mid)

        return ModifierGroupMatch(
            selections=[selections_by_id[mid] for mid in ordered_ids],
            unmatched_values=dedupe_keep_order(unmatched),
        )

    # -------------------------

    def _priority(self, sel: ModifierSelection) -> int:
        if sel.action == "remove":
            return 4
        if sel.instruction == "extra":
            return 3
        if sel.instruction == "less":
            return 2
        if sel.instruction == "on_side":
            return 2
        return 1

    def _build_candidates(self, text, slot_values):
        candidates = []

        # 1️⃣ FULL TEXT FIRST (critical fix)
        if text:
            candidates.append(text.strip())

        # 2️⃣ SPLIT TEXT
        splits = build_candidate_texts_normalized(
            normalized_user_text=text,
            normalized_slot_values=[],
            allow_split=True,
        )
        candidates.extend(splits)

        # 3️⃣ SLOT VALUES LAST (lowest priority)
        candidates.extend(slot_values)

        return dedupe_keep_order(candidates)

    def _parse(self, text: str):
        text = (text or "").strip()

        for p in REMOVE_PREFIXES:
            if text.startswith(p):
                return {
                    "action": "remove",
                    "instruction": None,
                    "target": text[len(p):].strip(),
                }

        tokens = text.split()
        if not tokens:
            return {"action": "add", "instruction": None, "target": ""}

        first = tokens[0]
        rest = " ".join(tokens[1:]).strip()

        if first in EXTRA_WORDS and rest:
            return {"action": "add", "instruction": "extra", "target": rest}

        if first in LESS_WORDS and rest:
            return {"action": "add", "instruction": "less", "target": rest}

        for suffix in ON_SIDE_SUFFIXES:
            if text.endswith(f" {suffix}"):
                return {
                    "action": "add",
                    "instruction": "on_side",
                    "target": text[: -(len(suffix) + 1)].strip(),
                }

        return {"action": "add", "instruction": None, "target": text}

    def _match(self, group, candidate):
        best = None
        best_conf = 0
        second = 0

        for c in group.choices:
            conf = self._confidence(candidate, c.normalized_name)
            if conf > best_conf:
                second = best_conf
                best_conf = conf
                best = c
            elif conf > second:
                second = conf

        if not best:
            return None

        if best_conf < CONFIRM_THRESHOLD:
            return None

        if best_conf < AUTO_ACCEPT_THRESHOLD and (best_conf - second) < MIN_CONFIRM_GAP:
            return None

        return best.modifier_id, best.name, best_conf

    def _confidence(self, a, b):
        if a == b:
            return 1.0

        score = SequenceMatcher(None, a, b).ratio()

        if is_strong_token_match(a, b):
            score = max(score, 0.92)

        if is_controlled_partial_match(a, b):
            score = max(score, 0.82)

        return score
