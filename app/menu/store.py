from __future__ import annotations

import json
import re
from pathlib import Path

from app.menu.exceptions import MenuLoadError
from app.menu.models import (
    MenuItem,
    ModifierChoice,
    ModifierGroup,
    Pricing,
    PricingVariant,
    SideChoice,
    SideGroup,
)
from app.nlu.query_normalization.text_preprocessor import normalize_text


_PROMPT_NOUN_NOISE_TOKENS = {
    "modification",
    "modifications",
    "mod",
    "mods",
    "add-on",
    "add-ons",
    "add on",
    "add ons",
    "choice",
    "choices",
    "options",
    "selection",
    "group",
}


def _derive_prompt_noun(name: str) -> str:
    label = re.sub(r"^choose\s+(your\s+|a\s+|an\s+)?", "", (name or "").strip(), flags=re.IGNORECASE)
    tokens = [t for t in re.split(r"\s+", label.lower()) if t and t not in _PROMPT_NOUN_NOISE_TOKENS]
    return " ".join(tokens) or label.lower().strip()


class MenuStore:
    """
    Immutable, in-memory menu store.

    Responsibilities:
    - Load menu.json and entity_index.json
    - Parse raw JSON into domain models
    - Build cheap runtime indexes
    - Provide deterministic low-level lookup helpers

    Non-responsibilities:
    - Intent interpretation
    - Ambiguity resolution
    - Conversational routing
    - Winner selection logic
    """

    def __init__(self, menu_path: Path, entity_index_path: Path):
        self.menu_path = menu_path
        self.entity_index_path = entity_index_path

        self.items: dict[str, MenuItem] = {}
        self.categories: dict[str, dict] = {}
        self.entity_index: dict[str, list[dict]] = {}

        self._item_by_name: dict[str, MenuItem] = {}
        self._item_ids_by_alias: dict[str, list[str]] = {}
        self._item_ids_by_voice_label: dict[str, list[str]] = {}
        self._category_name_index: dict[str, dict] = {}

        self._discoverable_item_ids: set[str] = set()
        self._modifier_entries_by_name: dict[str, list[dict]] = {}

        # Scoped hot-path indexes for waiting-state resolution.
        # group_id -> normalized_label -> [item_id]
        self._side_ids_by_group_and_label: dict[str, dict[str, list[str]]] = {}

        # group_id -> normalized_label -> [modifier_id]
        self._modifier_ids_by_group_and_label: dict[str, dict[str, list[str]]] = {}

        self._load()

    def _load(self) -> None:
        """
        Load menu and entity index, then build runtime indexes.

        This method must fail fast on malformed input and leave the store
        in a fully consistent state.
        """
        try:
            with open(self.menu_path, "r", encoding="utf-8") as f:
                raw_menu = json.load(f)

            raw_items = raw_menu.get("items", {})
            raw_categories = raw_menu.get("categories", {})

            if not raw_items:
                raise MenuLoadError("menu.json contains no items")

            with open(self.entity_index_path, "r", encoding="utf-8") as f:
                raw_entity_index = json.load(f)

            self.items = {
                item_id: self._parse_menu_item(raw_item)
                for item_id, raw_item in raw_items.items()
            }
            self.categories = dict(raw_categories)
            self.entity_index = self._normalize_entity_index(raw_entity_index)

            self._build_indexes()

        except Exception as e:
            raise MenuLoadError(str(e)) from e

    def _normalize_entity_index(self, raw_entity_index: dict) -> dict[str, list[dict]]:
        normalized: dict[str, list[dict]] = {}

        for raw_key, raw_value in raw_entity_index.items():
            norm_key = normalize_text(str(raw_key))
            if not norm_key:
                continue

            entries = raw_value if isinstance(raw_value, list) else [raw_value]
            bucket = normalized.setdefault(norm_key, [])
            for entry in entries:
                if isinstance(entry, dict):
                    bucket.append(entry)

        return normalized

    def _parse_menu_item(self, raw: dict) -> MenuItem:
        aliases = tuple(str(alias) for alias in raw.get("aliases", []))
        normalized_aliases = tuple(
            norm_alias
            for alias in aliases
            if (norm_alias := normalize_text(alias))
        )

        voice_labels = self._build_voice_labels(
            primary_name=str(raw["name"]),
            aliases=aliases,
        )

        return MenuItem(
            item_id=raw["item_id"],
            name=raw["name"],
            normalized_name=normalize_text(raw["name"]),
            aliases=aliases,
            normalized_aliases=normalized_aliases,
            voice_labels=voice_labels,
            pricing=self._parse_pricing(raw["pricing"]),
            side_groups=self._parse_side_groups(raw.get("side_groups", [])),
            modifier_groups=self._parse_modifier_groups(raw.get("modifier_groups", [])),
            available=raw.get("available", True),
        )

    def _parse_pricing(self, raw: dict) -> Pricing:
        mode = raw["mode"]

        if mode == "fixed":
            return Pricing(
                mode="fixed",
                price_cents=raw["price_cents"],
                currency=raw.get("currency", "USD"),
            )

        if mode == "variant":
            variants = [
                PricingVariant(
                    variant_id=variant["variant_id"],
                    label=variant["label"],
                    normalized_label=normalize_text(variant["label"]),
                    price_cents=variant["price_cents"],
                )
                for variant in raw.get("variants", [])
            ]
            return Pricing(
                mode="variant",
                variants=variants,
                currency=raw.get("currency", "USD"),
            )

        if mode == "unit":
            return Pricing(
                mode="unit",
                price_cents=raw["price_cents"],
                currency=raw.get("currency", "USD"),
            )

        raise MenuLoadError(f"Unknown pricing mode: {mode}")

    def _parse_side_groups(self, groups: list[dict]) -> list[SideGroup]:
        parsed: list[SideGroup] = []

        for group in groups:
            choices = []
            for choice in group.get("choices", []):
                aliases = tuple(str(alias) for alias in choice.get("aliases", []))
                normalized_aliases = tuple(
                    norm_alias
                    for alias in aliases
                    if (norm_alias := normalize_text(alias))
                )
                voice_labels = self._build_voice_labels(
                    primary_name=str(choice["name"]),
                    aliases=aliases,
                )

                choices.append(
                    SideChoice(
                        item_id=choice["item_id"],
                        name=choice["name"],
                        normalized_name=normalize_text(choice["name"]),
                        pricing=self._parse_pricing(choice["pricing"]),
                        aliases=aliases,
                        normalized_aliases=normalized_aliases,
                        voice_labels=voice_labels,
                    )
                )

            raw_prompt_noun = str(group.get("prompt_noun") or "").strip()
            prompt_noun = raw_prompt_noun or _derive_prompt_noun(group["name"]) or None
            prompt_verb = str(group.get("prompt_verb") or "").strip() or "would you like"

            parsed.append(
                SideGroup(
                    group_id=group["group_id"],
                    name=group["name"],
                    normalized_name=normalize_text(group["name"]),
                    is_required=group["is_required"],
                    min_selector=group["min_selector"],
                    max_selector=group["max_selector"],
                    choices=choices,
                    prompt_noun=prompt_noun,
                    prompt_verb=prompt_verb,
                    allow_duplicate_selections=bool(group.get("allow_duplicate_selections", True)),
                )
            )

        return parsed

    def _parse_modifier_groups(self, groups: list[dict]) -> list[ModifierGroup]:
        parsed: list[ModifierGroup] = []

        for group in groups:
            choices = []
            for choice in group.get("choices", []):
                aliases = tuple(str(alias) for alias in choice.get("aliases", []))
                normalized_aliases = tuple(
                    norm_alias
                    for alias in aliases
                    if (norm_alias := normalize_text(alias))
                )
                voice_labels = self._build_voice_labels(
                    primary_name=str(choice["name"]),
                    aliases=aliases,
                )

                choices.append(
                    ModifierChoice(
                        modifier_id=choice["modifier_id"],
                        name=choice["name"],
                        normalized_name=normalize_text(choice["name"]),
                        price_cents=choice["price_cents"],
                        aliases=aliases,
                        normalized_aliases=normalized_aliases,
                        voice_labels=voice_labels,
                    )
                )

            raw_prompt_noun = str(group.get("prompt_noun") or "").strip()
            prompt_noun = raw_prompt_noun or _derive_prompt_noun(group["name"]) or None
            prompt_verb = str(group.get("prompt_verb") or "").strip() or "would you like"

            parsed.append(
                ModifierGroup(
                    group_id=group["group_id"],
                    name=group["name"],
                    normalized_name=normalize_text(group["name"]),
                    is_required=group["is_required"],
                    min_selector=group["min_selector"],
                    max_selector=group["max_selector"],
                    choices=choices,
                    prompt_noun=prompt_noun,
                    prompt_verb=prompt_verb,
                )
            )

        return parsed

    def _build_indexes(self) -> None:
        """
        Build deterministic runtime indexes once at startup.

        Category index also supports singular/plural tolerance.
        """
        self._item_by_name.clear()
        self._item_ids_by_alias.clear()
        self._item_ids_by_voice_label.clear()
        self._category_name_index.clear()
        self._discoverable_item_ids.clear()
        self._modifier_entries_by_name.clear()
        self._side_ids_by_group_and_label.clear()
        self._modifier_ids_by_group_and_label.clear()

        for item in self.items.values():
            if item.normalized_name:
                self._item_by_name[item.normalized_name] = item

            for alias in item.normalized_aliases:
                self._item_ids_by_alias.setdefault(alias, []).append(item.item_id)

            for voice_label in item.voice_labels:
                self._item_ids_by_voice_label.setdefault(voice_label, []).append(item.item_id)

            for side_group in item.side_groups:
                group_bucket = self._side_ids_by_group_and_label.setdefault(side_group.group_id, {})

                for choice in side_group.choices:
                    for label in choice.voice_labels:
                        ids = group_bucket.setdefault(label, [])
                        if choice.item_id not in ids:
                            ids.append(choice.item_id)

            for modifier_group in item.modifier_groups:
                group_bucket = self._modifier_ids_by_group_and_label.setdefault(
                    modifier_group.group_id,
                    {},
                )

                for choice in modifier_group.choices:
                    for label in choice.voice_labels:
                        ids = group_bucket.setdefault(label, [])
                        if choice.modifier_id not in ids:
                            ids.append(choice.modifier_id)

        for category in self.categories.values():
            category_name = str(category.get("name", ""))
            norm_name = normalize_text(category_name)
            if not norm_name:
                continue

            self._category_name_index[norm_name] = category

            if norm_name.endswith("s"):
                singular = norm_name[:-1]
                if singular:
                    self._category_name_index[singular] = category
            else:
                self._category_name_index[f"{norm_name}s"] = category

            for item_id in category.get("item_ids", []) or []:
                if item_id in self.items:
                    self._discoverable_item_ids.add(item_id)

        for norm_key, entries in self.entity_index.items():
            for entry in entries:
                entity_type = entry.get("type")

                if entity_type == "item":
                    item_id = entry.get("item_id")
                    if item_id and item_id in self.items:
                        self._discoverable_item_ids.add(item_id)

                elif entity_type == "modifier":
                    self._modifier_entries_by_name.setdefault(norm_key, []).append(entry)

    def _build_voice_labels(
        self,
        *,
        primary_name: str,
        aliases: tuple[str, ...],
    ) -> tuple[str, ...]:
        values: list[str] = []

        def add(value: str | None) -> None:
            normalized = normalize_text(value or "")
            if not normalized:
                return
            if normalized not in values:
                values.append(normalized)

        add(primary_name)
        for alias in aliases:
            add(alias)

        expanded: list[str] = []
        for value in values:
            for variant in self._voice_label_variants(value):
                if variant and variant not in expanded:
                    expanded.append(variant)

        return tuple(expanded)

    def _voice_label_variants(self, normalized_text: str) -> list[str]:
        variants: list[str] = []

        def add(value: str) -> None:
            value = " ".join((value or "").split()).strip()
            if value and value not in variants:
                variants.append(value)

        base = normalized_text
        add(base)

        stripped_number = self._strip_leading_menu_number(base)
        add(stripped_number)

        # Unicode separator normalization: em dash (—), en dash (–), and
        # related Unicode dashes are not in string.punctuation so they survive
        # normalize_text.  Strip them so "Korean Tacos — Spicy Chicken" →
        # "korean tacos spicy chicken" is also a valid voice label.
        _UNICODE_DASH_RE = re.compile(r"[‐-―−—–﹘﹣－]")
        _sep_stripped = _UNICODE_DASH_RE.sub(" ", stripped_number)
        _sep_stripped = re.sub(r"\s+", " ", _sep_stripped).strip()
        if _sep_stripped and _sep_stripped != stripped_number:
            add(_sep_stripped)
            add(_sep_stripped.replace("bbq", "barbecue"))
            add(_sep_stripped.replace("barbecue", "bbq"))
        else:
            _sep_stripped = stripped_number  # No dashes — use original for compact join

        menu_number = self._leading_menu_number(base)
        if menu_number and stripped_number:
            stripped_tokens = [token for token in stripped_number.split() if token]
            if stripped_tokens:
                add(f"{menu_number} {stripped_tokens[-1]}")
                add(f"number {menu_number} {stripped_tokens[-1]}")

        no_hash = stripped_number.replace("#", "").strip()
        add(no_hash)

        bbq_to_barbecue = base.replace("bbq", "barbecue")
        barbecue_to_bbq = base.replace("barbecue", "bbq")
        add(bbq_to_barbecue)
        add(barbecue_to_bbq)

        stripped_bbq = stripped_number.replace("bbq", "barbecue")
        stripped_barbecue = stripped_number.replace("barbecue", "bbq")
        add(stripped_bbq)
        add(stripped_barbecue)

        # For multi-word labels emit a compact joined form so that spoken
        # contractions like "cheeseburger" match "Cheese Burger" and
        # "doublebaconburger" matches "Double Bacon Burger".
        # Only do this for labels with 2+ tokens and total length >= 7
        # to avoid spurious joins of very short tokens.
        # Use the separator-stripped form as the base so dashes don't embed
        # in the compact join (e.g. "Korean Tacos — Spicy Chicken" should
        # yield "koreanTacosspicychicken", not "koreanTacos—spicychicken").
        _join_base = _sep_stripped if _sep_stripped else stripped_number
        _base_tokens = [t for t in _join_base.split() if t]
        if len(_base_tokens) >= 2:
            joined = "".join(_base_tokens)
            if len(joined) >= 7:
                add(joined)
                # Also add singular/plural of the joined form.
                for sv in self._singular_plural_variants(joined):
                    add(sv)

        singular_plural_variants: list[str] = []
        for value in list(variants):
            singular_plural_variants.extend(self._singular_plural_variants(value))

        for value in singular_plural_variants:
            add(value)

        return variants

    def _strip_leading_menu_number(self, text: str) -> str:
        value = (text or "").strip()
        idx = 0

        while idx < len(value) and value[idx] in {"#", " "}:
            idx += 1

        start_digits = idx
        while idx < len(value) and value[idx].isdigit():
            idx += 1

        if idx > start_digits:
            while idx < len(value) and value[idx] in {".", ")", "-", " "}:
                idx += 1
            return value[idx:].strip()

        return value

    def _leading_menu_number(self, text: str) -> str:
        value = (text or "").strip()
        idx = 0

        while idx < len(value) and value[idx] in {"#", " "}:
            idx += 1

        start_digits = idx
        while idx < len(value) and value[idx].isdigit():
            idx += 1

        if idx > start_digits:
            return value[start_digits:idx]

        return ""

    def _singular_plural_variants(self, text: str) -> list[str]:
        tokens = [token for token in text.split() if token]
        if not tokens:
            return []

        variants: list[str] = []

        for i, token in enumerate(tokens):
            changed = self._toggle_plural(token)
            if not changed or changed == token:
                continue

            clone = list(tokens)
            clone[i] = changed
            variants.append(" ".join(clone))

        return variants

    def _toggle_plural(self, token: str) -> str:
        if len(token) <= 2:
            return token

        if token.endswith("ies") and len(token) > 3:
            return token[:-3] + "y"

        if token.endswith("es") and len(token) > 3:
            singular = token[:-2]
            if singular:
                return singular

        if token.endswith("s") and len(token) > 3:
            return token[:-1]

        if token.endswith("y") and len(token) > 2:
            return token[:-1] + "ies"

        return token + "s"

    def get_item(self, item_id: str) -> MenuItem:
        item = self.items.get(item_id)
        if item is None:
            raise KeyError(f"Item not found: {item_id}")
        return item

    def find_entity(
        self,
        key: str,
        *,
        allowed_types: set[str] | None = None,
        parent_item_id: str | None = None,
        group_id: str | None = None,
    ) -> list[dict]:
        raw_entries = self.entity_index.get(key)
        if not raw_entries:
            return []

        results: list[dict] = []

        for entry in raw_entries:
            entity_type = entry.get("type")
            if not entity_type:
                continue

            if allowed_types and entity_type not in allowed_types:
                continue

            if parent_item_id is not None and entry.get("parent_item_id") != parent_item_id:
                continue

            if group_id is not None and entry.get("group_id") != group_id:
                continue

            results.append(entry)

        return results

    def find_item_exact(self, normalized_name: str) -> MenuItem | None:
        return self._item_by_name.get(normalized_name)

    def find_item_ids_by_alias(self, normalized_alias: str) -> list[str]:
        return self._item_ids_by_alias.get(normalized_alias, [])

    def find_item_ids_by_voice_label(self, normalized_voice_label: str) -> list[str]:
        return self._item_ids_by_voice_label.get(normalized_voice_label, [])

    def find_side_ids_for_group_by_label(
        self,
        group_id: str,
        normalized_label: str,
    ) -> list[str]:
        if not group_id or not normalized_label:
            return []
        return list(self._side_ids_by_group_and_label.get(group_id, {}).get(normalized_label, []))

    def find_modifier_ids_for_group_by_label(
        self,
        group_id: str,
        normalized_label: str,
    ) -> list[str]:
        if not group_id or not normalized_label:
            return []
        return list(
            self._modifier_ids_by_group_and_label.get(group_id, {}).get(normalized_label, [])
        )

    def find_category_by_name(self, normalized_text: str) -> dict | None:
        return self._category_name_index.get(normalized_text)

    def is_discoverable_item(self, item_id: str) -> bool:
        return item_id in self._discoverable_item_ids

    def iter_discoverable_items(self) -> list[MenuItem]:
        return [
            self.items[item_id]
            for item_id in self._discoverable_item_ids
            if item_id in self.items
        ]

    def find_discoverable_item_mentions(self, normalized_text: str) -> list[dict]:
        if not normalized_text:
            return []

        mentions: list[dict] = []
        seen_mentions: set[tuple[int, int, str]] = set()

        for item in self.iter_discoverable_items():
            labels = sorted(
                {
                    item.normalized_name,
                    *item.normalized_aliases,
                    *item.voice_labels,
                },
                key=len,
                reverse=True,
            )

            for label in labels:
                if not label:
                    continue

                pattern = re.compile(rf"(?<!\w){re.escape(label)}(?!\w)")
                for match in pattern.finditer(normalized_text):
                    key = (match.start(), match.end(), item.item_id)
                    if key in seen_mentions:
                        continue

                    seen_mentions.add(key)
                    mentions.append(
                        {
                            "item_id": item.item_id,
                            "item_name": item.name,
                            "normalized_name": item.normalized_name,
                            "matched_text": match.group(0),
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )

        mentions.sort(key=lambda mention: (mention["start"], -(mention["end"] - mention["start"])))

        pruned: list[dict] = []
        for mention in mentions:
            overlaps_existing = any(
                mention["start"] < existing["end"] and mention["end"] > existing["start"]
                for existing in pruned
            )
            if overlaps_existing:
                continue
            pruned.append(mention)

        return pruned

    def find_modifier_entities(self, normalized_text: str) -> list[dict]:
        return list(self._modifier_entries_by_name.get(normalized_text, []))
