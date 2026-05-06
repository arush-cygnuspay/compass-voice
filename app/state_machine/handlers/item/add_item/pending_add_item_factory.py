# app/state_machine/handlers/item/add_item/pending_add_item_factory.py
from __future__ import annotations

import re

from app.menu.models import MenuItem
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.pending_item_models import (
    PendingAddItem,
    PendingModifierChoice,
    PendingModifierGroup,
    PendingSideChoice,
    PendingSideGroup,
    PendingVariantChoice,
)


def _build_variant_choices(raw_variants) -> list[PendingVariantChoice]:
    return [
        PendingVariantChoice(
            variant_id=variant.variant_id,
            name=variant.label,
            normalized_name=normalize_text(variant.label),
        )
        for variant in (raw_variants or [])
        if variant.label
    ]


def _index_variants(
    variants: list[PendingVariantChoice],
) -> tuple[
    dict[str, PendingVariantChoice],
    dict[str, PendingVariantChoice],
    tuple[str, ...],
    tuple[str, ...],
]:
    by_id: dict[str, PendingVariantChoice] = {}
    by_normalized_name: dict[str, PendingVariantChoice] = {}
    names: list[str] = []

    for variant in variants:
        by_id[variant.variant_id] = variant
        if variant.normalized_name and variant.normalized_name not in by_normalized_name:
            by_normalized_name[variant.normalized_name] = variant
        if variant.name:
            names.append(variant.name)

    name_tuple = tuple(names)
    return by_id, by_normalized_name, name_tuple, name_tuple[:4]


def _append_bucket_value[T](mapping: dict[str, list[T]], key: str, value: T) -> None:
    bucket = mapping.get(key)
    if bucket is None:
        mapping[key] = [value]
        return
    bucket.append(value)


def _simplify_measurement_label(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    simplified = re.sub(r"\b\d+\s*(?:oz|ounce|ounces|inch|inches|pc|pcs|piece|pieces)\b", " ", normalized)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    return simplified


def _build_match_texts(
    *,
    name: str,
    group_name: str = "",
    aliases: tuple[str, ...] = (),
    voice_labels: tuple[str, ...] = (),
) -> tuple[str, ...]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    add(name)
    simplified_name = _simplify_measurement_label(name)
    if simplified_name:
        add(simplified_name)

    for alias in aliases:
        add(alias)
        simplified_alias = _simplify_measurement_label(alias)
        if simplified_alias:
            add(simplified_alias)

    for label in voice_labels:
        add(label)
        simplified_label = _simplify_measurement_label(label)
        if simplified_label:
            add(simplified_label)

    normalized_group_name = normalize_text(group_name)
    normalized_name = normalize_text(name)
    name_tokens = normalized_name.split()
    group_tokens = set(normalized_group_name.split())
    if len(name_tokens) >= 2:
        suffix = name_tokens[-1]
        stem = " ".join(name_tokens[:-1]).strip()
        if stem and suffix in {"meat", "bun", "cheese"} and suffix in group_tokens:
            add(stem)

    return tuple(candidates)


def build_pending_add_item(item: MenuItem) -> PendingAddItem:
    item_variants: list[PendingVariantChoice] = []
    if item.pricing and item.pricing.mode == "variant":
        item_variants = _build_variant_choices(item.pricing.variants)

    (
        item_variants_by_id,
        item_variants_by_normalized_name,
        item_variant_names,
        top_item_variant_names,
    ) = _index_variants(item_variants)

    side_groups: list[PendingSideGroup] = []
    side_groups_by_id: dict[str, PendingSideGroup] = {}
    side_choice_by_item_id: dict[str, PendingSideChoice] = {}

    for group in item.side_groups or []:
        pending_choices: list[PendingSideChoice] = []
        choices_by_item_id: dict[str, PendingSideChoice] = {}
        choices_by_normalized_name: dict[str, list[PendingSideChoice]] = {}
        choice_names: list[str] = []
        normalized_choice_names: list[str] = []

        for choice in group.choices or []:
            variants = _build_variant_choices(choice.pricing.variants)
            (
                variants_by_id,
                variants_by_normalized_name,
                variant_names,
                top_variant_names,
            ) = _index_variants(variants)

            pending_choice = PendingSideChoice(
                item_id=choice.item_id,
                name=choice.name,
                pricing_mode=choice.pricing.mode,
                normalized_name=normalize_text(choice.name),
                match_texts=_build_match_texts(
                    name=choice.name,
                    group_name=group.name,
                    aliases=tuple(choice.aliases or ()),
                    voice_labels=tuple(choice.voice_labels or ()),
                ),
                variants=variants,
                variants_by_id=variants_by_id,
                variants_by_normalized_name=variants_by_normalized_name,
                variant_names=variant_names,
                top_variant_names=top_variant_names,
            )

            pending_choices.append(pending_choice)
            choices_by_item_id[pending_choice.item_id] = pending_choice
            side_choice_by_item_id[pending_choice.item_id] = pending_choice

            if pending_choice.normalized_name:
                _append_bucket_value(
                    choices_by_normalized_name,
                    pending_choice.normalized_name,
                    pending_choice,
                )
                normalized_choice_names.append(pending_choice.normalized_name)

            if pending_choice.name:
                choice_names.append(pending_choice.name)

        pending_group = PendingSideGroup(
            group_id=group.group_id,
            name=group.name,
            is_required=bool(group.is_required),
            min_selector=int(group.min_selector or 1),
            max_selector=int(group.max_selector or 1),
            choices=pending_choices,
            choices_by_item_id=choices_by_item_id,
            choices_by_normalized_name=choices_by_normalized_name,
            choice_names=tuple(choice_names),
            normalized_choice_names=tuple(normalized_choice_names),
            top_choice_names=tuple(choice_names[:3]),
        )

        side_groups.append(pending_group)
        side_groups_by_id[pending_group.group_id] = pending_group

    modifier_groups: list[PendingModifierGroup] = []
    modifier_groups_by_id: dict[str, PendingModifierGroup] = {}
    modifier_choice_by_id: dict[str, PendingModifierChoice] = {}

    for group in item.modifier_groups or []:
        pending_choices: list[PendingModifierChoice] = []
        choices_by_modifier_id: dict[str, PendingModifierChoice] = {}
        choices_by_normalized_name: dict[str, list[PendingModifierChoice]] = {}
        choice_names: list[str] = []
        normalized_choice_names: list[str] = []

        for choice in group.choices or []:
            pending_choice = PendingModifierChoice(
                modifier_id=choice.modifier_id,
                name=choice.name,
                group_id=group.group_id,
                normalized_name=normalize_text(choice.name),
                match_texts=_build_match_texts(
                    name=choice.name,
                    group_name=group.name,
                    aliases=tuple(choice.aliases or ()),
                    voice_labels=tuple(choice.voice_labels or ()),
                ),
            )

            pending_choices.append(pending_choice)
            choices_by_modifier_id[pending_choice.modifier_id] = pending_choice
            modifier_choice_by_id[pending_choice.modifier_id] = pending_choice

            if pending_choice.normalized_name:
                _append_bucket_value(
                    choices_by_normalized_name,
                    pending_choice.normalized_name,
                    pending_choice,
                )
                normalized_choice_names.append(pending_choice.normalized_name)

            if pending_choice.name:
                choice_names.append(pending_choice.name)

        pending_group = PendingModifierGroup(
            group_id=group.group_id,
            name=group.name,
            is_required=bool(group.is_required),
            min_selector=int(group.min_selector or 1),
            max_selector=int(group.max_selector or 1),
            choices=pending_choices,
            choices_by_modifier_id=choices_by_modifier_id,
            choices_by_normalized_name=choices_by_normalized_name,
            choice_names=tuple(choice_names),
            normalized_choice_names=tuple(normalized_choice_names),
            top_choice_names=tuple(choice_names[:4]),
        )

        modifier_groups.append(pending_group)
        modifier_groups_by_id[pending_group.group_id] = pending_group

    return PendingAddItem(
        item_id=item.item_id,
        item_name=item.name,
        item_variants=item_variants,
        side_groups=side_groups,
        modifier_groups=modifier_groups,
        item_variants_by_id=item_variants_by_id,
        item_variants_by_normalized_name=item_variants_by_normalized_name,
        item_variant_names=item_variant_names,
        top_item_variant_names=top_item_variant_names,
        side_groups_by_id=side_groups_by_id,
        side_choice_by_item_id=side_choice_by_item_id,
        modifier_groups_by_id=modifier_groups_by_id,
        modifier_choice_by_id=modifier_choice_by_id,
    )
