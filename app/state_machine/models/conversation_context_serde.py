# app/state_machine/models/conversation_context_serde.py
"""Serialization / deserialization helpers for ConversationContext.

Extracted from conversation_context.py so the dataclass itself stays focused
on state representation and behaviour.  All functions here are private — only
ConversationContext.to_dict / from_dict are the intended consumers.
"""
from __future__ import annotations

from typing import Any

from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.pending_item_models import (
    InterruptProposal,
    ModifierSelection,
    PendingAddItem,
    PendingModifierChoice,
    PendingModifierGroup,
    PendingSideChoice,
    PendingSideGroup,
    PendingVariantChoice,
)


# ---------------------------------------------------------------------------
# Internal bucket utility
# ---------------------------------------------------------------------------

def _append_bucket_value(mapping: dict[str, list[Any]], key: str, value: Any) -> None:
    bucket = mapping.get(key)
    if bucket is None:
        mapping[key] = [value]
        return
    bucket.append(value)


# ---------------------------------------------------------------------------
# Variant index helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# to_dict helpers
# ---------------------------------------------------------------------------

def _pending_variant_to_dict(value: PendingVariantChoice) -> dict:
    return {
        "variant_id": value.variant_id,
        "name": value.name,
        "normalized_name": value.normalized_name,
    }


def _pending_side_choice_to_dict(value: PendingSideChoice) -> dict:
    return {
        "item_id": value.item_id,
        "name": value.name,
        "pricing_mode": value.pricing_mode,
        "normalized_name": value.normalized_name,
        "match_texts": list(value.match_texts),
        "variants": [_pending_variant_to_dict(variant) for variant in value.variants],
    }


def _pending_modifier_choice_to_dict(value: PendingModifierChoice) -> dict:
    return {
        "modifier_id": value.modifier_id,
        "name": value.name,
        "group_id": value.group_id,
        "normalized_name": value.normalized_name,
        "match_texts": list(value.match_texts),
    }


def _pending_side_group_to_dict(value: PendingSideGroup) -> dict:
    return {
        "group_id": value.group_id,
        "name": value.name,
        "is_required": value.is_required,
        "min_selector": value.min_selector,
        "max_selector": value.max_selector,
        "choices": [_pending_side_choice_to_dict(choice) for choice in value.choices],
        "allow_duplicate_selections": value.allow_duplicate_selections,
    }


def _pending_modifier_group_to_dict(value: PendingModifierGroup) -> dict:
    return {
        "group_id": value.group_id,
        "name": value.name,
        "is_required": value.is_required,
        "min_selector": value.min_selector,
        "max_selector": value.max_selector,
        "choices": [_pending_modifier_choice_to_dict(choice) for choice in value.choices],
    }


def _pending_add_item_to_dict(value: PendingAddItem) -> dict:
    return {
        "item_id": value.item_id,
        "item_name": value.item_name,
        "item_variants": [_pending_variant_to_dict(v) for v in value.item_variants],
        "side_groups": [_pending_side_group_to_dict(g) for g in value.side_groups],
        "modifier_groups": [_pending_modifier_group_to_dict(g) for g in value.modifier_groups],
    }


def _modifier_selection_to_dict(value: ModifierSelection) -> dict:
    return {
        "modifier_id": value.modifier_id,
        "name": value.name,
        "action": value.action,
        "instruction": value.instruction,
    }


# ---------------------------------------------------------------------------
# from_dict helpers
# ---------------------------------------------------------------------------

def _pending_variant_from_dict(data: dict) -> PendingVariantChoice:
    name = data["name"]
    normalized_name = data.get("normalized_name") or normalize_text(name)
    return PendingVariantChoice(
        variant_id=data["variant_id"],
        name=name,
        normalized_name=normalized_name,
    )


def _pending_side_choice_from_dict(data: dict) -> PendingSideChoice:
    variants = [_pending_variant_from_dict(v) for v in data.get("variants", [])]
    (
        variants_by_id,
        variants_by_normalized_name,
        variant_names,
        top_variant_names,
    ) = _index_variants(variants)

    name = data["name"]
    normalized_name = data.get("normalized_name") or normalize_text(name)

    return PendingSideChoice(
        item_id=data["item_id"],
        name=name,
        pricing_mode=data["pricing_mode"],
        normalized_name=normalized_name,
        match_texts=tuple(data.get("match_texts", ()) or ()),
        variants=variants,
        variants_by_id=variants_by_id,
        variants_by_normalized_name=variants_by_normalized_name,
        variant_names=variant_names,
        top_variant_names=top_variant_names,
    )


def _pending_modifier_choice_from_dict(data: dict) -> PendingModifierChoice:
    name = data["name"]
    normalized_name = data.get("normalized_name") or normalize_text(name)
    return PendingModifierChoice(
        modifier_id=data["modifier_id"],
        name=name,
        group_id=data["group_id"],
        normalized_name=normalized_name,
        match_texts=tuple(data.get("match_texts", ()) or ()),
    )


def _pending_side_group_from_dict(data: dict) -> PendingSideGroup:
    choices = [_pending_side_choice_from_dict(c) for c in data.get("choices", [])]

    choices_by_item_id: dict[str, PendingSideChoice] = {}
    choices_by_normalized_name: dict[str, list[PendingSideChoice]] = {}
    choice_names: list[str] = []
    normalized_choice_names: list[str] = []

    for choice in choices:
        choices_by_item_id[choice.item_id] = choice
        if choice.normalized_name:
            _append_bucket_value(choices_by_normalized_name, choice.normalized_name, choice)
            normalized_choice_names.append(choice.normalized_name)
        if choice.name:
            choice_names.append(choice.name)

    return PendingSideGroup(
        group_id=data["group_id"],
        name=data["name"],
        is_required=bool(data["is_required"]),
        min_selector=int(data["min_selector"]),
        max_selector=int(data["max_selector"]),
        choices=choices,
        choices_by_item_id=choices_by_item_id,
        choices_by_normalized_name=choices_by_normalized_name,
        choice_names=tuple(choice_names),
        normalized_choice_names=tuple(normalized_choice_names),
        top_choice_names=tuple(choice_names[:3]),
        allow_duplicate_selections=bool(data.get("allow_duplicate_selections", True)),
    )


def _pending_modifier_group_from_dict(data: dict) -> PendingModifierGroup:
    choices = [_pending_modifier_choice_from_dict(c) for c in data.get("choices", [])]

    choices_by_modifier_id: dict[str, PendingModifierChoice] = {}
    choices_by_normalized_name: dict[str, list[PendingModifierChoice]] = {}
    choice_names: list[str] = []
    normalized_choice_names: list[str] = []

    for choice in choices:
        choices_by_modifier_id[choice.modifier_id] = choice
        if choice.normalized_name:
            _append_bucket_value(choices_by_normalized_name, choice.normalized_name, choice)
            normalized_choice_names.append(choice.normalized_name)
        if choice.name:
            choice_names.append(choice.name)

    return PendingModifierGroup(
        group_id=data["group_id"],
        name=data["name"],
        is_required=bool(data["is_required"]),
        min_selector=int(data["min_selector"]),
        max_selector=int(data["max_selector"]),
        choices=choices,
        choices_by_modifier_id=choices_by_modifier_id,
        choices_by_normalized_name=choices_by_normalized_name,
        choice_names=tuple(choice_names),
        normalized_choice_names=tuple(normalized_choice_names),
        top_choice_names=tuple(choice_names[:4]),
    )


def _pending_add_item_from_dict(data: dict) -> PendingAddItem:
    item_variants = [_pending_variant_from_dict(v) for v in data.get("item_variants", [])]
    (
        item_variants_by_id,
        item_variants_by_normalized_name,
        item_variant_names,
        top_item_variant_names,
    ) = _index_variants(item_variants)

    side_groups = [_pending_side_group_from_dict(g) for g in data.get("side_groups", [])]
    modifier_groups = [_pending_modifier_group_from_dict(g) for g in data.get("modifier_groups", [])]

    side_groups_by_id: dict[str, PendingSideGroup] = {}
    side_choice_by_item_id: dict[str, PendingSideChoice] = {}
    for group in side_groups:
        side_groups_by_id[group.group_id] = group
        for choice in group.choices:
            side_choice_by_item_id[choice.item_id] = choice

    modifier_groups_by_id: dict[str, PendingModifierGroup] = {}
    modifier_choice_by_id: dict[str, PendingModifierChoice] = {}
    for group in modifier_groups:
        modifier_groups_by_id[group.group_id] = group
        for choice in group.choices:
            modifier_choice_by_id[choice.modifier_id] = choice

    return PendingAddItem(
        item_id=data["item_id"],
        item_name=data["item_name"],
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


def _modifier_selection_from_dict(data: dict) -> ModifierSelection:
    return ModifierSelection(
        modifier_id=data["modifier_id"],
        name=data["name"],
        action=data.get("action", "add"),
        instruction=data.get("instruction"),
    )


def _deserialize_segment_slots(slot_list: list[dict]) -> tuple:
    """Rebuild a tuple of SlotValue from serialised dicts."""
    if not slot_list:
        return ()
    return tuple(
        SlotValue(
            name=s.get("name", ""),
            value=s.get("value", ""),
            raw=s.get("raw", ""),
            start=s.get("start"),
            end=s.get("end"),
            confidence=s.get("confidence"),
        )
        for s in slot_list
    )
