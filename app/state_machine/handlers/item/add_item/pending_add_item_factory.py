# app/state_machine/handlers/item/add_item/pending_add_item_factory.py
from __future__ import annotations

from app.menu.models import MenuItem
from app.state_machine.conversation_context import (
    PendingAddItem,
    PendingModifierChoice,
    PendingModifierGroup,
    PendingSideChoice,
    PendingSideGroup,
    PendingVariantChoice,
)
from app.nlu.query_normalization.text_preprocessor import normalize_text


def build_pending_add_item(item: MenuItem) -> PendingAddItem:
    item_variants: list[PendingVariantChoice] = []
    if item.pricing and item.pricing.mode == "variant":
        item_variants = [
            PendingVariantChoice(
                variant_id=v.variant_id,
                name=v.label,
                normalized_name=normalize_text(v.label),
            )
            for v in (item.pricing.variants or [])
            if v.label
        ]

    side_groups: list[PendingSideGroup] = []
    for group in item.side_groups or []:
        side_groups.append(
            PendingSideGroup(
                group_id=group.group_id,
                name=group.name,
                is_required=bool(group.is_required),
                min_selector=int(group.min_selector or 1),
                max_selector=int(group.max_selector or 1),
                choices=[
                    PendingSideChoice(
                        item_id=choice.item_id,
                        name=choice.name,
                        pricing_mode=choice.pricing.mode,
                        normalized_name=normalize_text(choice.name),
                        variants=[
                            PendingVariantChoice(
                                variant_id=v.variant_id,
                                name=v.label,
                                normalized_name=normalize_text(v.label),
                            )
                            for v in (choice.pricing.variants or [])
                            if v.label
                        ],
                    )
                    for choice in (group.choices or [])
                ],
            )
        )

    modifier_groups: list[PendingModifierGroup] = []
    for group in item.modifier_groups or []:
        modifier_groups.append(
            PendingModifierGroup(
                group_id=group.group_id,
                name=group.name,
                is_required=bool(group.is_required),
                min_selector=int(group.min_selector or 1),
                max_selector=int(group.max_selector or 1),
                choices=[
                    PendingModifierChoice(
                        modifier_id=choice.modifier_id,
                        name=choice.name,
                        group_id=group.group_id,
                        normalized_name=normalize_text(choice.name),
                    )
                    for choice in (group.choices or [])
                ],
            )
        )

    return PendingAddItem(
        item_id=item.item_id,
        item_name=item.name,
        item_variants=item_variants,
        side_groups=side_groups,
        modifier_groups=modifier_groups,
    )