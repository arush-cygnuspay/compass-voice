from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PricingVariant:
    variant_id: str
    label: str
    normalized_label: str
    price_cents: int


@dataclass(slots=True)
class Pricing:
    mode: str  # fixed | variant | unit
    price_cents: Optional[int] = None
    variants: list[PricingVariant] | None = None
    currency: str = "USD"


@dataclass(slots=True)
class SideChoice:
    item_id: str
    name: str
    normalized_name: str
    pricing: Pricing
    aliases: tuple[str, ...] = ()
    normalized_aliases: tuple[str, ...] = ()
    voice_labels: tuple[str, ...] = ()


@dataclass(slots=True)
class SideGroup:
    group_id: str
    name: str
    normalized_name: str
    is_required: bool
    min_selector: int
    max_selector: int
    choices: list[SideChoice]


@dataclass(slots=True)
class ModifierChoice:
    modifier_id: str
    name: str
    normalized_name: str
    price_cents: int
    aliases: tuple[str, ...] = ()
    normalized_aliases: tuple[str, ...] = ()
    voice_labels: tuple[str, ...] = ()


@dataclass(slots=True)
class ModifierGroup:
    group_id: str
    name: str
    normalized_name: str
    is_required: bool
    min_selector: int
    max_selector: int
    choices: list[ModifierChoice]


@dataclass(slots=True)
class MenuItem:
    item_id: str
    name: str
    normalized_name: str
    aliases: tuple[str, ...]
    normalized_aliases: tuple[str, ...]
    voice_labels: tuple[str, ...]
    pricing: Pricing
    side_groups: list[SideGroup]
    modifier_groups: list[ModifierGroup]
    available: bool


@dataclass(slots=True)
class ItemResolution:
    item: MenuItem
    score: float