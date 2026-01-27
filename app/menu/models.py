# app/menu/models.py

from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class PricingVariant:
    variant_id: str
    label: str
    price_cents: int


@dataclass
class Pricing:
    mode: str                     # fixed | variant | unit
    price_cents: Optional[int] = None
    variants: Optional[List[PricingVariant]] = None
    currency: str = "USD"


@dataclass
class SideChoice:
    item_id: str
    name: str
    pricing: Pricing


@dataclass
class SideGroup:
    group_id: str
    name: str
    is_required: bool
    min_selector: int
    max_selector: int
    choices: List[SideChoice]


@dataclass
class ModifierChoice:
    modifier_id: str
    name: str
    price_cents: int


@dataclass
class ModifierGroup:
    group_id: str
    name: str
    is_required: bool
    min_selector: int
    max_selector: int
    choices: List[ModifierChoice]


@dataclass
class MenuItem:
    item_id: str
    name: str
    aliases: List[str]
    pricing: Pricing
    side_groups: List[SideGroup]
    modifier_groups: List[ModifierGroup]
    available: bool


@dataclass
class ItemResolution:
    item: MenuItem
    score: float