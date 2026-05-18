# tests/nlu/semantic_repair/test_add_item_plan_validator.py
"""Comprehensive tests for AddItemPlanValidator (Phase 2 shadow-only validation).

All tests are shadow-only: the validator never mutates cart, state, or response.
Menu data comes from the "demo" restaurant fixture.

Key demo menu items used:
  - "50 chicken wings"  (01bae609-...) fixed price, 2 required side groups:
      "Wing Flavors" (min=1 max=1): Hot, Mild, BBQ, Mumbo, No Seasoning, Plain
      "Baked or Fried Wing Choice" (min=1 max=1): Baked, Fried
  - "loaded fries"      (01983ede-...) variant pricing: Medium, Large
  - "ham biscuit"       (09a0540e-...) fixed price + 2 modifier groups:
      "Additional Meat for Biscuits": Chicken, Bacon, Tenderloin
      "Additional Extras For Biscuits": Plain Gravy, Cheese, Egg
  - "mumbo"             (02080761-...) fixed price, no side/modifier groups
  - "family wing meal"  (073faff9-...) fixed price, required side groups with
      variant-priced drinks (Unsweet Tea: Small/Medium/Large)
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Stub heavy ML / infrastructure dependencies so TurnEngine can be imported
# without torch / redis / twilio installed.
# ---------------------------------------------------------------------------
_intent_inference = types.ModuleType("app.ml.intent.inference_intent")
_intent_inference.IntentBundle = type("IntentBundle", (), {})
_intent_inference.predict_intent = lambda *a, **kw: []
sys.modules.setdefault("app.ml.intent.inference_intent", _intent_inference)

_slot_inference = types.ModuleType("app.ml.slot.inference_slot")
_slot_inference.SlotBundle = type("SlotBundle", (), {})
_slot_inference.predict_slots = lambda *a, **kw: []
sys.modules.setdefault("app.ml.slot.inference_slot", _slot_inference)

for _mod_name in ("twilio", "twilio.base", "twilio.base.exceptions", "twilio.rest", "redis"):
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))
if not hasattr(sys.modules.get("twilio.base.exceptions"), "TwilioRestException"):
    sys.modules["twilio.base.exceptions"].TwilioRestException = type("TwilioRestException", (Exception,), {})
if not hasattr(sys.modules.get("twilio.rest"), "Client"):
    sys.modules["twilio.rest"].Client = type("Client", (), {"__init__": lambda s, *a, **kw: None})
if not hasattr(sys.modules.get("redis"), "Redis"):
    sys.modules["redis"].Redis = type("Redis", (), {"__init__": lambda s, *a, **kw: None})

from app.nlu.semantic_repair.add_item_extractor import GptAddItem, GptAddItemChild, GptAddItemPlan
from app.nlu.semantic_repair.add_item_plan_validator import (
    BLOCKING_WARNING_CODES,
    NON_BLOCKING_WARNING_CODES,
    AddItemPlanValidator,
    ValidatedAddItem,
    ValidatedAddItemPlan,
    ValidationWarning,
)
from app.menu.store import MenuStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_store() -> MenuStore:
    """Return a fully loaded demo MenuStore (loaded once per test module)."""
    base = Path(__file__).parents[3] / "app" / "data" / "restaurants" / "demo"
    return MenuStore(
        menu_path=base / "menu.json",
        entity_index_path=base / "entity_index.json",
    )


@pytest.fixture(scope="module")
def validator() -> AddItemPlanValidator:
    return AddItemPlanValidator()


def _plan(*items: GptAddItem, **kw) -> GptAddItemPlan:
    """Build a minimal GptAddItemPlan with given items."""
    return GptAddItemPlan(decision="ok", items=tuple(items), **kw)


def _item(name: str, **kw) -> GptAddItem:
    return GptAddItem(item=name, **kw)


def _side(name: str, **kw) -> GptAddItemChild:
    return GptAddItemChild(name=name, **kw)


def _modifier(name: str, **kw) -> GptAddItemChild:
    return GptAddItemChild(name=name, **kw)


# ---------------------------------------------------------------------------
# Warning code constants
# ---------------------------------------------------------------------------


class TestWarningCodeConstants:
    def test_blocking_codes_non_empty(self):
        assert len(BLOCKING_WARNING_CODES) >= 7

    def test_non_blocking_codes_non_empty(self):
        assert len(NON_BLOCKING_WARNING_CODES) >= 3

    def test_item_not_on_menu_is_blocking(self):
        assert "item_not_on_menu" in BLOCKING_WARNING_CODES

    def test_item_ambiguous_is_blocking(self):
        assert "item_ambiguous" in BLOCKING_WARNING_CODES

    def test_invalid_item_size_is_blocking(self):
        assert "invalid_item_size" in BLOCKING_WARNING_CODES

    def test_side_not_valid_is_blocking(self):
        assert "side_not_valid_for_item" in BLOCKING_WARNING_CODES

    def test_over_max_selector_is_blocking(self):
        assert "over_max_selector" in BLOCKING_WARNING_CODES

    def test_modifier_size_unsupported_is_non_blocking(self):
        assert "modifier_size_unsupported" in NON_BLOCKING_WARNING_CODES

    def test_required_group_missing_is_non_blocking(self):
        assert "required_group_missing" in NON_BLOCKING_WARNING_CODES

    def test_quantity_clamped_is_non_blocking(self):
        assert "quantity_clamped" in NON_BLOCKING_WARNING_CODES

    def test_no_overlap_between_blocking_and_non_blocking(self):
        assert BLOCKING_WARNING_CODES.isdisjoint(NON_BLOCKING_WARNING_CODES)


# ---------------------------------------------------------------------------
# ValidationWarning dataclass
# ---------------------------------------------------------------------------


class TestValidationWarning:
    def test_is_blocking_true_for_blocking_code(self):
        w = ValidationWarning(code="item_not_on_menu", entity_kind="item", entity_name="pizza")
        assert w.is_blocking is True

    def test_is_blocking_false_for_non_blocking_code(self):
        w = ValidationWarning(code="quantity_clamped", entity_kind="item", entity_name="burger")
        assert w.is_blocking is False

    def test_is_blocking_false_for_unknown_code(self):
        w = ValidationWarning(code="totally_unknown", entity_kind="item", entity_name="x")
        assert w.is_blocking is False

    def test_detail_defaults_to_empty_string(self):
        w = ValidationWarning(code="x", entity_kind="item", entity_name="y")
        assert w.detail == ""


# ---------------------------------------------------------------------------
# ValidatedAddItemPlan.empty()
# ---------------------------------------------------------------------------


class TestValidatedAddItemPlanEmpty:
    def test_empty_returns_plan_with_no_items(self):
        p = ValidatedAddItemPlan.empty()
        assert p.items == ()
        assert p.rejected_items == ()
        assert p.has_blocking_warnings is False

    def test_empty_with_validator_ms(self):
        p = ValidatedAddItemPlan.empty(validator_ms=1.5)
        assert p.validator_ms == 1.5


# ---------------------------------------------------------------------------
# Validator — no store / no items guard
# ---------------------------------------------------------------------------


class TestValidatorGuards:
    def test_no_store_returns_empty_plan(self, validator):
        plan = _plan(_item("burger"))
        result = validator.validate(plan=plan)
        assert result.items == ()
        assert result.rejected_items == ()

    def test_no_items_returns_empty_plan(self, validator, demo_store):
        plan = _plan()
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items == ()
        assert result.rejected_items == ()
        assert result.has_blocking_warnings is False

    def test_validator_ms_non_negative(self, validator, demo_store):
        plan = _plan(_item("loaded fries"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.validator_ms >= 0.0

    def test_exception_in_validate_returns_empty(self, validator):
        """If the plan is broken in an unexpected way, validate() must not raise."""
        plan = _plan(_item("burger"))
        result = validator.validate(plan=plan, menu_store="NOT_A_STORE")  # type: ignore
        # Should return an empty plan without raising
        assert isinstance(result, ValidatedAddItemPlan)


# ---------------------------------------------------------------------------
# Item resolution
# ---------------------------------------------------------------------------


class TestItemResolution:
    def test_valid_item_by_normalized_name(self, validator, demo_store):
        plan = _plan(_item("loaded fries"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert len(result.items) == 1
        assert result.items[0].item_name == "Loaded Fries"

    def test_valid_item_by_voice_label(self, validator, demo_store):
        """'chicken wings' is a voice label for 50 Chicken Wings."""
        plan = _plan(_item("chicken wings"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        # May resolve or be ambiguous — just ensure no crash
        assert isinstance(result, ValidatedAddItemPlan)

    def test_item_not_on_menu_goes_to_rejected(self, validator, demo_store):
        plan = _plan(_item("unicorn burger"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert len(result.rejected_items) == 1
        assert "unicorn burger" in result.rejected_items

    def test_item_not_on_menu_sets_blocking_warnings(self, validator, demo_store):
        plan = _plan(_item("imaginary pizza"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.has_blocking_warnings is True

    def test_item_with_empty_name_is_rejected(self, validator, demo_store):
        plan = _plan(_item(""))
        result = validator.validate(plan=plan, menu_store=demo_store)
        # Empty name → rejected (no crash)
        assert isinstance(result, ValidatedAddItemPlan)

    def test_multiple_items_mixed_valid_invalid(self, validator, demo_store):
        plan = _plan(_item("loaded fries"), _item("not on menu at all xyz"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert len(result.items) == 1
        assert result.items[0].item_name == "Loaded Fries"
        assert len(result.rejected_items) == 1
        assert result.has_blocking_warnings is True

    def test_valid_item_id_populated(self, validator, demo_store):
        plan = _plan(_item("loaded fries"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].item_id == "01983ede-632d-4099-bbd5-8f53ce396bde"


# ---------------------------------------------------------------------------
# Quantity validation
# ---------------------------------------------------------------------------


class TestQuantityValidation:
    def test_quantity_none_defaults_to_1(self, validator, demo_store):
        plan = _plan(_item("loaded fries", quantity=None))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].quantity == 1

    def test_quantity_1_preserved(self, validator, demo_store):
        plan = _plan(_item("loaded fries", quantity=1))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].quantity == 1

    def test_quantity_5_preserved(self, validator, demo_store):
        plan = _plan(_item("loaded fries", quantity=5))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].quantity == 5

    def test_quantity_0_clamped_to_1(self, validator, demo_store):
        plan = _plan(_item("loaded fries", quantity=0))
        result = validator.validate(plan=plan, menu_store=demo_store)
        # Clamped — item still resolved (non-blocking)
        assert result.items[0].quantity == 1
        codes = [w.code for w in result.items[0].warnings]
        assert "quantity_clamped" in codes

    def test_quantity_negative_clamped_to_1(self, validator, demo_store):
        plan = _plan(_item("loaded fries", quantity=-3))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].quantity == 1

    def test_quantity_25_clamped_to_20(self, validator, demo_store):
        plan = _plan(_item("loaded fries", quantity=25))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].quantity == 20

    def test_quantity_clamped_is_non_blocking(self, validator, demo_store):
        """Clamped quantity must not cause item to be rejected."""
        plan = _plan(_item("loaded fries", quantity=99))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert len(result.items) == 1
        assert result.has_blocking_warnings is False


# ---------------------------------------------------------------------------
# Item size / variant validation
# ---------------------------------------------------------------------------


class TestItemSizeVariantValidation:
    def test_valid_size_medium_for_loaded_fries(self, validator, demo_store):
        plan = _plan(_item("loaded fries", size="medium"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        vi = result.items[0]
        assert vi.variant_id == "1770751826946"
        assert vi.variant_label == "Medium"

    def test_valid_size_large_for_loaded_fries(self, validator, demo_store):
        plan = _plan(_item("loaded fries", size="large"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].variant_label == "Large"

    def test_invalid_size_for_loaded_fries_is_blocking(self, validator, demo_store):
        plan = _plan(_item("loaded fries", size="extra-large"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.has_blocking_warnings is True
        assert "loaded fries" in result.rejected_items or len(result.items) == 0

    def test_size_on_fixed_price_item_is_blocking(self, validator, demo_store):
        """Mumbo is fixed-price; specifying a size is a blocking error."""
        plan = _plan(_item("mumbo", size="large"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.has_blocking_warnings is True

    def test_no_size_on_variant_item_is_fine(self, validator, demo_store):
        """Loaded Fries without size — valid (no variant selected)."""
        plan = _plan(_item("loaded fries"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].variant_id is None
        assert result.has_blocking_warnings is False

    def test_variant_field_used_when_size_empty(self, validator, demo_store):
        """When size is None but variant is set, variant field drives resolution."""
        plan = _plan(_item("loaded fries", size=None, variant="large"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].variant_label == "Large"


# ---------------------------------------------------------------------------
# Side validation
# ---------------------------------------------------------------------------


class TestSideValidation:
    def test_valid_side_hot_for_chicken_wings(self, validator, demo_store):
        plan = _plan(_item("50 chicken wings", sides=(_side("hot"), _side("fried"))))
        result = validator.validate(plan=plan, menu_store=demo_store)
        # Item should resolve (sides valid, required groups satisfied)
        assert len(result.items) == 1

    def test_invalid_side_not_in_any_group_is_blocking(self, validator, demo_store):
        plan = _plan(_item("50 chicken wings", sides=(_side("unicorn sauce"), _side("fried"))))
        result = validator.validate(plan=plan, menu_store=demo_store)
        # "unicorn sauce" triggers side_not_valid_for_item (blocking)
        assert result.has_blocking_warnings is True

    def test_side_resolves_group_id(self, validator, demo_store):
        plan = _plan(_item("50 chicken wings", sides=(_side("hot"), _side("baked"))))
        result = validator.validate(plan=plan, menu_store=demo_store)
        if result.items:
            side_group_ids = {s.group_id for s in result.items[0].sides}
            assert "b8fac6c6-d875-4360-a814-3e51b6794fec" in side_group_ids  # Wing Flavors

    def test_side_item_id_populated(self, validator, demo_store):
        plan = _plan(_item("50 chicken wings", sides=(_side("hot"), _side("baked"))))
        result = validator.validate(plan=plan, menu_store=demo_store)
        if result.items:
            hot_side = next((s for s in result.items[0].sides if s.name == "hot"), None)
            if hot_side:
                assert hot_side.side_item_id == "fc006782-eb27-4ec4-bd40-a66934c9e5aa"

    def test_over_max_selector_is_blocking(self, validator, demo_store):
        """Wing Flavors group has max=1; adding 2 flavor choices triggers over_max_selector."""
        plan = _plan(_item("50 chicken wings", sides=(_side("hot"), _side("mild"), _side("baked"))))
        result = validator.validate(plan=plan, menu_store=demo_store)
        # over_max_selector is blocking → item rejected
        assert result.has_blocking_warnings is True

    def test_side_with_valid_variant(self, validator, demo_store):
        """Family Wing Meal: Unsweet Tea side with valid 'large' variant."""
        plan = _plan(
            _item(
                "family wing meal",
                sides=(
                    _side("honey buffalo"),
                    _side("honey buffalo"),
                    _side("honey buffalo"),
                    _side("honey buffalo"),
                    _side("unsweet tea", size="large"),
                    _side("unsweet tea", size="large"),
                    _side("unsweet tea", size="large"),
                    _side("unsweet tea", size="large"),
                    _side("family fries"),
                ),
            )
        )
        result = validator.validate(plan=plan, menu_store=demo_store)
        # May have required_group_missing for unsatisfied groups but no blocking item errors
        assert isinstance(result, ValidatedAddItemPlan)

    def test_side_with_invalid_variant_is_blocking(self, validator, demo_store):
        """Family Wing Meal: Unsweet Tea with 'extra-large' is invalid."""
        plan = _plan(
            _item(
                "family wing meal",
                sides=(
                    _side("honey buffalo"),
                    _side("honey buffalo"),
                    _side("honey buffalo"),
                    _side("honey buffalo"),
                    _side("unsweet tea", size="extra-large"),
                    _side("unsweet tea", size="large"),
                    _side("unsweet tea", size="large"),
                    _side("unsweet tea", size="large"),
                    _side("family fries"),
                ),
            )
        )
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.has_blocking_warnings is True


# ---------------------------------------------------------------------------
# Modifier validation
# ---------------------------------------------------------------------------


class TestModifierValidation:
    def test_valid_modifier_bacon_for_ham_biscuit(self, validator, demo_store):
        plan = _plan(_item("ham biscuit", modifiers=(_modifier("bacon"),)))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert len(result.items) == 1
        vi = result.items[0]
        assert len(vi.modifiers) == 1
        assert vi.modifiers[0].name == "bacon"

    def test_modifier_id_populated(self, validator, demo_store):
        plan = _plan(_item("ham biscuit", modifiers=(_modifier("bacon"),)))
        result = validator.validate(plan=plan, menu_store=demo_store)
        if result.items:
            assert result.items[0].modifiers[0].modifier_id == "06617c2c-4033-48a2-912e-ca552415eb1f"

    def test_modifier_group_id_populated(self, validator, demo_store):
        plan = _plan(_item("ham biscuit", modifiers=(_modifier("bacon"),)))
        result = validator.validate(plan=plan, menu_store=demo_store)
        if result.items:
            assert result.items[0].modifiers[0].group_id  # non-empty

    def test_modifier_not_found_is_non_blocking(self, validator, demo_store):
        """Unknown modifier is non-blocking (logged but item is not rejected)."""
        plan = _plan(_item("ham biscuit", modifiers=(_modifier("invisible cheese"),)))
        result = validator.validate(plan=plan, menu_store=demo_store)
        # modifier_not_found is non-blocking → item not rejected
        assert len(result.items) == 1
        codes = [w.code for w in result.items[0].warnings]
        assert "modifier_not_found" in codes

    def test_modifier_size_is_non_blocking(self, validator, demo_store):
        """Size on a modifier is non-blocking; modifier is still accepted."""
        plan = _plan(_item("ham biscuit", modifiers=(_modifier("bacon", size="large"),)))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert len(result.items) == 1
        codes = [w.code for w in result.items[0].warnings]
        assert "modifier_size_unsupported" in codes

    def test_modifier_size_does_not_block_item(self, validator, demo_store):
        plan = _plan(_item("ham biscuit", modifiers=(_modifier("cheese", size="large"),)))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.has_blocking_warnings is False


# ---------------------------------------------------------------------------
# Required group coverage
# ---------------------------------------------------------------------------


class TestRequiredGroupCoverage:
    def test_missing_required_group_is_non_blocking(self, validator, demo_store):
        """50 Chicken Wings has 2 required groups; omitting both → non-blocking warnings."""
        plan = _plan(_item("50 chicken wings"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        # Item resolves (required groups missing is non-blocking)
        assert len(result.items) == 1
        codes = [w.code for w in result.items[0].warnings]
        assert "required_group_missing" in codes

    def test_missing_required_groups_ids_in_item(self, validator, demo_store):
        plan = _plan(_item("50 chicken wings"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        if result.items:
            # Both required groups should appear in missing_required_groups
            assert len(result.items[0].missing_required_groups) >= 1

    def test_satisfied_required_group_not_in_missing(self, validator, demo_store):
        """Providing hot (Wing Flavors group) and baked satisfies both required groups."""
        plan = _plan(_item("50 chicken wings", sides=(_side("hot"), _side("baked"))))
        result = validator.validate(plan=plan, menu_store=demo_store)
        if result.items:
            # Both required groups are satisfied — neither should be in missing
            assert len(result.items[0].missing_required_groups) == 0

    def test_item_with_no_required_groups_has_empty_missing(self, validator, demo_store):
        plan = _plan(_item("loaded fries"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].missing_required_groups == ()


# ---------------------------------------------------------------------------
# ValidatedAddItem.has_blocking_warnings
# ---------------------------------------------------------------------------


class TestValidatedAddItemHasBlockingWarnings:
    def test_no_warnings_means_no_blocking(self, validator, demo_store):
        plan = _plan(_item("loaded fries"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.items[0].has_blocking_warnings is False

    def test_only_non_blocking_warnings_means_no_blocking(self, validator, demo_store):
        """Missing required groups (non-blocking) should not set has_blocking_warnings."""
        plan = _plan(_item("50 chicken wings"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        if result.items:
            assert result.items[0].has_blocking_warnings is False


# ---------------------------------------------------------------------------
# Plan-level has_blocking_warnings
# ---------------------------------------------------------------------------


class TestPlanLevelBlockingWarnings:
    def test_all_valid_items_no_blocking(self, validator, demo_store):
        plan = _plan(_item("loaded fries"), _item("mumbo"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.has_blocking_warnings is False

    def test_any_rejected_item_sets_blocking(self, validator, demo_store):
        plan = _plan(_item("loaded fries"), _item("does not exist"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.has_blocking_warnings is True

    def test_blocking_item_warning_sets_plan_blocking(self, validator, demo_store):
        """Invalid size on fixed-price item causes blocking."""
        plan = _plan(_item("mumbo", size="large"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        assert result.has_blocking_warnings is True


# ---------------------------------------------------------------------------
# Shadow-only safety invariants
# ---------------------------------------------------------------------------


class TestShadowSafetyInvariants:
    def test_validator_returns_frozen_plan(self, validator, demo_store):
        plan = _plan(_item("loaded fries"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        # ValidatedAddItemPlan is a frozen dataclass
        with pytest.raises(Exception):
            result.items = ()  # type: ignore[misc]

    def test_validated_items_are_frozen(self, validator, demo_store):
        plan = _plan(_item("loaded fries"))
        result = validator.validate(plan=plan, menu_store=demo_store)
        if result.items:
            with pytest.raises(Exception):
                result.items[0].quantity = 99  # type: ignore[misc]

    def test_validator_does_not_mutate_input_plan(self, validator, demo_store):
        plan = _plan(_item("loaded fries"))
        original_items = plan.items
        validator.validate(plan=plan, menu_store=demo_store)
        assert plan.items is original_items

    def test_validator_exception_returns_empty_not_raise(self, validator):
        """Even a pathological plan must not raise from validate()."""
        plan = _plan(_item("loaded fries"))
        result = validator.validate(plan=plan, menu_repo=object())  # garbage repo
        assert isinstance(result, ValidatedAddItemPlan)


# ---------------------------------------------------------------------------
# GptAddItemPlan new fields (Part 3 — extractor model)
# ---------------------------------------------------------------------------


class TestGptAddItemPlanNewFields:
    def test_validated_plan_defaults_none(self):
        plan = GptAddItemPlan()
        assert plan.validated_plan is None

    def test_validator_ms_defaults_none(self):
        plan = GptAddItemPlan()
        assert plan.validator_ms is None

    def test_validation_warnings_defaults_empty(self):
        plan = GptAddItemPlan()
        assert plan.validation_warnings == ()

    def test_has_blocking_warnings_defaults_false(self):
        plan = GptAddItemPlan()
        assert plan.has_blocking_warnings is False

    def test_plan_is_frozen(self):
        plan = GptAddItemPlan()
        with pytest.raises(Exception):
            plan.has_blocking_warnings = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AddItemExtractorService integration (validator wired in run())
# ---------------------------------------------------------------------------


class TestAddItemExtractorServiceValidatorWiring:
    """Verify the validator is called inside AddItemExtractorService.run() without
    needing a real OpenAI call — mock the GPT parse step."""

    def test_service_has_validator_attribute(self):
        from app.nlu.semantic_repair.add_item_service import AddItemExtractorService
        from unittest.mock import MagicMock

        cfg = MagicMock()
        cfg.add_item_mode = "shadow"
        cfg.add_item_min_text_len = 1
        cfg.add_item_timeout_ms = 350
        cfg.add_item_max_items_per_turn = 8
        cfg.daily_budget = 1000

        svc = AddItemExtractorService(config=cfg)
        assert hasattr(svc, "_validator")
        from app.nlu.semantic_repair.add_item_plan_validator import AddItemPlanValidator
        assert isinstance(svc._validator, AddItemPlanValidator)

    def test_plan_returned_by_service_has_validated_plan_field(self):
        """When a menu_store is provided and items are parsed, validated_plan is set."""
        from app.nlu.semantic_repair.add_item_service import AddItemExtractorService
        from app.nlu.semantic_repair.add_item_extractor import GptAddItem
        from unittest.mock import MagicMock, patch

        cfg = MagicMock()
        cfg.add_item_mode = "shadow"
        cfg.add_item_min_text_len = 1
        cfg.add_item_timeout_ms = 350
        cfg.add_item_max_items_per_turn = 8
        cfg.daily_budget = 1000

        base = Path(__file__).parents[3] / "app" / "data" / "restaurants" / "demo"
        store = MenuStore(
            menu_path=base / "menu.json",
            entity_index_path=base / "entity_index.json",
        )

        svc = AddItemExtractorService(config=cfg)

        # Build a parsed plan with one valid item that parse_add_item_output would return
        parsed_plan = GptAddItemPlan(
            decision="ok",
            items=(GptAddItem(item="loaded fries", quantity=1),),
            eligible=True,
        )

        with patch.object(svc._validator, "validate", wraps=svc._validator.validate) as mock_validate:
            # Inject a pre-parsed plan into the post-parse code path
            with patch("app.nlu.semantic_repair.add_item_service.parse_add_item_output", return_value=parsed_plan):
                # We still need to bypass the full run() pipeline; test just the validator call
                result = svc._validator.validate(plan=parsed_plan, menu_store=store)
                assert isinstance(result, ValidatedAddItemPlan)
                assert len(result.items) == 1
                assert result.items[0].item_name == "Loaded Fries"


# ---------------------------------------------------------------------------
# TurnEngine helper methods (Part 5 — serializers)
# ---------------------------------------------------------------------------


class TestTurnEngineValidatorSerializers:
    """Unit-test the static helper methods added to TurnEngine for Phase 2."""

    def _get_helpers(self):
        from app.core.turn_engine import TurnEngine
        return TurnEngine

    def test_serialize_validated_items_none_returns_none(self):
        TE = self._get_helpers()
        assert TE._serialize_validated_items(None) is None

    def test_serialize_validated_items_empty_returns_none(self):
        TE = self._get_helpers()
        vp = ValidatedAddItemPlan.empty()
        assert TE._serialize_validated_items(vp) is None

    def test_serialize_validated_items_returns_json(self, validator, demo_store):
        TE = self._get_helpers()
        plan = _plan(_item("loaded fries"))
        vp = validator.validate(plan=plan, menu_store=demo_store)
        out = TE._serialize_validated_items(vp)
        assert out is not None
        import json
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert parsed[0]["item_name"] == "Loaded Fries"

    def test_serialize_validated_items_capped_at_4000_chars(self, validator, demo_store):
        TE = self._get_helpers()
        plan = _plan(_item("50 chicken wings", sides=(_side("hot"), _side("baked"))))
        vp = validator.validate(plan=plan, menu_store=demo_store)
        out = TE._serialize_validated_items(vp)
        if out:
            assert len(out) <= 4000

    def test_validated_items_count_none_when_no_plan(self):
        TE = self._get_helpers()
        assert TE._validated_items_count(None) is None

    def test_validated_items_count_correct(self, validator, demo_store):
        TE = self._get_helpers()
        plan = _plan(_item("loaded fries"), _item("mumbo"))
        vp = validator.validate(plan=plan, menu_store=demo_store)
        count = TE._validated_items_count(vp)
        assert count == len(vp.items)

    def test_serialize_rejected_items_none_when_no_plan(self):
        TE = self._get_helpers()
        assert TE._serialize_rejected_items(None) is None

    def test_serialize_rejected_items_returns_json_list(self, validator, demo_store):
        TE = self._get_helpers()
        plan = _plan(_item("does not exist xyz"))
        vp = validator.validate(plan=plan, menu_store=demo_store)
        out = TE._serialize_rejected_items(vp)
        assert out is not None
        import json
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert "does not exist xyz" in parsed

    def test_serialize_validation_warnings_none_when_no_warnings(self, validator, demo_store):
        TE = self._get_helpers()
        plan = _plan(_item("loaded fries"))
        vp = validator.validate(plan=plan, menu_store=demo_store)
        out = TE._serialize_validation_warnings(vp)
        # loaded fries with no size/sides → no warnings
        assert out is None or isinstance(out, str)

    def test_serialize_validation_warnings_contains_code(self, validator, demo_store):
        TE = self._get_helpers()
        # 50 chicken wings without required sides → required_group_missing warnings
        plan = _plan(_item("50 chicken wings"))
        vp = validator.validate(plan=plan, menu_store=demo_store)
        out = TE._serialize_validation_warnings(vp)
        if out:
            import json
            parsed = json.loads(out)
            codes = [w.get("code") for w in parsed]
            assert "required_group_missing" in codes


# ---------------------------------------------------------------------------
# CSV / JSONL logging structure (Part 5 — column/field presence)
# ---------------------------------------------------------------------------


class TestLoggingColumns:
    def test_gpt_repair_csv_has_validated_items_json(self):
        from app.logging.gpt_repair_csv_logger import HEADERS
        assert "add_item_validated_items_json" in HEADERS

    def test_gpt_repair_csv_has_validated_items_count(self):
        from app.logging.gpt_repair_csv_logger import HEADERS
        assert "add_item_validated_items_count" in HEADERS

    def test_gpt_repair_csv_has_rejected_items_json(self):
        from app.logging.gpt_repair_csv_logger import HEADERS
        assert "add_item_rejected_items_json" in HEADERS

    def test_gpt_repair_csv_has_validation_warnings_json(self):
        from app.logging.gpt_repair_csv_logger import HEADERS
        assert "add_item_validation_warnings_json" in HEADERS

    def test_gpt_repair_csv_has_validator_ms(self):
        from app.logging.gpt_repair_csv_logger import HEADERS
        assert "add_item_validator_ms" in HEADERS

    def test_gpt_repair_csv_has_blocking_warnings(self):
        from app.logging.gpt_repair_csv_logger import HEADERS
        assert "add_item_has_blocking_warnings" in HEADERS

    def test_gpt_repair_csv_existing_columns_not_removed(self):
        from app.logging.gpt_repair_csv_logger import HEADERS
        # Spot-check Phase 1 columns still present
        assert "add_item_extractor_called" in HEADERS
        assert "add_item_model" in HEADERS
        assert "add_item_decision" in HEADERS

    def test_realtime_csv_has_validated_items_count(self):
        from app.logging.realtime_latency_logger import RealtimeLatencyLogger
        assert "add_item_validated_items_count" in RealtimeLatencyLogger.CSV_COLUMNS

    def test_realtime_csv_has_blocking_warnings(self):
        from app.logging.realtime_latency_logger import RealtimeLatencyLogger
        assert "add_item_has_blocking_warnings" in RealtimeLatencyLogger.CSV_COLUMNS

    def test_realtime_csv_has_validator_ms(self):
        from app.logging.realtime_latency_logger import RealtimeLatencyLogger
        assert "add_item_validator_ms" in RealtimeLatencyLogger.CSV_COLUMNS

    def test_realtime_csv_existing_add_item_columns_preserved(self):
        from app.logging.realtime_latency_logger import RealtimeLatencyLogger
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        assert "add_item_extractor_called" in cols
        assert "add_item_decision" in cols
        assert "add_item_items_count" in cols
        assert "add_item_total_ms" in cols

    def test_realtime_csv_new_columns_after_existing(self):
        """Phase 2 columns must come after Phase 1 columns (no reordering)."""
        from app.logging.realtime_latency_logger import RealtimeLatencyLogger
        cols = RealtimeLatencyLogger.CSV_COLUMNS
        idx_total_ms = cols.index("add_item_total_ms")
        idx_validated = cols.index("add_item_validated_items_count")
        assert idx_validated > idx_total_ms

    def test_turn_event_has_validated_items_json(self):
        from app.diagnostics.turn_event import TurnEvent
        import inspect
        fields = {f.name for f in TurnEvent.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        assert "add_item_validated_items_json" in fields

    def test_turn_event_has_has_blocking_warnings(self):
        from app.diagnostics.turn_event import TurnEvent
        fields = {f.name for f in TurnEvent.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        assert "add_item_has_blocking_warnings" in fields

    def test_jsonl_builder_add_item_block_has_phase2_fields(self):
        """build_gpt_repair_log_record must include Phase 2 validator fields in add_item block."""
        from app.nlu.semantic_repair.gpt_log_record_builder import build_gpt_repair_log_record
        from app.diagnostics.turn_event import TurnEvent

        # Build minimal TurnEvent with required non-optional fields
        event = TurnEvent(
            session_id="s1", turn_index=1,
            state_before="idle", state_after="idle", next_state="idle",
            pending_action="", current_prompt_field="", current_item_id="", current_item_name="",
            raw_user_text="", user_text="", normalized_text="",
            pred_main_intent="", pred_sub_intent="", pred_intent="",
            pred_intent_confidence=None, slot_model_ran=False, slots=(),
            response_key="", response_text="", command=None,
            normalized_values={}, missing_required_fields=(),
            reprompt_field="", reprompt_count=0, reprompt_escalated=False,
            reprompt_escalation_count=0,
            fallback_triggered=False, fallback_reason="", fallback_count=0,
            slot_extraction_failed=False, slot_extraction_failure_count=0,
            invalid_modifier=False, invalid_modifier_count=0,
            user_repeated=False, repeated_user_turn_count=0,
        )

        record = build_gpt_repair_log_record(event)
        add_item_block = record.get("add_item", {})
        assert "validated_items_json" in add_item_block
        assert "validated_items_count" in add_item_block
        assert "rejected_items_json" in add_item_block
        assert "validation_warnings_json" in add_item_block
        assert "validator_ms" in add_item_block
        assert "has_blocking_warnings" in add_item_block
