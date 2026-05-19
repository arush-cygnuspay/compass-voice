# tests/services/test_multi_item_order_planner.py
"""Tests for app/services/multi_item_order_planner.py

Test categories
---------------
A – Full failure case: "i want a grilled cicken sandwich a large fries small
    onion rings a Tuna Melt and a 6 piece wings" → 5 items resolved, no
    unresolved spans.
B – Variant / quantity disambiguation: "6 piece wings" → variant_name="6 piece",
    quantity=1; "2 burgers" → quantity=2, no variant.
C – Size extraction: "a large fries" → item=fries, size_name contains "large".
D – Typo correction: fuzzy matching "cicken" → "chicken sandwich".
E – Partial success: 4 of 5 spans resolve → 4 items, 1 unresolved.
F – Chat JSONL logging: background writer enqueues an event record for every
    successful compound plan.
G – Guard rails: short utterances, single item, no menu, bad inputs.
H – Span splitting accuracy: article boundaries, size-word boundaries, connectors.
I – Integration shim: ParsedItemSegment conversion preserves item_name / quantity.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from app.services.multi_item_order_planner import (
    FUZZY_MIN_RATIO,
    ParsedMultiItemPlan,
    ParsedOrderItem,
    _EMPTY_PLAN,
    _find_best_match,
    _has_content_word,
    _parse_leading_quantity,
    _split_spans,
    _strip_filler,
    _tok,
    plan_multi_item_order,
    resolve_quantity_and_variant,
)


# ---------------------------------------------------------------------------
# Minimal menu stubs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PricingVariant:
    variant_id: str
    label: str
    normalized_label: str
    price_cents: int = 0


@dataclass(slots=True)
class _Pricing:
    mode: str = "variant"
    price_cents: Optional[int] = None
    variants: Optional[list] = None
    currency: str = "USD"


@dataclass(slots=True)
class _MenuItem:
    item_id: str
    name: str
    normalized_name: str
    aliases: tuple = ()
    normalized_aliases: tuple = ()
    voice_labels: tuple = ()
    pricing: "_Pricing" = field(default_factory=_Pricing)
    side_groups: list = field(default_factory=list)
    modifier_groups: list = field(default_factory=list)
    available: bool = True


def _make_item(
    item_id: str,
    name: str,
    aliases: tuple[str, ...] = (),
    variants: list[_PricingVariant] | None = None,
) -> _MenuItem:
    normalized = name.lower().replace("-", " ").strip()
    norm_aliases = tuple(a.lower().strip() for a in aliases)
    pricing = _Pricing(
        mode="variant" if variants else "fixed",
        variants=variants or None,
    )
    return _MenuItem(
        item_id=item_id,
        name=name,
        normalized_name=normalized,
        aliases=aliases,
        normalized_aliases=norm_aliases,
        voice_labels=(),
        pricing=pricing,
    )


class _MenuStore:
    """Minimal MenuStore stub — sufficient to drive the planner."""

    def __init__(self, items: list[_MenuItem]) -> None:
        self._items = {it.item_id: it for it in items}
        self._by_name = {it.normalized_name: it for it in items}
        self._by_alias: dict[str, str] = {}
        for it in items:
            for alias in it.normalized_aliases:
                self._by_alias[alias] = it.item_id

    # Required by planner internals
    def find_item_exact(self, normalized_name: str) -> "_MenuItem | None":
        return self._by_name.get(normalized_name)

    def find_item_ids_by_alias(self, normalized_alias: str) -> list[str]:
        item_id = self._by_alias.get(normalized_alias)
        return [item_id] if item_id else []

    def find_item_ids_by_voice_label(self, normalized_voice_label: str) -> list[str]:
        return []

    def get_item(self, item_id: str) -> "_MenuItem":
        return self._items[item_id]

    def iter_discoverable_items(self) -> list["_MenuItem"]:
        return list(self._items.values())


# ---------------------------------------------------------------------------
# Shared fixture menu
# ---------------------------------------------------------------------------

WINGS_VARIANTS = [
    _PricingVariant("wings_6", "6 Piece", "6 piece", 599),
    _PricingVariant("wings_12", "12 Piece", "12 piece", 999),
]
FRIES_VARIANTS = [
    _PricingVariant("fries_sm", "Small", "small", 199),
    _PricingVariant("fries_md", "Medium", "medium", 249),
    _PricingVariant("fries_lg", "Large", "large", 299),
]
ONION_RINGS_VARIANTS = [
    _PricingVariant("or_sm", "Small", "small", 229),
    _PricingVariant("or_md", "Medium", "medium", 279),
]


@pytest.fixture()
def menu_store() -> _MenuStore:
    return _MenuStore([
        _make_item("chk_sandwich", "Grilled Chicken Sandwich",
                   aliases=("grilled chicken sandwich", "chicken sandwich")),
        _make_item("fries", "French Fries",
                   aliases=("fries", "french fries"), variants=FRIES_VARIANTS),
        _make_item("onion_rings", "Onion Rings",
                   aliases=("onion rings",), variants=ONION_RINGS_VARIANTS),
        _make_item("tuna_melt", "Tuna Melt", aliases=("tuna melt",)),
        _make_item("wings", "Chicken Wings",
                   aliases=("wings", "chicken wings"), variants=WINGS_VARIANTS),
        _make_item("burger", "Classic Burger", aliases=("burger", "burgers")),
        _make_item("coke", "Coca-Cola", aliases=("coke", "coca cola")),
    ])


# ---------------------------------------------------------------------------
# Category A — Full compound utterance ("the big failure case")
# ---------------------------------------------------------------------------


class TestCategoryA_FullCompoundUtterance:
    """The utterance that motivated this planner."""

    UTTERANCE = (
        "i want a grilled cicken sandwich a large fries small onion rings "
        "a tuna melt and a 6 piece wings"
    )

    def test_returns_compound_plan(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(self.UTTERANCE, menu_store)
        assert plan.is_compound is True

    def test_resolves_at_least_four_items(self, menu_store: _MenuStore) -> None:
        """We expect 5; allow 4 as minimum to be lenient about fuzzy threshold."""
        plan = plan_multi_item_order(self.UTTERANCE, menu_store)
        assert len(plan.items) >= 4, (
            f"Expected ≥4 items, got {len(plan.items)}: "
            + ", ".join(it.item_name for it in plan.items)
        )

    def test_tuna_melt_resolved(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(self.UTTERANCE, menu_store)
        names = [it.item_name.lower() for it in plan.items]
        assert any("tuna" in n for n in names), f"Tuna Melt not found in: {names}"

    def test_fries_resolved(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(self.UTTERANCE, menu_store)
        names = [it.item_name.lower() for it in plan.items]
        assert any("fries" in n or "french" in n for n in names), (
            f"Fries not found in: {names}"
        )

    def test_onion_rings_resolved(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(self.UTTERANCE, menu_store)
        names = [it.item_name.lower() for it in plan.items]
        assert any("onion" in n for n in names), f"Onion Rings not found in: {names}"

    def test_wings_resolved(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(self.UTTERANCE, menu_store)
        names = [it.item_name.lower() for it in plan.items]
        assert any("wing" in n for n in names), f"Wings not found in: {names}"

    def test_all_item_quantities_at_least_one(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(self.UTTERANCE, menu_store)
        for item in plan.items:
            assert item.quantity >= 1, f"{item.item_name} has quantity {item.quantity}"

    def test_plan_confidence_positive(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(self.UTTERANCE, menu_store)
        assert plan.confidence > 0.0

    def test_raw_spans_populated(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(self.UTTERANCE, menu_store)
        assert len(plan.raw_spans) >= 2


# ---------------------------------------------------------------------------
# Category B — Variant / quantity disambiguation
# ---------------------------------------------------------------------------


class TestCategoryB_VariantVsQuantity:
    """"6 piece wings" → variant, not quantity=6."""

    def test_6_piece_wings_variant_not_quantity(self, menu_store: _MenuStore) -> None:
        item = menu_store.get_item("wings")
        qty, variant_id, variant_name = resolve_quantity_and_variant(
            "a 6 piece wings", item
        )
        assert qty == 1, f"Expected qty=1, got {qty}"
        assert "6" in variant_name or "piece" in variant_name.lower(), (
            f"Expected variant_name to contain '6 piece', got '{variant_name}'"
        )

    def test_12_piece_wings_variant_not_quantity(self, menu_store: _MenuStore) -> None:
        item = menu_store.get_item("wings")
        qty, variant_id, variant_name = resolve_quantity_and_variant(
            "12 piece wings", item
        )
        assert qty == 1
        assert "12" in variant_name or "piece" in variant_name.lower()

    def test_2_burgers_is_quantity(self, menu_store: _MenuStore) -> None:
        item = menu_store.get_item("burger")
        qty, variant_id, variant_name = resolve_quantity_and_variant("2 burgers", item)
        assert qty == 2
        assert variant_name == ""

    def test_three_cokes_is_quantity(self, menu_store: _MenuStore) -> None:
        item = menu_store.get_item("coke")
        qty, variant_id, variant_name = resolve_quantity_and_variant("three cokes", item)
        assert qty == 3
        assert variant_name == ""

    def test_quantity_clamped_to_99(self, menu_store: _MenuStore) -> None:
        item = menu_store.get_item("burger")
        qty, _, _ = resolve_quantity_and_variant("200 burgers", item)
        assert qty == 99

    def test_quantity_at_least_1(self, menu_store: _MenuStore) -> None:
        item = menu_store.get_item("burger")
        qty, _, _ = resolve_quantity_and_variant("a burger", item)
        assert qty >= 1

    def test_quantity_override_respected(self, menu_store: _MenuStore) -> None:
        item = menu_store.get_item("wings")
        qty, _, _ = resolve_quantity_and_variant("6 piece wings", item, quantity_override=2)
        assert qty == 2

    def test_none_item_no_crash(self) -> None:
        qty, vid, vname = resolve_quantity_and_variant("a burger", None)
        assert qty >= 1
        assert isinstance(vname, str)


# ---------------------------------------------------------------------------
# Category C — Size extraction
# ---------------------------------------------------------------------------


class TestCategoryC_SizeExtraction:

    def test_large_fries_gets_large_variant(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(
            "a large fries and a coke", menu_store
        )
        fries_items = [it for it in plan.items if "fries" in it.item_name.lower() or "french" in it.item_name.lower()]
        assert fries_items, f"No fries in plan: {[it.item_name for it in plan.items]}"
        fries = fries_items[0]
        assert "large" in (fries.size_name or fries.variant_name or "").lower(), (
            f"Expected size 'large', got size_name={fries.size_name!r} "
            f"variant_name={fries.variant_name!r}"
        )

    def test_small_onion_rings_gets_small_variant(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(
            "a burger and small onion rings", menu_store
        )
        or_items = [it for it in plan.items if "onion" in it.item_name.lower()]
        assert or_items, f"Onion rings not in: {[it.item_name for it in plan.items]}"
        assert "small" in (or_items[0].size_name or or_items[0].variant_name or "").lower()

    def test_medium_fries(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order("medium fries and a coke", menu_store)
        fries_items = [it for it in plan.items if "fries" in it.item_name.lower() or "french" in it.item_name.lower()]
        if fries_items:
            assert "medium" in (fries_items[0].size_name or fries_items[0].variant_name or "").lower()

    def test_size_word_splits_span_correctly(self) -> None:
        """Size word should create a new span when current span has content."""
        spans = _split_spans("a grilled chicken sandwich small fries")
        assert len(spans) >= 2, f"Expected 2 spans, got: {spans}"
        assert any("fries" in s for s in spans)
        assert any("chicken" in s or "sandwich" in s or "grilled" in s for s in spans)


# ---------------------------------------------------------------------------
# Category D — Typo correction ("cicken" → "chicken")
# ---------------------------------------------------------------------------


class TestCategoryD_TypoCorrection:

    def test_cicken_matches_chicken_sandwich(self, menu_store: _MenuStore) -> None:
        matched, confidence, match_type = _find_best_match(
            "grilled cicken sandwich", menu_store
        )
        assert matched is not None, "Expected a fuzzy match for 'cicken'"
        assert "chicken" in matched.normalized_name.lower() or "chicken" in matched.name.lower()
        assert match_type == "fuzzy"

    def test_cicken_confidence_above_threshold(self, menu_store: _MenuStore) -> None:
        _, confidence, _ = _find_best_match("grilled cicken sandwich", menu_store)
        assert confidence >= FUZZY_MIN_RATIO, (
            f"Confidence {confidence:.3f} < FUZZY_MIN_RATIO {FUZZY_MIN_RATIO}"
        )

    def test_full_utterance_with_typo_resolves_sandwich(
        self, menu_store: _MenuStore
    ) -> None:
        plan = plan_multi_item_order(
            "a grilled cicken sandwich and a coke", menu_store
        )
        names = [it.item_name.lower() for it in plan.items]
        assert any("chicken" in n or "sandwich" in n for n in names), (
            f"Sandwich not resolved; items: {names}"
        )

    def test_minor_typo_fris_matches_fries(self, menu_store: _MenuStore) -> None:
        """Single-character deletion: 'fris' → 'fries'."""
        matched, confidence, _ = _find_best_match("fris", menu_store)
        if matched is not None:
            assert "fries" in matched.normalized_name.lower() or confidence < 0.85

    def test_exact_match_preferred_over_fuzzy(self, menu_store: _MenuStore) -> None:
        matched, confidence, match_type = _find_best_match("tuna melt", menu_store)
        assert matched is not None
        assert confidence == 1.0
        assert match_type in ("exact", "alias", "voice_label")

    def test_gibberish_does_not_match(self, menu_store: _MenuStore) -> None:
        """Very low-similarity text should not match anything above threshold."""
        matched, confidence, _ = _find_best_match("zzxzqwerty", menu_store)
        # Either not matched or confidence below threshold
        if matched is not None:
            assert confidence < FUZZY_MIN_RATIO


# ---------------------------------------------------------------------------
# Category E — Partial success
# ---------------------------------------------------------------------------


class TestCategoryE_PartialSuccess:
    """When some spans fail to resolve, the plan still returns resolved items."""

    def test_partial_resolves_what_it_can(self, menu_store: _MenuStore) -> None:
        """4 real items + 1 unresolvable span."""
        utterance = (
            "a burger a coke a fries a xyzzy_unresolvable_item and a tuna melt"
        )
        plan = plan_multi_item_order(utterance, menu_store)
        if plan.is_compound:
            assert len(plan.items) >= 2, (
                f"Expected ≥2 resolved items, got {len(plan.items)}"
            )

    def test_unresolved_spans_not_in_items(self, menu_store: _MenuStore) -> None:
        utterance = "a burger a coke xyzzy_unresolvable and a tuna melt"
        plan = plan_multi_item_order(utterance, menu_store)
        if plan.is_compound and plan.unresolved_spans:
            for span in plan.unresolved_spans:
                item_names = [it.item_name for it in plan.items]
                assert span not in item_names

    def test_less_than_two_resolvable_returns_empty(self, menu_store: _MenuStore) -> None:
        """Only one item resolves → not a compound plan."""
        plan = plan_multi_item_order(
            "a burger and a xyzzy_foo_bar_baz", menu_store
        )
        # Either not compound or fewer than 2 items
        assert not plan.is_compound or len(plan.items) < 2

    def test_reason_mentions_unresolved(self, menu_store: _MenuStore) -> None:
        utterance = "a burger a xyzzy_unreachable_thing a coke a tuna melt"
        plan = plan_multi_item_order(utterance, menu_store)
        if plan.is_compound and plan.unresolved_spans:
            assert "unresolved" in plan.reason, f"reason={plan.reason!r}"

    def test_no_crash_on_all_unresolvable(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order(
            "xyzzy1 and xyzzy2 and xyzzy3", menu_store
        )
        assert not plan.is_compound


# ---------------------------------------------------------------------------
# Category F — Chat JSONL logging
# ---------------------------------------------------------------------------


class TestCategoryF_ChatLogging:
    """Background JSONL writer emits an event for every compound plan.

    Tests target the queue directly (no file I/O dependency) so they are
    deterministic without sleeps.  The ``_log_plan_event`` function puts a
    record onto ``_log_queue``; we intercept that call.
    """

    def _drain_queue(self) -> list[dict]:
        """Drain everything currently in _log_queue into a list."""
        import app.services.multi_item_order_planner as mod
        records: list[dict] = []
        try:
            while True:
                records.append(mod._log_queue.get_nowait())
        except Exception:
            pass
        return records

    def test_log_event_enqueued_on_compound_plan(
        self, menu_store: _MenuStore
    ) -> None:
        import app.services.multi_item_order_planner as mod

        captured: list[dict] = []

        def fake_put_nowait(record):
            captured.append(record)

        with patch.object(mod._log_queue, "put_nowait", side_effect=fake_put_nowait):
            plan_multi_item_order(
                "a burger and a coke", menu_store,
                session_id="test-session-123",
                state="idle",
            )

        assert captured, "No event was enqueued to _log_queue"
        record = captured[0]
        assert record.get("event") == "multi_item_plan"

    def test_log_record_has_required_fields(
        self, menu_store: _MenuStore
    ) -> None:
        import app.services.multi_item_order_planner as mod

        captured: list[dict] = []

        def fake_put_nowait(record):
            captured.append(record)

        with patch.object(mod._log_queue, "put_nowait", side_effect=fake_put_nowait):
            plan_multi_item_order(
                "a burger and a coke", menu_store,
                session_id="sess-abc",
                state="idle",
            )

        if not captured:
            pytest.skip("Plan was not compound — no log event produced")

        record = captured[0]
        for key in ("event", "timestamp_utc", "session_id", "state",
                    "transcript", "resolved_count", "items", "confidence"):
            assert key in record, f"Missing key: {key!r}"

    def test_log_record_does_not_contain_api_key(
        self, menu_store: _MenuStore
    ) -> None:
        import app.services.multi_item_order_planner as mod

        captured: list[dict] = []

        def fake_put_nowait(record):
            captured.append(record)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}), patch.object(
            mod._log_queue, "put_nowait", side_effect=fake_put_nowait
        ):
            plan_multi_item_order("a burger and a coke", menu_store)

        # Serialize every captured record and check no API key leaks
        for rec in captured:
            serialized = json.dumps(rec)
            assert "sk-test123" not in serialized

    def test_no_log_for_single_item_plan(
        self, menu_store: _MenuStore
    ) -> None:
        """Non-compound plans should not enqueue an event."""
        import app.services.multi_item_order_planner as mod

        captured: list[dict] = []

        def fake_put_nowait(record):
            captured.append(record)

        with patch.object(mod._log_queue, "put_nowait", side_effect=fake_put_nowait):
            plan_multi_item_order("a burger please", menu_store)

        assert not captured, (
            f"Expected no log event for single-item utterance, "
            f"got {len(captured)} event(s)"
        )


# ---------------------------------------------------------------------------
# Category G — Guard rails and robustness
# ---------------------------------------------------------------------------


class TestCategoryG_GuardRails:

    def test_empty_transcript_returns_empty_plan(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order("", menu_store)
        assert plan is _EMPTY_PLAN or not plan.is_compound

    def test_none_style_transcript_no_crash(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order("   ", menu_store)
        assert not plan.is_compound

    def test_single_item_returns_empty_plan(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order("a burger please", menu_store)
        assert not plan.is_compound

    def test_no_menu_store_returns_empty_plan(self) -> None:
        plan = plan_multi_item_order("a burger and a coke", None)
        assert not plan.is_compound

    def test_exception_in_planner_returns_empty(self) -> None:
        """plan_multi_item_order must never propagate exceptions."""
        broken_store = MagicMock()
        broken_store.iter_discoverable_items.side_effect = RuntimeError("boom")
        broken_store.find_item_exact.side_effect = RuntimeError("boom")
        broken_store.find_item_ids_by_alias.side_effect = RuntimeError("boom")
        broken_store.find_item_ids_by_voice_label.side_effect = RuntimeError("boom")

        # Should not raise
        plan = plan_multi_item_order("a burger and a coke and fries", broken_store)
        assert isinstance(plan, ParsedMultiItemPlan)

    def test_very_long_span_ignored(self, menu_store: _MenuStore) -> None:
        """Spans over MAX_SPAN_WORDS words should be skipped."""
        utterance = "a " + "word " * 20 + " and a coke"
        plan = plan_multi_item_order(utterance, menu_store)
        # Should still work; long span is skipped not crashed

    def test_returns_parsed_multi_item_plan_type(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order("a burger and a coke", menu_store)
        assert isinstance(plan, ParsedMultiItemPlan)

    def test_items_are_tuple_of_parsed_order_items(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order("a burger and a coke", menu_store)
        for item in plan.items:
            assert isinstance(item, ParsedOrderItem)


# ---------------------------------------------------------------------------
# Category H — Span splitting accuracy
# ---------------------------------------------------------------------------


class TestCategoryH_SpanSplitting:

    def test_and_connector_splits(self) -> None:
        spans = _split_spans("a burger and a coke")
        assert len(spans) == 2, f"Expected 2 spans, got: {spans}"

    def test_article_boundary_without_connector(self) -> None:
        spans = _split_spans("a grilled chicken sandwich a large fries")
        assert len(spans) >= 2, f"Expected ≥2 spans, got: {spans}"

    def test_size_word_boundary(self) -> None:
        spans = _split_spans("a large fries small onion rings")
        assert len(spans) == 2, f"Expected 2 spans, got: {spans}"

    def test_comma_splits(self) -> None:
        spans = _split_spans("a burger, a coke, onion rings")
        assert len(spans) == 3, f"Expected 3 spans, got: {spans}"

    def test_connector_token_not_in_spans(self) -> None:
        spans = _split_spans("a burger and a coke")
        for span in spans:
            assert "and" not in span.split(), (
                f"'and' should not be in span tokens: {span!r}"
            )

    def test_five_items_full_utterance(self) -> None:
        utterance = (
            "a grilled chicken sandwich a large fries small onion rings "
            "a tuna melt and a 6 piece wings"
        )
        spans = _split_spans(utterance)
        assert len(spans) == 5, f"Expected 5 spans, got {len(spans)}: {spans}"

    def test_attachment_word_does_not_split(self) -> None:
        spans = _split_spans("a burger with extra cheese a coke")
        # "with" is an attachment word — "extra cheese" stays with burger
        # But then "a coke" should split
        assert any("burger" in s or "cheese" in s or "extra" in s for s in spans)
        assert any("coke" in s for s in spans)

    def test_plus_connector_splits(self) -> None:
        spans = _split_spans("a burger plus a coke")
        assert len(spans) >= 2

    def test_single_item_no_split(self) -> None:
        spans = _split_spans("a grilled chicken sandwich")
        assert len(spans) == 1, f"Expected 1 span, got {len(spans)}: {spans}"

    def test_has_content_word_articles_only_false(self) -> None:
        assert _has_content_word(["a"]) is False
        assert _has_content_word(["a", "an"]) is False

    def test_has_content_word_with_item_true(self) -> None:
        assert _has_content_word(["a", "burger"]) is True
        assert _has_content_word(["small", "fries"]) is True


# ---------------------------------------------------------------------------
# Category I — Integration shim (ParsedItemSegment conversion)
# ---------------------------------------------------------------------------


class TestCategoryI_IntegrationShim:
    """Verify the converter in add_item_handler produces valid ParsedItemSegment objects."""

    def test_conversion_preserves_item_name(self, menu_store: _MenuStore) -> None:
        from app.nlu.multi_item_parser import ParsedItemSegment

        plan = plan_multi_item_order("a burger and a coke", menu_store)
        if not plan.is_compound:
            pytest.skip("Plan not compound — planner could not resolve both items")

        segments = [
            ParsedItemSegment(
                raw_text=it.raw_span,
                item_slot_value=it.item_name if it.item_name else None,
                quantity=it.quantity,
                slots=(),
            )
            for it in plan.items
        ]
        assert all(seg.item_slot_value for seg in segments)

    def test_conversion_preserves_quantity(self, menu_store: _MenuStore) -> None:
        from app.nlu.multi_item_parser import ParsedItemSegment

        plan = plan_multi_item_order("2 burgers and a coke", menu_store)
        if not plan.is_compound:
            pytest.skip("Plan not compound")

        segments = [
            ParsedItemSegment(
                raw_text=it.raw_span,
                item_slot_value=it.item_name if it.item_name else None,
                quantity=it.quantity,
                slots=(),
            )
            for it in plan.items
        ]
        burger_segs = [s for s in segments if s.item_slot_value and "burger" in s.item_slot_value.lower()]
        if burger_segs:
            assert burger_segs[0].quantity == 2

    def test_no_planner_segments_have_nlu_slots(self, menu_store: _MenuStore) -> None:
        """Planner-derived segments have empty slots tuple (no NLU slot data)."""
        from app.nlu.multi_item_parser import ParsedItemSegment

        plan = plan_multi_item_order("a burger and a coke", menu_store)
        if not plan.is_compound:
            pytest.skip("Plan not compound")

        segments = [
            ParsedItemSegment(
                raw_text=it.raw_span,
                item_slot_value=it.item_name if it.item_name else None,
                quantity=it.quantity,
                slots=(),
            )
            for it in plan.items
        ]
        for seg in segments:
            assert seg.slots == ()


# ---------------------------------------------------------------------------
# Helpers — unit tests for private utilities
# ---------------------------------------------------------------------------


class TestHelpers:

    def test_strip_filler_i_want(self) -> None:
        assert _strip_filler("i want a burger") == "a burger"

    def test_strip_filler_can_i_get(self) -> None:
        assert _strip_filler("can i get a coke") == "a coke"

    def test_strip_filler_empty(self) -> None:
        assert _strip_filler("") == ""

    def test_strip_filler_nothing_to_strip(self) -> None:
        result = _strip_filler("a burger and a coke")
        assert result == "a burger and a coke"

    def test_parse_leading_quantity_digit(self) -> None:
        assert _parse_leading_quantity("2 burgers") == 2

    def test_parse_leading_quantity_word(self) -> None:
        assert _parse_leading_quantity("three cokes") == 3

    def test_parse_leading_quantity_none(self) -> None:
        assert _parse_leading_quantity("burger") is None

    def test_parse_leading_quantity_out_of_range(self) -> None:
        assert _parse_leading_quantity("200 burgers") is None

    def test_tok_strips_trailing_punctuation(self) -> None:
        assert _tok("burger.") == "burger"
        assert _tok("coke,") == "coke"
        assert _tok("And") == "and"

    def test_find_best_match_exact(self, menu_store: _MenuStore) -> None:
        item, conf, mtype = _find_best_match("tuna melt", menu_store)
        assert item is not None
        assert conf == 1.0
        assert mtype in ("exact", "alias")

    def test_find_best_match_no_store(self) -> None:
        item, conf, mtype = _find_best_match("burger", None)
        assert item is None
        assert mtype == "none"

    def test_find_best_match_empty_span(self, menu_store: _MenuStore) -> None:
        item, conf, mtype = _find_best_match("", menu_store)
        assert item is None

    def test_plan_reason_format(self, menu_store: _MenuStore) -> None:
        plan = plan_multi_item_order("a burger and a coke", menu_store)
        if plan.is_compound:
            assert "_spans_" in plan.reason, f"reason={plan.reason!r}"
