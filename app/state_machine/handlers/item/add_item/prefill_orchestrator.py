# app/state_machine/handlers/item/add_item/prefill_orchestrator.py
"""Prefill orchestration for the add-item flow.

Contains two public classes:

PendingItemCaptureHelper
    Low-level capture utilities (quantity, side groups, modifier groups, side
    variants, item variant).  Used by the "waiting for …" handlers as well as
    PrefillOrchestrator itself.

PrefillOrchestrator
    High-level coordinator that initialises a PendingAddItem snapshot,
    applies the unified multi-group prefill engine, delegates variant-side
    prefill, builds feedback/debug payloads, and hands off to
    ConfirmationDecisionHelper to produce the final HandlerResult.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from app.core.quantity_formatter import normalize_food_quantity
from app.menu.models import MenuItem
from app.nlu.modifier_instructions import speak as _speak_modifier
from app.nlu.nlu_result import SlotValue
from app.nlu.order_scaffolding import ORDER_FILLER_PREFIXES, ORDER_FILLER_TOKENS
from app.nlu.quantity_resolver import QuantityResolver
from app.state_machine.handlers.item.add_item.item_quantity_policy import (
    normalize_item_quantity,
)
from app.state_machine.models.conversation_state import ConversationState
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.nlu.slot_consumption import consume_slot_or_fallback
from app.state_machine.handler_result import HandlerResult
from app.state_machine.handlers.item.add_item.confirmation_decision_helper import (
    ConfirmationDecisionHelper,
)
from app.state_machine.handlers.item.add_item.group_collection_utils import (
    effective_group_selector_bounds,
)
from app.state_machine.handlers.item.add_item.modifier_group_resolver import (
    ModifierGroupResolver,
    build_modifier_option_candidates,
    extract_modifier_slot_values_normalized,
)
from app.state_machine.handlers.item.add_item.multi_group_prefill import (
    MultiGroupPrefillEngine,
    PrefillResult,
)
from app.state_machine.handlers.item.add_item.option_matching import (
    build_scoped_phrase_candidates,
)
from app.state_machine.handlers.item.add_item.pending_add_item_factory import (
    build_pending_add_item,
)
from app.state_machine.handlers.item.add_item.side_group_resolver import (
    SideGroupResolver,
    build_side_option_candidates,
    extract_side_slot_values_normalized,
)
from app.state_machine.models.conversation_context import ConversationContext
from app.utils.quantity_detection import (
    UNIT_PATTERN,
    extract_leading_quantity_phrase,
    normalize_quantity,
)
from app.utils.token_matcher import (
    is_controlled_partial_match,
    is_strong_token_match,
    tokenize,
)

logger = logging.getLogger(__name__)

_QUANTITY_RESOLVER = QuantityResolver()

# ---------------------------------------------------------------------------
# Module-level constants (shared with add_item_handler.py)
# ---------------------------------------------------------------------------

SIZE_WORDS = (
    "extra large",
    "large",
    "medium",
    "small",
    "regular",
    "mini",
    "xl",
)

# Re-export for any external code that still imports from this module.
ITEM_FILLER_PREFIXES: tuple[str, ...] = ORDER_FILLER_PREFIXES


def normalize_item_request_text(text: str) -> str:
    """Strip filler prefixes ("I want a", "add", "a", etc.) from item request text."""
    normalized = normalize_text(text or "")
    if not normalized:
        return ""
    changed = True
    while changed and normalized:
        changed = False
        for prefix in ORDER_FILLER_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                changed = True
                break
    return normalized


def _parse_quantity_value(raw: str) -> int | None:
    """Coerce a raw QUANTITY slot string to a positive int (or None).

    Handles:
    - plain digit strings: "2" → 2
    - ASR decimal encoding: "0.2" → 2 (via normalize_food_quantity)
    - word strings: "two", "a dozen" → 2, 12 (via normalize_quantity)
    - invalid/ambiguous: "1.5", garbage → None
    """
    text = (raw or "").strip()
    if not text:
        return None
    if text.isdigit():
        value = int(text)
        return value if value > 0 else None
    # Try decimal-encoding path first ("0.2" → 2, "1.0" → 1)
    nfq = normalize_food_quantity(text)
    if nfq is not None and nfq > 0:
        return nfq
    # Word / compound-number path ("two", "half dozen", "a pair")
    coerced = normalize_quantity(text)
    if isinstance(coerced, int) and coerced > 0:
        return coerced
    return None


# ---------------------------------------------------------------------------
# PendingItemCaptureHelper
# ---------------------------------------------------------------------------

class PendingItemCaptureHelper:
    """Low-level slot-capture utilities consumed by waiting-state handlers."""

    def __init__(
        self,
        side_resolver: SideGroupResolver | None = None,
        modifier_resolver: ModifierGroupResolver | None = None,
    ) -> None:
        self.side_resolver = side_resolver or SideGroupResolver()
        self.modifier_resolver = modifier_resolver or ModifierGroupResolver()

    # ------------------------------------------------------------------
    # Quantity
    # ------------------------------------------------------------------

    def prefill_quantity(
        self,
        *,
        context: ConversationContext,
        user_text: str,
    ) -> bool:
        if isinstance(context.quantity, int) and context.quantity > 0:
            return False

        for slot in context.last_slots or ():
            if str(getattr(slot, "name", "")).upper() != "QUANTITY":
                continue
            value = getattr(slot, "value", None)
            # Use normalize_food_quantity for all numeric types (int, float,
            # Decimal, numeric str).  This correctly decodes 0.N → N without
            # using round(), and rejects ambiguous values (1.5, negative).
            decoded = normalize_food_quantity(value)
            if decoded is not None and decoded > 0:
                context.quantity = decoded
                return True

        normalized_quantity = self._infer_quantity_from_text(
            context=context,
            user_text=user_text,
        )
        if isinstance(normalized_quantity, int) and normalized_quantity > 0:
            context.quantity = normalized_quantity
            return True

        return False

    def _infer_quantity_from_text(
        self,
        *,
        context: ConversationContext,
        user_text: str,
    ) -> int | None:
        normalized_text = normalize_text(user_text or "")
        if not normalized_text:
            return None

        def _extract_quantity_via_regex() -> int | None:
            if normalized_text.isdigit():
                return int(normalized_text)

            if re.fullmatch(
                r"(a|an|single|couple|one|two|three|four|five|six|seven|eight|nine|ten)",
                normalized_text,
            ):
                return normalize_quantity(normalized_text)

            if re.search(
                rf"\b(\d+|a|an|single|couple|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:{UNIT_PATTERN})\b",
                normalized_text,
            ):
                return normalize_quantity(normalized_text)

            if "dozen" in normalized_text:
                return normalize_quantity(normalized_text)

            leading = extract_leading_quantity_phrase(normalized_text)
            if leading is None:
                return None

            quantity, remainder, token = leading
            if not remainder:
                return quantity

            pending = context.pending_add_item
            if pending is None:
                return None

            pending_item_name = normalize_text(pending.item_name or "")
            if pending_item_name.startswith(token):
                return None

            item_slot_values = [
                normalize_text(str(getattr(slot, "value", "") or ""))
                for slot in (context.last_slots or ())
                if str(getattr(slot, "name", "")).upper() in {"ITEM", "MENU_ITEM"}
            ]
            candidate_names = [pending_item_name, *item_slot_values]
            candidate_names = [value for value in candidate_names if value]
            if not candidate_names:
                return None

            remainder_tokens = set(tokenize(remainder))
            if not remainder_tokens:
                return None

            for candidate_name in candidate_names:
                candidate_tokens = set(tokenize(candidate_name))
                if not candidate_tokens:
                    continue
                if candidate_tokens.issubset(remainder_tokens):
                    return quantity

            return None

        resolution = consume_slot_or_fallback(
            slots=context.last_slots or (),
            slot_labels=("QUANTITY",),
            fallback=_extract_quantity_via_regex,
            parse=_parse_quantity_value,
            consumer_site="add_item_handler.quantity",
        )
        return resolution.value

    # ------------------------------------------------------------------
    # Side groups
    # ------------------------------------------------------------------

    def prefill_side_groups(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
        start_index: int = 0,
    ) -> list[dict]:
        pending = context.pending_add_item
        if pending is None or not pending.side_groups:
            return []

        slot_values = extract_side_slot_values_normalized(context)
        feedback: list[dict] = []

        for group in pending.side_groups[start_index:]:
            existing_ids = list(context.selected_side_groups.get(group.group_id, []))
            if existing_ids:
                continue

            resolution = self.side_resolver.resolve(
                group=group,
                normalized_user_text=normalized_user_text,
                option_candidates=build_side_option_candidates(context, normalized_user_text),
                normalized_slot_values=slot_values,
                already_selected_ids=existing_ids,
            )
            unmatched_values = self._clean_prefill_unmatched_values(
                list(resolution.unmatched_values),
                item_name=pending.item_name,
            )
            min_selector, max_selector = effective_group_selector_bounds(group)
            accepted_limit = max_selector if max_selector > 0 else len(resolution.matched_item_ids)

            if not resolution.matched_item_ids and not unmatched_values:
                continue

            capped_ids = resolution.matched_item_ids[:accepted_limit]
            dropped_ids = resolution.matched_item_ids[accepted_limit:]
            accepted_names = [
                group.choices_by_item_id[item_id].name
                for item_id in capped_ids
                if item_id in group.choices_by_item_id
            ]
            dropped_names = [
                group.choices_by_item_id[item_id].name
                for item_id in dropped_ids
                if item_id in group.choices_by_item_id
            ]

            over_max = bool(dropped_ids)
            if capped_ids and not over_max:
                context.selected_side_groups[group.group_id] = capped_ids
                context.skipped_side_groups.discard(group.group_id)

            feedback.append(
                {
                    "group_id": group.group_id,
                    "kind": "side",
                    "accepted_names": [] if over_max else accepted_names,
                    "requested_names": accepted_names + dropped_names,
                    "dropped_names": dropped_names,
                    "unmatched_names": unmatched_values,
                    "max_selector": max_selector,
                    "min_selector": min_selector,
                    "over_max": over_max,
                }
            )

        return feedback

    def prefill_selected_side_variants(
        self,
        *,
        context: ConversationContext,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> None:
        pending = context.pending_add_item
        if pending is None:
            return

        selected_variant_side_choices = []
        for group in pending.side_groups:
            for selected_item_id in context.selected_side_groups.get(group.group_id, []):
                choice = group.choices_by_item_id.get(selected_item_id)
                if choice is None:
                    continue
                if choice.pricing_mode != "variant":
                    continue
                if selected_item_id in context.selected_side_variants:
                    continue
                if not choice.variants:
                    continue
                selected_variant_side_choices.append(choice)

        if len(selected_variant_side_choices) != 1:
            return

        requested_size = self.extract_requested_size(user_text=user_text, slots=slots)
        if not requested_size:
            return

        side_choice = selected_variant_side_choices[0]
        matched_variant = self.match_variant_label(
            requested_size=requested_size,
            pending_variants=side_choice.variants,
        )
        if matched_variant is None:
            return

        context.selected_side_variants[side_choice.item_id] = matched_variant.variant_id

    # ------------------------------------------------------------------
    # Modifier groups
    # ------------------------------------------------------------------

    def prefill_modifier_groups(
        self,
        *,
        context: ConversationContext,
        normalized_user_text: str,
        start_index: int = 0,
    ) -> list[dict]:
        pending = context.pending_add_item
        if pending is None or not pending.modifier_groups:
            return []

        slot_values = extract_modifier_slot_values_normalized(context)
        ignored_values = self._prefill_ignored_modifier_values(context)
        feedback: list[dict] = []

        for group in pending.modifier_groups[start_index:]:
            existing_selections = list(context.selected_modifier_groups.get(group.group_id, []))
            existing_ids = [sel.modifier_id for sel in existing_selections]
            if existing_selections:
                continue

            resolution = self.modifier_resolver.resolve(
                group=group,
                normalized_user_text=normalized_user_text,
                option_candidates=build_modifier_option_candidates(context, normalized_user_text),
                normalized_slot_values=slot_values,
                already_selected_ids=existing_ids,
                ignored_values=ignored_values,
                known_choice_phrases=self.all_modifier_choice_phrases(pending),
            )
            unmatched_values = self._clean_prefill_unmatched_values(
                list(resolution.unmatched_values),
                item_name=pending.item_name,
            )
            min_selector, max_selector = effective_group_selector_bounds(group)
            accepted_limit = max_selector if max_selector > 0 else len(resolution.selections)

            if not resolution.selections and not unmatched_values:
                continue

            capped = resolution.selections[:accepted_limit]
            dropped = resolution.selections[accepted_limit:]
            over_max = bool(dropped)

            if capped and not over_max:
                context.selected_modifier_groups[group.group_id] = capped
                context.skipped_modifier_groups.discard(group.group_id)

            feedback.append(
                {
                    "group_id": group.group_id,
                    "kind": "modifier",
                    "accepted_names": [] if over_max else [sel.name for sel in capped],
                    "requested_names": [sel.name for sel in resolution.selections],
                    "dropped_names": [sel.name for sel in dropped],
                    "unmatched_names": unmatched_values,
                    "max_selector": max_selector,
                    "min_selector": min_selector,
                    "over_max": over_max,
                }
            )

        return feedback

    # ------------------------------------------------------------------
    # Item variant
    # ------------------------------------------------------------------

    def prefill_item_variant(
        self,
        *,
        context: ConversationContext,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> None:
        pending = context.pending_add_item
        if pending is None or not pending.item_variants:
            return

        requested_size = self.extract_requested_size(user_text=user_text, slots=slots)
        if not requested_size:
            return

        matched_variant = self.match_variant_label(
            requested_size=requested_size,
            pending_variants=pending.item_variants,
        )
        if matched_variant is None:
            return

        context.selected_variant_id = matched_variant.variant_id
        context.size_target = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def all_modifier_choice_phrases(pending) -> list[str]:
        phrases: list[str] = []
        seen: set[str] = set()
        for group in pending.modifier_groups:
            for choice in group.choices:
                for value in getattr(choice, "match_texts", ()) or (choice.normalized_name,):
                    if value and value not in seen:
                        seen.add(value)
                        phrases.append(value)
        return phrases

    @staticmethod
    def collect_matched_names(feedback_entries: list[dict]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for entry in feedback_entries:
            for name in entry.get("accepted_names") or []:
                cleaned = str(name).strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                names.append(cleaned)
        return names

    @staticmethod
    def extract_requested_size(
        *,
        user_text: str,
        slots: Sequence[SlotValue],
    ) -> str | None:
        normalized_user_text = user_text or ""

        def _extract_size_via_regex() -> str | None:
            for size in SIZE_WORDS:
                if re.search(rf"\b{re.escape(size)}\b", normalized_user_text):
                    return normalize_text(size)
            return None

        resolution = consume_slot_or_fallback(
            slots=slots,
            slot_labels=("SIZE", "VARIANT"),
            fallback=_extract_size_via_regex,
            parse=lambda raw: normalize_text(raw) or None,
            consumer_site="add_item_handler.size",
        )
        return resolution.value

    @staticmethod
    def match_variant_label(
        *,
        requested_size: str,
        pending_variants,
    ) -> object | None:
        if not requested_size:
            return None

        for variant in pending_variants:
            if variant.normalized_name == requested_size:
                return variant

        for variant in pending_variants:
            if is_strong_token_match(requested_size, variant.normalized_name):
                return variant

        for variant in pending_variants:
            if is_controlled_partial_match(requested_size, variant.normalized_name):
                return variant

        return None

    @staticmethod
    def _clean_prefill_unmatched_values(
        values: list[str],
        *,
        item_name: str,
    ) -> list[str]:
        normalized_item_name = normalize_text(item_name)
        return [
            value
            for value in values
            if value and normalize_text(value) != normalized_item_name
        ]

    @staticmethod
    def _prefill_ignored_modifier_values(context: ConversationContext) -> list[str]:
        pending = context.pending_add_item
        if pending is None:
            return []

        ignored: list[str] = []
        normalized_item_name = normalize_text(pending.item_name)
        if normalized_item_name:
            ignored.append(normalized_item_name)

        for group in pending.side_groups:
            for selected_item_id in context.selected_side_groups.get(group.group_id, []):
                choice = group.choices_by_item_id.get(selected_item_id)
                if choice and choice.normalized_name:
                    ignored.append(choice.normalized_name)
                    ignored.extend(getattr(choice, "match_texts", ()) or ())

        seen: set[str] = set()
        result: list[str] = []
        for value in ignored:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result


# ---------------------------------------------------------------------------
# PrefillOrchestrator
# ---------------------------------------------------------------------------

class PrefillOrchestrator:
    """High-level coordinator for the add-item prefill flow.

    Initialises the PendingAddItem snapshot, runs quantity prefill, runs the
    unified multi-group prefill engine, runs side-variant prefill, builds
    spoken summaries and debug payloads, then delegates to
    ConfirmationDecisionHelper to produce the final HandlerResult.
    """

    def __init__(
        self,
        capture_helper: PendingItemCaptureHelper,
        prefill_engine: MultiGroupPrefillEngine,
        confirmation_helper: ConfirmationDecisionHelper | None = None,
    ) -> None:
        self.capture_helper = capture_helper
        self.prefill_engine = prefill_engine
        self.confirmation_helper = confirmation_helper or ConfirmationDecisionHelper()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def enter_add_flow_for_item(
        self,
        *,
        context: ConversationContext,
        item: MenuItem,
        user_text: str,
        slots: Sequence[SlotValue],
        staged: "object | None" = None,
    ) -> HandlerResult:
        context.current_item_id = item.item_id
        context.current_item_name = item.name
        context.candidate_item_id = item.item_id
        context.pending_add_item = build_pending_add_item(item)

        if staged is not None:
            # ── Structured staged item path ────────────────────────────────────
            # Apply pre-resolved data from StagedItemPlan directly —
            # no NLU re-parse, no prefill engine invocation.
            context.quantity = max(1, int(getattr(staged, "quantity", 1) or 1))
            self._apply_staged_plan(context, staged)
            # Build minimal payloads
            prefilled_summary = self._build_prefilled_summary(context)
            prefill_debug = {
                "staged_plan_applied": True,
                "plan_source": getattr(staged, "plan_source", ""),
            }
            logger.debug(
                "staged_item_prefill_applied",
                extra={
                    "item_name": item.name,
                    "plan_source": getattr(staged, "plan_source", ""),
                },
            )
            return self.confirmation_helper.build_handler_result(
                context=context,
                item=item,
                prefilled_summary=prefilled_summary,
                prefill_feedback="",
                prefill_debug=prefill_debug,
            )
        # ── Normal NLU-based path (unchanged) ─────────────────────────────────

        missing_groups_before_prefill = self._missing_group_names(context)
        prefill_user_text = self._prefill_segment_text_for_item(
            item_name=item.name,
            user_text=user_text,
        )

        # 0) Quantity is still handled separately because it can come from
        #    a leading numeric (e.g. "2 chicken tacos with...") rather than
        #    an option phrase.
        self.capture_helper.prefill_quantity(
            context=context,
            user_text=prefill_user_text,
        )

        # Apply centralized quantity policy using the item-name-stripped text so
        # that vague expressions ("some burgers") are detected before sides/modifiers
        # are captured.  Non-vague missing quantity defaults to 1 immediately so
        # we never route to WAITING_FOR_QUANTITY unnecessarily.
        _qty_policy = normalize_item_quantity(context, user_text=prefill_user_text)
        if not _qty_policy.needs_clarification:
            context.quantity = _qty_policy.quantity
        # For ambiguous/vague: context.quantity stays None; handled after prefill below.

        # 1) Unified, segment-scoped prefill across variants + sides + modifiers.
        prefill_result: PrefillResult = self.prefill_engine.prefill(
            pending=context.pending_add_item,
            segment_text=user_text,
            slots=tuple(slots or ()),
        )
        self._apply_prefill_result(context=context, result=prefill_result)

        # 2) Side sizes are dependent on which side choices were just bound,
        #    so resolve them after the unified pass.
        self.capture_helper.prefill_selected_side_variants(
            context=context,
            user_text=prefill_user_text,
            slots=slots,
        )

        prefilled_summary = self._build_prefilled_summary(context)
        prefill_feedback = self._build_prefill_feedback_summary(
            context,
            list(prefill_result.feedback),
            unresolved_phrases=prefill_result.unresolved_phrases,
        )
        missing_groups_after_prefill = self._missing_group_names(context)
        prefill_debug = self._build_prefill_debug_payload(
            context=context,
            segment_text=user_text,
            missing_groups_before_prefill=missing_groups_before_prefill,
            missing_groups_after_prefill=missing_groups_after_prefill,
            engine_debug=prefill_result.debug,
        )
        logger.debug("pending_item_prefill %s", prefill_debug)

        # Vague quantity ("some burgers", "a few cokes") — sides/modifiers are
        # already captured above; now explicitly route to WAITING_FOR_QUANTITY
        # so that determine_next_add_item_step() never sees a None quantity that
        # it would silently default.
        if not (isinstance(context.quantity, int) and context.quantity > 0):
            context.current_prompt_field = "quantity"
            context.available_choices_kind = None
            context.available_choices_values = ()
            return HandlerResult(
                next_state=ConversationState.WAITING_FOR_QUANTITY,
                response_key="ask_for_quantity",
                response_payload={"item_name": item.name, "prefill_debug": prefill_debug},
            )

        return self.confirmation_helper.build_handler_result(
            context=context,
            item=item,
            prefilled_summary=prefilled_summary,
            prefill_feedback=prefill_feedback,
            prefill_debug=prefill_debug,
        )

    # ------------------------------------------------------------------
    # Prefill application
    # ------------------------------------------------------------------

    def _apply_prefill_result(
        self,
        *,
        context: ConversationContext,
        result: PrefillResult,
    ) -> None:
        if result.variant_id:
            context.selected_variant_id = result.variant_id
            context.size_target = None

        for group_id, item_ids in result.side_selections.items():
            if not item_ids:
                continue
            context.selected_side_groups[group_id] = list(item_ids)
            context.skipped_side_groups.discard(group_id)

        for group_id, selections in result.modifier_selections.items():
            if not selections:
                continue
            context.selected_modifier_groups[group_id] = list(selections)
            context.skipped_modifier_groups.discard(group_id)

    def _apply_staged_plan(
        self,
        context: ConversationContext,
        staged: "object",
    ) -> None:
        """Apply a StagedItemPlan's resolved data directly to the context.

        Resolves variant, sides, and modifiers against the pending_add_item
        snapshot using normalized name matching.  Unresolved entries are
        silently skipped — the flow will ask for the missing requirement.
        """
        from app.state_machine.models.pending_item_models import ModifierSelection

        pending = context.pending_add_item
        if pending is None:
            return

        # Item-level variant / size
        variant_id = getattr(staged, "variant_id", None)
        variant_label = getattr(staged, "variant_label", None)
        if variant_id and variant_id in pending.item_variants_by_id:
            context.selected_variant_id = variant_id
        elif variant_label:
            vlabel = normalize_text(str(variant_label))
            pv = pending.item_variants_by_normalized_name.get(vlabel)
            if pv is not None:
                context.selected_variant_id = pv.variant_id

        # Requested sides
        for staged_side in (getattr(staged, "requested_sides", ()) or ()):
            side_norm = normalize_text(getattr(staged_side, "name", "") or "")
            if not side_norm:
                continue
            for group in pending.side_groups:
                # Try choices_by_normalized_name first (fuzzy-friendly)
                choices = group.choices_by_normalized_name.get(side_norm, [])
                if not choices:
                    # Fall back to exact scan
                    choices = [
                        c for c in group.choices
                        if normalize_text(c.name) == side_norm
                        or side_norm in (normalize_text(t) for t in c.match_texts)
                    ]
                if not choices:
                    continue
                choice = choices[0]
                gid = group.group_id
                if gid not in context.selected_side_groups:
                    context.selected_side_groups[gid] = []
                if choice.item_id not in context.selected_side_groups[gid]:
                    context.selected_side_groups[gid].append(choice.item_id)
                context.skipped_side_groups.discard(gid)

                # Side-level variant (e.g. "small" for "small coke")
                side_vlabel = getattr(staged_side, "variant_label", None)
                if side_vlabel and getattr(choice, "variants_by_normalized_name", None):
                    vlabel = normalize_text(str(side_vlabel))
                    pvs = choice.variants_by_normalized_name.get(vlabel, [])
                    if pvs:
                        context.selected_side_variants[choice.item_id] = pvs[0].variant_id
                break  # side matched — move on

        # Requested modifiers
        for staged_mod in (getattr(staged, "requested_modifiers", ()) or ()):
            mod_norm = normalize_text(getattr(staged_mod, "name", "") or "")
            if not mod_norm:
                continue
            for group in pending.modifier_groups:
                choices = group.choices_by_normalized_name.get(mod_norm, [])
                if not choices:
                    choices = [
                        c for c in group.choices
                        if normalize_text(c.name) == mod_norm
                        or mod_norm in (normalize_text(t) for t in c.match_texts)
                    ]
                if not choices:
                    continue
                choice = choices[0]
                gid = group.group_id
                mod_op = getattr(staged_mod, "operation", "add") or "add"
                sel = ModifierSelection(
                    modifier_id=choice.modifier_id,
                    name=choice.name,
                    action=mod_op if mod_op in ("add", "remove") else "add",
                )
                if gid not in context.selected_modifier_groups:
                    context.selected_modifier_groups[gid] = []
                context.selected_modifier_groups[gid].append(sel)
                context.skipped_modifier_groups.discard(gid)
                break  # modifier matched

    # ------------------------------------------------------------------
    # Missing-group inspection
    # ------------------------------------------------------------------

    def _missing_group_names(self, context: ConversationContext) -> list[str]:
        pending = context.pending_add_item
        if pending is None:
            return []

        missing: list[str] = []
        if pending.item_variants and not context.selected_variant_id:
            missing.append("Size")

        for group in pending.side_groups:
            selected = context.selected_side_groups.get(group.group_id, ())
            skipped = group.group_id in context.skipped_side_groups
            min_selector, _ = effective_group_selector_bounds(group)
            if bool(getattr(group, "is_required", False)):
                if len(selected) < min_selector:
                    missing.append(group.name)
            elif not selected and not skipped:
                missing.append(group.name)

        for group in pending.modifier_groups:
            selections = context.selected_modifier_groups.get(group.group_id, ())
            skipped = group.group_id in context.skipped_modifier_groups
            min_selector, _ = effective_group_selector_bounds(group)
            if bool(getattr(group, "is_required", False)):
                if len(selections) < min_selector:
                    missing.append(group.name)
            elif not selections and not skipped:
                missing.append(group.name)

        if not isinstance(context.quantity, int) or context.quantity <= 0:
            missing.append("Quantity")

        return missing

    # ------------------------------------------------------------------
    # Text normalisation helpers
    # ------------------------------------------------------------------

    def _prefill_segment_text_for_item(self, *, item_name: str, user_text: str) -> str:
        normalized = normalize_item_request_text(user_text)
        item_normalized = normalize_text(item_name or "")
        if not normalized or not item_normalized:
            return normalized
        if normalized.startswith(item_normalized):
            remainder = normalized[len(item_normalized):].strip()
            return remainder or normalized
        return normalized

    # ------------------------------------------------------------------
    # Spoken summary builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prefilled_summary(context: ConversationContext) -> str:
        pending = context.pending_add_item
        if pending is None:
            return ""

        parts: list[str] = []

        if context.selected_variant_id and pending.item_variants_by_id:
            variant = pending.item_variants_by_id.get(context.selected_variant_id)
            if variant:
                parts.append(variant.name)

        for group in pending.side_groups:
            selected_ids = context.selected_side_groups.get(group.group_id, [])
            for sid in selected_ids:
                choice = group.choices_by_item_id.get(sid)
                if choice:
                    side_variant_id = context.selected_side_variants.get(sid)
                    if side_variant_id and choice.variants_by_id:
                        sv = choice.variants_by_id.get(side_variant_id)
                        if sv:
                            parts.append(f"{sv.name} {choice.name}")
                            continue
                    parts.append(choice.name)

        for group in pending.modifier_groups:
            for sel in context.selected_modifier_groups.get(group.group_id, []):
                spoken = _speak_modifier(
                    sel.name,
                    action=sel.action,
                    instruction=sel.instruction,
                )
                if spoken:
                    parts.append(spoken)

        if not parts:
            return ""
        if len(parts) == 1:
            return f"with {parts[0]}"
        if len(parts) == 2:
            return f"with {parts[0]} and {parts[1]}"
        return f"with {', '.join(parts[:-1])}, and {parts[-1]}"

    @staticmethod
    def _format_feedback_names(values: list[str]) -> str:
        clean = [str(value).strip() for value in values if str(value).strip()]
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} and {clean[1]}"
        return f"{', '.join(clean[:-1])}, and {clean[-1]}"

    def _build_prefill_feedback_summary(
        self,
        context: ConversationContext,
        feedback_entries: list[dict],
        *,
        unresolved_phrases: list[str] | None = None,
    ) -> str:
        parts: list[str] = []
        pending = context.pending_add_item
        item_name = pending.item_name if pending else ""
        scope_to_item = bool(getattr(context, "pending_item_queue", None))

        for entry in feedback_entries:
            accepted_names = entry.get("accepted_names") or []
            requested_names = entry.get("requested_names") or []
            dropped_names = entry.get("dropped_names") or []
            max_selector = int(entry.get("max_selector", 0) or 0)

            if dropped_names:
                dropped_text = self._format_feedback_names(requested_names or dropped_names)
                accepted_text = self._format_feedback_names(accepted_names)
                if accepted_text and max_selector > 0:
                    parts.append(
                        f"I heard {accepted_text} and {dropped_text}, but you can only pick {max_selector} there."
                    )
                elif accepted_text:
                    parts.append(f"I heard {accepted_text} and {dropped_text}, so I'll ask you to choose.")
                elif max_selector > 0:
                    parts.append(f"I heard {dropped_text}, but you can only pick {max_selector} there.")
                else:
                    parts.append(f"I heard {dropped_text}, so I'll ask you to choose.")

        cleaned_unresolved = self._collapse_unresolved_for_feedback(
            unresolved_phrases or [],
            pending=pending,
        )
        if cleaned_unresolved:
            unresolved_text = self._format_feedback_names(cleaned_unresolved)
            if scope_to_item and item_name:
                parts.append(f"For the {item_name}, I couldn't find {unresolved_text}.")
            else:
                parts.append(f"I couldn't find {unresolved_text}.")

        return " ".join(parts).strip()

    @staticmethod
    def _collapse_unresolved_for_feedback(
        unresolved_phrases: list[str],
        *,
        pending,
    ) -> list[str]:
        if not unresolved_phrases:
            return []

        # Build exact-match set of item label forms (canonical + aliases +
        # voice labels) so that the compact ASR form ("cheeseburger") is
        # suppressed from "I couldn't find" output when it resolved correctly.
        item_label_exact: set[str] = set()
        if pending is not None:
            _item_norm = normalize_text(getattr(pending, "item_name", "") or "")
            if _item_norm:
                item_label_exact.add(_item_norm)
                item_label_exact.add(_item_norm.replace(" ", ""))
            for alias in (getattr(pending, "item_aliases", ()) or ()):
                _a = normalize_text(alias)
                if _a:
                    item_label_exact.add(_a)
                    item_label_exact.add(_a.replace(" ", ""))
            for vl in (getattr(pending, "item_voice_labels", ()) or ()):
                _v = normalize_text(vl)
                if _v:
                    item_label_exact.add(_v)
                    item_label_exact.add(_v.replace(" ", ""))

        ignored_tokens: set[str] = set()
        if pending is not None:
            ignored_tokens.update(tokenize(normalize_text(getattr(pending, "item_name", "") or "")))
            for group in pending.side_groups:
                for choice in group.choices:
                    ignored_tokens.update(tokenize(choice.normalized_name))
                    for label in (getattr(choice, "match_texts", ()) or ()):
                        ignored_tokens.update(tokenize(label))
            for group in pending.modifier_groups:
                for choice in group.choices:
                    ignored_tokens.update(tokenize(choice.normalized_name))
                    for label in (getattr(choice, "match_texts", ()) or ()):
                        ignored_tokens.update(tokenize(label))
            for variant in pending.item_variants:
                ignored_tokens.update(tokenize(variant.normalized_name))

        ignored_tokens.update(
            {
                "with", "and", "plus", "also", "or",
                "extra", "more", "double", "less", "light",
                "on", "the", "side",
                "a", "an",
            }
        )
        # Merge the canonical shared filler-token set so that residue like
        # "wanted", "needed", "said", "will", "take" etc. is never surfaced
        # as "I couldn't find wanted / needed / said …".
        ignored_tokens.update(ORDER_FILLER_TOKENS)
        canonical_ignored = ignored_tokens | {"no", "without", "hold", "remove"}

        result: list[str] = []
        seen_canonical: set[str] = set()
        seen_phrases: set[str] = set()
        for phrase in unresolved_phrases:
            normalized = normalize_text(phrase or "").strip()
            if not normalized or normalized in seen_phrases:
                continue
            # Suppress item label forms (canonical name / alias / voice label).
            if normalized in item_label_exact:
                continue
            compact = normalized.replace(" ", "")
            if compact and compact in item_label_exact:
                continue
            canonical = " ".join(
                t for t in tokenize(normalized) if t not in canonical_ignored
            )
            if not canonical or canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            seen_phrases.add(normalized)
            result.append(normalized)
        return result

    # ------------------------------------------------------------------
    # Debug payload builders
    # ------------------------------------------------------------------

    def _build_prefill_debug_payload(
        self,
        *,
        context: ConversationContext,
        segment_text: str,
        missing_groups_before_prefill: list[str],
        missing_groups_after_prefill: list[str],
        engine_debug: dict | None = None,
    ) -> dict[str, object]:
        resolved_group_values = self._resolved_group_values(context)
        pending = context.pending_add_item
        candidate_text = segment_text
        if pending is not None:
            candidate_text = self._prefill_segment_text_for_item(
                item_name=pending.item_name,
                user_text=segment_text,
            )
        engine_debug = engine_debug or {}
        candidate_phrases = (
            engine_debug.get("candidate_phrases")
            or self._collect_prefill_candidate_phrases(
                context=context,
                segment_text=candidate_text,
            )
        )
        return {
            "segment_text": segment_text,
            "segment_text_after_item": engine_debug.get("segment_text_after_item", candidate_text),
            "candidate_phrases": list(candidate_phrases),
            "resolved_group_values": resolved_group_values,
            "pending_item_prefill_before_missing_groups": resolved_group_values,
            "missing_groups_before_prefill": missing_groups_before_prefill,
            "missing_groups_after_prefill": missing_groups_after_prefill,
            "skipped_groups_because_prefilled": [
                group_name
                for group_name in missing_groups_before_prefill
                if group_name not in missing_groups_after_prefill
            ],
            "bindings": engine_debug.get("bindings", []),
        }

    def _collect_prefill_candidate_phrases(
        self,
        *,
        context: ConversationContext,
        segment_text: str,
    ) -> list[str]:
        pending = context.pending_add_item
        if pending is None:
            return []

        candidates: list[str] = []
        seen: set[str] = set()

        def add(value: str | None) -> None:
            normalized = normalize_text(value or "")
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        for candidate in build_side_option_candidates(context, segment_text):
            add(candidate.text)
        for candidate in build_modifier_option_candidates(context, segment_text):
            add(candidate.text)

        known_phrases: list[str] = []
        for group in pending.side_groups:
            for choice in group.choices:
                known_phrases.extend(getattr(choice, "match_texts", ()) or (choice.normalized_name,))
        for group in pending.modifier_groups:
            for choice in group.choices:
                known_phrases.extend(getattr(choice, "match_texts", ()) or (choice.normalized_name,))
        known_phrases.extend(variant.normalized_name for variant in pending.item_variants)

        for candidate in build_scoped_phrase_candidates(
            raw_utterance=segment_text,
            phrases=known_phrases,
        ):
            add(candidate.text)

        return candidates

    @staticmethod
    def _resolved_group_values(context: ConversationContext) -> dict[str, list[str]]:
        pending = context.pending_add_item
        if pending is None:
            return {}

        resolved: dict[str, list[str]] = {}
        if context.selected_variant_id and context.selected_variant_id in pending.item_variants_by_id:
            resolved["Size"] = [pending.item_variants_by_id[context.selected_variant_id].name]

        for group in pending.side_groups:
            selected_ids = context.selected_side_groups.get(group.group_id, [])
            if not selected_ids:
                continue
            resolved[group.name] = [
                group.choices_by_item_id[item_id].name
                for item_id in selected_ids
                if item_id in group.choices_by_item_id
            ]

        for group in pending.modifier_groups:
            selections = context.selected_modifier_groups.get(group.group_id, [])
            if not selections:
                continue
            resolved[group.name] = [selection.name for selection in selections]

        if isinstance(context.quantity, int) and context.quantity > 0:
            resolved["Quantity"] = [str(context.quantity)]

        return resolved
