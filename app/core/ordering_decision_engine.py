# app/core/ordering_decision_engine.py
"""Pure decision engine for menu-item ordering flows.

Responsibilities
----------------
* Translate a ``MenuQueryResult`` + context state into an ``OrderingDecision``.
* Encapsulate all confidence-tier routing (EXACT / HIGH / MEDIUM / LOW /
  ESCALATION) in one place.
* Provide a read-only view of the post-add idle checkout decision so that
  FlowGate can be migrated incrementally in a future pass.

Design principles
-----------------
* **Pure input/output** — no side effects on ConversationContext.  Handlers
  call context.bump_not_found() *after* receiving the decision.
* **No response key coupling** — the engine does not know response keys.
  It only decides *what* to do; handlers decide *how* to phrase it.
* **No FSM state mutation** — next_state hints are advisory; the handler
  writes ConversationState.
* All dependencies are passed via constructor (testable without live menu).
"""
from __future__ import annotations

from typing import Sequence

from app.contracts.ordering_decision import (
    CandidateDecision,
    OrderingAction,
    OrderingDecision,
)
from app.menu.query_result import MenuQueryResult, MenuQueryType
from app.menu.repository import MenuRepository
from app.nlu.intent_resolution.intent import Intent
from app.nlu.nlu_result import SlotValue
from app.nlu.query_normalization.text_preprocessor import normalize_text
from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.conversation_state import ConversationState

# Intents that, from IDLE with a non-empty cart, should route to checkout.
_CHECKOUT_LIKE_INTENTS: frozenset[Intent] = frozenset(
    {
        Intent.DENY,
        Intent.END_ADDING,
        Intent.START_ORDER,
        Intent.CHECKOUT,
        Intent.CONFIRM_ORDER,
        Intent.FINISH_ORDER,
        Intent.PAYMENT_REQUEST,
        Intent.REVIEW_ORDER,
    }
)


class OrderingDecisionEngine:
    """Central decision layer for menu-item ordering.

    Currently wired to:
      * add-item resolution path (via ItemResolutionHandler)

    Pending migration (see migration notes at bottom of module):
      * post-add idle checkout shortcut (currently in FlowGate._apply_idle_shortcuts)
      * quantity resolution (currently split across PrefillOrchestrator and
        WaitingForQuantityHandler)
    """

    def __init__(self, menu_repo: MenuRepository) -> None:
        self._menu_repo = menu_repo

    # ------------------------------------------------------------------
    # Primary API: item resolution
    # ------------------------------------------------------------------

    def decide_from_menu_result(
        self,
        *,
        result: MenuQueryResult,
        requested_item_text: str,
        context: ConversationContext,
        slots: Sequence[SlotValue],
    ) -> OrderingDecision:
        """Map a ``MenuQueryResult`` to the correct ``OrderingDecision``.

        This method is **pure**: it reads ``context`` for the not-found
        attempt count but never mutates it.  The caller is responsible for
        calling ``context.bump_not_found()`` once the decision has been acted on.

        Decision rules (in priority order)
        ------------------------------------
        1. ITEM match → ADD_ITEM (exact)
        2. CATEGORY_SINGLE_ITEM (one item in category) → ADD_ITEM
        3. CATEGORY → ASK_ITEM_CONFIRMATION (category_detected)
        4. AMBIGUOUS → ASK_ITEM_CONFIRMATION (multiple_matches)
        5. NOT_FOUND, attempt ≥ 3 → ESCALATE_TO_AGENT
        6. NOT_FOUND + near-miss (score ≥ NEAR_MISS_THRESHOLD) → ASK_ITEM_CONFIRMATION (HIGH/MEDIUM tier)
        7. NOT_FOUND, no near-miss → SUGGEST_ALTERNATIVES (LOW)
        """
        rtype = result.type

        # ── Rule 1: exact single-item match ──────────────────────────────
        if rtype == MenuQueryType.ITEM and result.item is not None:
            return OrderingDecision(
                action=OrderingAction.ADD_ITEM,
                reason="exact_match",
                item_id=result.item.item_id,
                item_name=result.item.name,
            )

        # ── Rule 2: category resolves to exactly one item ─────────────────
        if (
            rtype == MenuQueryType.CATEGORY_SINGLE_ITEM
            and result.items
            and len(result.items) == 1
        ):
            return OrderingDecision(
                action=OrderingAction.ADD_ITEM,
                reason="category_single_item",
                item_id=result.items[0].item_id,
                item_name=result.items[0].name,
            )

        # ── Rule 3: category with multiple items → ask which one ──────────
        if rtype == MenuQueryType.CATEGORY:
            candidate_items = result.items or []
            return OrderingDecision(
                action=OrderingAction.ASK_ITEM_CONFIRMATION,
                reason="category_detected",
                category_id=result.category_id,
                category_name=result.category_name,
                query=requested_item_text,
                candidates=tuple(
                    CandidateDecision(item_id=item.item_id, item_name=item.name)
                    for item in candidate_items
                ),
                requires_confirmation=True,
            )

        # ── Rule 4: ambiguous (multiple items / categories matched) ───────
        if rtype in {MenuQueryType.ITEM_AMBIGUOUS, MenuQueryType.CATEGORY_AMBIGUOUS}:
            item_candidates = tuple(
                CandidateDecision(item_id=item.item_id, item_name=item.name)
                for item in (result.matched_items or [])
            )
            matched_cat_names = tuple(
                c.get("name", "")
                for c in (result.matched_categories or [])
                if c.get("name")
            )
            return OrderingDecision(
                action=OrderingAction.ASK_ITEM_CONFIRMATION,
                reason="multiple_matches",
                query=requested_item_text,
                candidates=item_candidates,
                matched_category_names=matched_cat_names,
                requires_confirmation=True,
            )

        # ── Rules 5-7: NOT_FOUND confidence-tiered routing ────────────────
        return self._decide_not_found(
            result=result,
            requested_item_text=requested_item_text,
            context=context,
        )

    # ------------------------------------------------------------------
    # Secondary API: post-add idle checkout (not yet wired to FlowGate)
    # ------------------------------------------------------------------

    def decide_from_idle_completion(
        self,
        *,
        intent: Intent,
        cart_is_empty: bool,
        last_response_key: str | None = None,
        raw_text: str | None = None,
    ) -> OrderingDecision | None:
        """Decide whether an IDLE-state intent triggers checkout routing.

        Returns ``None`` if the intent is unrelated to checkout.  Returns
        an ``OrderingDecision`` with action CHECKOUT or UNCLEAR otherwise.

        **Not yet wired** — see migration notes at bottom of this module.
        FlowGate._apply_idle_shortcuts() still owns this path.  Call this
        method to get the equivalent decision without side-effects; wire it
        once FlowGate tests are in place.
        """
        if intent not in _CHECKOUT_LIKE_INTENTS:
            return None

        if not cart_is_empty:
            return OrderingDecision(
                action=OrderingAction.CHECKOUT,
                reason="cart_non_empty_done_signal",
            )

        return OrderingDecision(
            action=OrderingAction.UNCLEAR,
            reason="cart_empty_checkout_intent",
            handler_result_hint="idle_nothing_to_checkout",
        )

    # ------------------------------------------------------------------
    # Internal: NOT_FOUND confidence routing
    # ------------------------------------------------------------------

    def _decide_not_found(
        self,
        *,
        result: MenuQueryResult,
        requested_item_text: str,
        context: ConversationContext,
    ) -> OrderingDecision:
        normalized_query = normalize_text(requested_item_text)

        # Predict the post-bump attempt count (engine never mutates context).
        predicted_attempt = context.not_found_count(normalized_query) + 1

        # ── Rule 5: escalate after repeated failures ──────────────────────
        if predicted_attempt >= 3:
            return OrderingDecision(
                action=OrderingAction.ESCALATE_TO_AGENT,
                reason="repeated_not_found",
                query=requested_item_text,
                tier="LOW",
                attempt=predicted_attempt,
            )

        # ── Build suggestion list ─────────────────────────────────────────
        # Prefer same-category items so "veggie pizza" gets pizza alternatives
        # rather than unrelated scored items.  Fall back to the result's
        # general scored suggestions when no category can be inferred.
        category_items = self._find_category_suggestions(normalized_query)

        if category_items:
            base_list = category_items
        else:
            all_items = [item.name for item in (result.suggested_items or [])]
            all_cats = [
                c.get("name")
                for c in (result.suggested_categories or [])
                if c.get("name")
            ]
            base_list = all_items if all_items else all_cats

        # Rotate on attempt 2 for variety (avoids identical repeat).
        if predicted_attempt == 2 and len(base_list) > 1:
            suggestions = tuple(base_list[1:5])
        else:
            suggestions = tuple(base_list[:4])

        # ── Rule 6: near-miss → confirmation prompt ───────────────────────
        near_miss = self._menu_repo.find_near_miss_item_normalized(normalized_query)
        if near_miss is not None:
            return OrderingDecision(
                action=OrderingAction.ASK_ITEM_CONFIRMATION,
                reason="near_miss_suggestion",
                item_id=near_miss.item.item_id,
                item_name=near_miss.item.name,
                query=requested_item_text,
                suggestions=suggestions,
                tier=near_miss.tier,
                attempt=predicted_attempt,
                requires_confirmation=True,
            )

        # ── Rule 7: low confidence → suggest alternatives ─────────────────
        return OrderingDecision(
            action=OrderingAction.SUGGEST_ALTERNATIVES,
            reason="no_match",
            query=requested_item_text,
            suggestions=suggestions,
            tier="LOW",
            attempt=predicted_attempt,
        )

    def _find_category_suggestions(
        self,
        normalized_query: str,
        limit: int = 4,
    ) -> list[str]:
        """Return item names from the best-matching category for the query.

        Tries the full query first, then individual long words (e.g. "pizza"
        from "veggie pizza").  Returns [] when no category can be inferred.
        Pure — no side effects.
        """
        if not normalized_query:
            return []

        cat_result = self._menu_repo.resolve_category_query_normalized(
            normalized_query, limit=limit
        )
        if cat_result is not None and cat_result.items:
            return [item.name for item in cat_result.items if item.available][:limit]

        # Try individual words from longest to shortest; skip stop words and
        # single-character tokens.
        _STOP = frozenset({"a", "an", "the", "with", "and", "or", "for", "on", "in",
                           "my", "i", "of", "no", "not"})
        words = sorted(
            [w for w in normalized_query.split() if len(w) > 2 and w not in _STOP],
            key=len,
            reverse=True,
        )
        for word in words:
            cat_result = self._menu_repo.resolve_category_query_normalized(
                word, limit=limit
            )
            if cat_result is not None and cat_result.items:
                return [item.name for item in cat_result.items if item.available][:limit]

        return []


# ------------------------------------------------------------------
# Migration notes
# ------------------------------------------------------------------
#
# PASS 2 (pending):
#   FlowGate._apply_idle_shortcuts() → wire to decide_from_idle_completion().
#   Requires FlowGate tests to be written first so the rewrite is safe.
#
# PASS 3 (pending):
#   Quantity resolution unification.  Three independent extraction pipelines:
#     1. PrefillOrchestrator._infer_quantity_from_text()
#     2. WaitingForQuantityHandler._extract_quantity_from_context_or_text()
#     3. (implicit) slot-first fallback in both
#   These should be merged into a single QuantityDecision produced by this
#   engine, with handlers consuming the result.
#
# PASS 4 (pending):
#   _signal_candidates() deduplication between ControlIntentResolver and
#   ConfirmationResolver — extract to a shared utility.
