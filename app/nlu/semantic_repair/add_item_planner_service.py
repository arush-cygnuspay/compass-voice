# app/nlu/semantic_repair/add_item_planner_service.py
# TODO(priority-2-migration): Migrate GPT call site here to GptSafeClient
#   for circuit-breaker protection and structured GptSafeResult failure handling.
"""Phase 4 GPT Add-Item Planner Service.

Orchestrates the full planner pipeline for complex add-item utterances:
  routing → context building → GPT call → Phase 4 output parsing →
  AddItemPlanValidator → apply gate → AddItemPlannerResult.

Shadow-only and inline contracts
----------------------------------
* SHADOW mode  — GPT is called; result returned for logging only.
  safe_to_apply is always False.
* INLINE mode  — GPT is called; apply gate marks safe_to_apply=True only
  when all gate conditions pass (validator, confidence, no timeout, etc.).
* The handler decides whether to apply the plan based on safe_to_apply.

Safety invariants
------------------
* API key is read from the environment at call time; never stored or logged.
* Payload contains only candidate items from the utterance — not the full menu.
* Cart, prices, PII are never included in the payload.
* run() never raises — all exceptions produce a safe AddItemPlannerResult.
* The result is NEVER applied by this service; the handler owns application.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from app.nlu.semantic_repair.add_item_plan_validator import (
    AddItemPlanValidator,
    PlannerApplyGate,
)
from app.nlu.semantic_repair.add_item_planner_context_builder import (
    GptAddItemPlannerContextBuilder,
)
from app.nlu.semantic_repair.add_item_planner_output_parser import parse_planner_output
from app.nlu.semantic_repair.add_item_planner_result import (
    ADD_ITEM_PLANNER_NOT_CALLED,
    AddItemPlannerResult,
)
from app.nlu.semantic_repair.add_item_planner_routing_policy import (
    AddItemPlannerRouteMode,
    GptAddItemPlannerRoutingPolicy,
)
from app.nlu.semantic_repair.daily_budget import GptDailyBudget

if TYPE_CHECKING:
    from app.config.semantic_repair import SemanticRepairConfig

# Maximum tokens for the planner response (longer than option resolver —
# we need item + modifier + side arrays).
_MAX_TOKENS = 512

# Terminal states that should never trigger the planner.
_TERMINAL_STATES: frozenset[str] = frozenset({
    "COMPLETED",
    "TRANSFERRING_TO_HUMAN_AGENT",
    "ERROR_RECOVERY",
})


class GptAddItemPlannerService:
    """Inline/shadow GPT service for planning complex add-item utterances.

    Instantiate once (e.g. in AddItemHandler.__init__) and reuse across turns.
    The OpenAI client is created lazily on first use.

    run() always returns an AddItemPlannerResult — never raises.
    """

    def __init__(
        self,
        config: "SemanticRepairConfig | None" = None,
        *,
        mock_client: Any = None,
    ) -> None:
        from app.config.semantic_repair import get_semantic_repair_config

        self._config: SemanticRepairConfig = config or get_semantic_repair_config()
        self._routing_policy = GptAddItemPlannerRoutingPolicy()
        self._context_builder = GptAddItemPlannerContextBuilder()
        self._validator = AddItemPlanValidator()
        self._apply_gate = PlannerApplyGate()
        self._daily_budget = GptDailyBudget(limit=self._config.daily_budget)
        # mock_client is used in tests / replay harness to avoid real OpenAI calls.
        self._client: Any = mock_client  # None = lazy-init on first real call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        user_text: str,
        local_intent: str | None = None,
        local_confidence: float = 0.0,
        local_slots: list[dict] | None = None,
        top_k_intents: list[dict] | None = None,
        candidate_items: list[dict] | None = None,
        cart_item_names: list[str] | None = None,
        previous_turns: list[tuple[str, str]] | None = None,
        state: str | None = None,
        menu_store: Any = None,
        menu_repo: Any = None,
    ) -> AddItemPlannerResult:
        """Run the add-item planner for the current turn.

        Parameters
        ----------
        user_text:
            Normalized customer utterance.
        local_intent:
            Local NLU intent label (e.g. "add_item", "UNKNOWN").
        local_confidence:
            Local NLU confidence (0.0–1.0).
        local_slots:
            Local NLU slot list; each {"n": slot_name, "v": value}.
        top_k_intents:
            Top-K local NLU candidates; each {"i": label, "c": conf}.
        candidate_items:
            Pre-resolved menu item candidates.  If None, the service attempts
            to build them from menu_store using local_slots (ITEM slot value).
        cart_item_names:
            Current cart item names (compact list — no prices).
        previous_turns:
            Recent (role, text) pairs for conversation context.
        state:
            Current FSM state value (string).  Terminal states are skipped.
        menu_store:
            MenuStore instance for validator + candidate building (optional).
        menu_repo:
            MenuRepository instance (alternative to menu_store for validator).

        Returns
        -------
        AddItemPlannerResult — never raises.
        """
        cfg = self._config
        normalized_text = (user_text or "").strip()

        # ── 0. Terminal state guard ──────────────────────────────────────
        if state and state.upper() in _TERMINAL_STATES:
            return AddItemPlannerResult(
                decision="skipped",
                route_mode="no_gpt",
                route_reason="terminal_state",
                skipped_reason="terminal_state",
            )

        # ── 1. Routing policy ─────────────────────────────────────────────
        item_candidates_exist = bool(candidate_items)
        route, reason = self._routing_policy.decide(
            config=cfg,
            user_text=normalized_text,
            local_intent=local_intent,
            local_confidence=local_confidence,
            local_slots=local_slots,
            item_candidates_exist=item_candidates_exist,
        )

        if route == AddItemPlannerRouteMode.NO_GPT:
            return AddItemPlannerResult(
                decision="skipped",
                route_mode=route.value,
                route_reason=reason,
                skipped_reason=reason,
            )

        # ── 2. Daily budget ───────────────────────────────────────────────
        if not self._daily_budget.try_consume():
            return AddItemPlannerResult(
                decision="skipped",
                route_mode=route.value,
                route_reason=reason,
                skipped_reason="daily_budget_exceeded",
            )

        # ── 3. API key ────────────────────────────────────────────────────
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return AddItemPlannerResult(
                decision="skipped",
                route_mode=route.value,
                route_reason=reason,
                skipped_reason="missing_api_key",
            )

        # ── 4. Build candidate items if not provided ──────────────────────
        resolved_candidates = candidate_items or []
        if not resolved_candidates and (menu_store is not None or menu_repo is not None):
            resolved_candidates = self._build_candidates_from_slots(
                local_slots=local_slots or [],
                menu_store=menu_store,
                menu_repo=menu_repo,
                max_candidates=cfg.add_item_planner_max_item_candidates,
                max_options=cfg.add_item_planner_max_option_candidates,
            )

        # ── 5. Build payload ──────────────────────────────────────────────
        messages = self._context_builder.build_messages(
            user_text=normalized_text,
            local_intent=local_intent,
            local_confidence=local_confidence,
            top_k_intents=top_k_intents,
            local_slots=local_slots,
            candidate_items=resolved_candidates or None,
            cart_item_names=cart_item_names,
            previous_turns=previous_turns,
        )
        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        # ── 6. GPT call ───────────────────────────────────────────────────
        client = self._get_client()
        if client is None:
            return AddItemPlannerResult(
                decision="skipped",
                route_mode=route.value,
                route_reason=reason,
                skipped_reason="missing_api_key",
            )

        timeout_ms = int(getattr(cfg, "add_item_planner_timeout_ms", 1800))
        timeout_s = timeout_ms / 1000.0
        model = cfg.model
        t_start = time.perf_counter()

        raw_response: str = ""
        completion_chars: int = 0
        timed_out: bool = False
        call_error: str | None = None

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=_MAX_TOKENS,
                temperature=0.0,
                timeout=timeout_s,
            )
            raw_response = response.choices[0].message.content or ""
            completion_chars = len(raw_response)
        except Exception as exc:
            exc_name = type(exc).__name__.lower()
            timed_out = "timeout" in exc_name or "timedout" in exc_name
            call_error = f"{type(exc).__name__}: {exc}"[:200]

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if call_error and not raw_response:
            return AddItemPlannerResult(
                decision="error",
                gpt_called=True,
                route_mode=route.value,
                route_reason=reason,
                parse_error=call_error,
                latency_ms=latency_ms,
                model=model,
                prompt_chars=prompt_chars,
            )

        # ── 7. Parse output ───────────────────────────────────────────────
        # Build name sets for hallucination guard
        cand_names: set[str] = {
            (c.get("name") or "").lower()
            for c in resolved_candidates
            if c.get("name")
        }
        cand_option_names: set[str] = set()
        for c in resolved_candidates:
            for mg in (c.get("modifier_groups") or []):
                cand_option_names.update(n.lower() for n in (mg.get("choices") or []))
            for sg in (c.get("side_groups") or []):
                cand_option_names.update(n.lower() for n in (sg.get("choices") or []))

        decision, items, unresolved, confidence, reason_code, parse_error = parse_planner_output(
            raw=raw_response,
            utterance_text=normalized_text,
            candidate_names=cand_names,
            candidate_option_names=cand_option_names,
            max_items=8,
        )

        if parse_error and not items:
            return AddItemPlannerResult(
                decision="error",
                gpt_called=True,
                route_mode=route.value,
                route_reason=reason,
                parse_error=parse_error,
                latency_ms=latency_ms,
                model=model,
                prompt_chars=prompt_chars,
                completion_chars=completion_chars,
            )

        # ── 8. Validate against live menu ─────────────────────────────────
        validated_plan = None
        validator_passed = False
        validator_reject_reason: str | None = None

        if items and (menu_store is not None or menu_repo is not None):
            try:
                validated_plan = self._validator.validate_planner_items(
                    planner_items=items,
                    menu_store=menu_store,
                    menu_repo=menu_repo,
                )
                if validated_plan is not None:
                    validator_passed = (
                        not validated_plan.has_blocking_warnings
                        and bool(validated_plan.items)
                    )
                    if not validator_passed:
                        if validated_plan.rejected_items:
                            validator_reject_reason = f"rejected_items:{validated_plan.rejected_items[:3]}"
                        elif validated_plan.has_blocking_warnings:
                            validator_reject_reason = "blocking_warnings"
                        else:
                            validator_reject_reason = "no_valid_items"
            except Exception:
                validator_reject_reason = "validator_exception"

        # ── 9. Apply gate ─────────────────────────────────────────────────
        min_conf = float(getattr(cfg, "add_item_planner_min_confidence", 0.75))
        safe_to_apply, gate_reason = self._apply_gate.should_apply(
            route_mode=route.value,
            decision=decision,
            validated_plan=validated_plan,
            confidence=confidence,
            min_confidence=min_conf,
            parse_error=parse_error,
            gpt_called=True,
            timed_out=timed_out,
        )
        if not safe_to_apply and validator_reject_reason is None:
            validator_reject_reason = gate_reason

        return AddItemPlannerResult(
            decision=decision,
            gpt_called=True,
            route_mode=route.value,
            route_reason=reason,
            items=items,
            unresolved=unresolved,
            confidence=confidence,
            reason_code=reason_code,
            validated_plan=validated_plan,
            validator_passed=validator_passed,
            validator_reject_reason=validator_reject_reason,
            safe_to_apply=safe_to_apply,
            parse_error=parse_error,
            latency_ms=latency_ms,
            model=model,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_candidates_from_slots(
        self,
        *,
        local_slots: list[dict],
        menu_store: Any,
        menu_repo: Any,
        max_candidates: int,
        max_options: int,
    ) -> list[dict]:
        """Build candidate item dicts from ITEM slots using the menu store."""
        store = None
        if menu_store is not None:
            store = menu_store
        elif menu_repo is not None:
            store = getattr(menu_repo, "store", None)
        if store is None:
            return []

        candidates: list[dict] = []
        seen_ids: set[str] = set()

        for slot in local_slots:
            if slot.get("n", "").upper() not in {"ITEM", "MENU_ITEM"}:
                continue
            value = (slot.get("v") or "").strip()
            if not value:
                continue
            try:
                from app.nlu.query_normalization.text_preprocessor import normalize_text
                norm = normalize_text(value)
                item = store.find_item_exact(norm)
                if item is None:
                    alias_ids = store.find_item_ids_by_alias(norm) or []
                    vl_ids = store.find_item_ids_by_voice_label(norm) or []
                    all_ids = alias_ids or vl_ids
                    if all_ids:
                        item = store.items.get(all_ids[0])
                if item is not None:
                    iid = getattr(item, "item_id", "") or ""
                    if iid and iid not in seen_ids:
                        seen_ids.add(iid)
                        candidates.append(
                            self._context_builder.build_candidate_from_menu_item(
                                item, max_options=max_options
                            )
                        )
                        if len(candidates) >= max_candidates:
                            break
            except Exception:
                continue

        return candidates

    def _get_client(self) -> Any:
        """Lazily create the OpenAI client. Returns None on failure."""
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None
        try:
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        except Exception:
            return None
        return self._client
