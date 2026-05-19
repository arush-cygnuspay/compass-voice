# app/services/smart_turn_planner.py
# TODO(priority-2-migration): Migrate GPT call site here to GptSafeClient
#   for circuit-breaker protection and structured GptSafeResult failure handling.
"""SmartTurnPlanner — surgical GPT layer for risky customer turns.

Invoked ONLY for turns that the local FSM is likely to handle poorly:
  - Compound add-item utterances ("burger with mozzarella and a coke")
  - Self-correction phrases ("no I said X", "actually", "scratch that")
  - Off-menu / unknown item in WAITING_FOR_MODIFIER or WAITING_FOR_SIDE
  - Low local-NLU confidence on actionable states

Safety invariants
-----------------
* plan_smart_turn() NEVER raises — always returns a SmartTurnPlan | None.
* The plan is NEVER applied by this service — the calling handler owns that.
* The plan does NOT mutate cart, session, FSM state, or response text.
* API key is read from the environment at call time; never stored or logged.
* Full menu, cart prices, PII are never sent to the LLM.
* On timeout or invalid JSON the function returns None; the local path runs.

Feature flag
------------
Controlled by SMART_TURN_PLANNER_ENABLED (default: false).
When disabled the module still imports cleanly — all public functions return
None / (False, "disabled") immediately.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

from app.nlu.semantic_repair.daily_budget import GptDailyBudget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-flag helpers
# ---------------------------------------------------------------------------

def _is_enabled() -> bool:
    """Return True when SMART_TURN_PLANNER_ENABLED is set to a truthy value."""
    return os.getenv("SMART_TURN_PLANNER_ENABLED", "false").lower() in {
        "1", "true", "yes", "on",
    }


def _timeout_ms() -> int:
    return int(os.getenv("SMART_TURN_PLANNER_TIMEOUT_MS", "1200"))


def _daily_budget() -> int:
    return int(os.getenv("SMART_TURN_PLANNER_DAILY_BUDGET", "5000"))


# ---------------------------------------------------------------------------
# Plan schema dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmartTurnModifier:
    """A modifier resolved from the user utterance."""

    name: str
    operation: str = "add"   # "add" | "remove" | "extra" | "light"


@dataclass(frozen=True, slots=True)
class SmartTurnSide:
    """A side dish resolved from the user utterance."""

    name: str
    quantity: int = 1
    size: str | None = None      # e.g. "small", "large"
    variant: str | None = None   # e.g. "6 piece"


@dataclass(frozen=True, slots=True)
class SmartTurnItem:
    """One item extracted from a compound utterance."""

    item_name: str
    quantity: int = 1
    size: str | None = None
    variant: str | None = None
    modifiers: tuple[SmartTurnModifier, ...] = ()
    sides: tuple[SmartTurnSide, ...] = ()


@dataclass(frozen=True, slots=True)
class SmartTurnCorrection:
    """A self-correction extracted from the utterance."""

    original_text: str    # what the user had said / what we had captured
    corrected_text: str   # what the user actually means


@dataclass(frozen=True, slots=True)
class SmartTurnPlan:
    """The fully-parsed output of one plan_smart_turn() call.

    Fields
    ------
    decision:
        "add_items"   — items[] is populated; apply to ADD_ITEM flow.
        "correction"  — correction field is set; re-interpret user intent.
        "clarify"     — ambiguous; ask the user a follow-up question.
        "no_action"   — GPT says local flow should handle this normally.
        "unclear"     — GPT could not determine intent.
    confidence:
        0.0–1.0, GPT self-reported.  Informational only.
    items:
        Parsed items (possibly empty even for add_items if GPT failed).
    correction:
        Set when decision == "correction".
    response:
        Optional suggested clarification text (for "clarify" decision only).
        Never used to override a live customer-facing response — only
        logged for training data.
    reason:
        Short English phrase explaining the GPT decision.
    trigger_reason:
        Why the SmartTurnPlanner was invoked (from the policy layer).
    latency_ms:
        Wall-clock time from GPT request to parsed response.
    gpt_called:
        True when an OpenAI call was actually made.
    skipped_reason:
        Set when gpt_called=False — reason why GPT was not called.
    model:
        OpenAI model identifier used.
    """

    decision: str = "no_action"
    confidence: float = 0.0
    items: tuple[SmartTurnItem, ...] = ()
    correction: SmartTurnCorrection | None = None
    response: str | None = None
    reason: str = ""
    trigger_reason: str = ""
    latency_ms: float | None = None
    gpt_called: bool = False
    skipped_reason: str | None = None
    model: str | None = None
    unresolved_spans: tuple[str, ...] = ()   # spans that could not be matched

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for structured logging."""
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "items": [
                {
                    "item_name": it.item_name,
                    "quantity": it.quantity,
                    "size": it.size,
                    "variant": it.variant,
                    "modifiers": [
                        {"name": m.name, "operation": m.operation}
                        for m in it.modifiers
                    ],
                    "sides": [
                        {"name": s.name, "quantity": s.quantity, "size": s.size, "variant": s.variant}
                        for s in it.sides
                    ],
                }
                for it in self.items
            ],
            "correction": (
                {
                    "original_text": self.correction.original_text,
                    "corrected_text": self.correction.corrected_text,
                }
                if self.correction
                else None
            ),
            "response": self.response,
            "reason": self.reason,
            "trigger_reason": self.trigger_reason,
            "latency_ms": self.latency_ms,
            "gpt_called": self.gpt_called,
            "skipped_reason": self.skipped_reason,
            "model": self.model,
            "unresolved_spans": list(self.unresolved_spans),
        }


# ---------------------------------------------------------------------------
# Sentinel: returned by callers when no plan was produced
# ---------------------------------------------------------------------------

SMART_TURN_NOT_CALLED = SmartTurnPlan(
    decision="no_action",
    gpt_called=False,
    skipped_reason="not_called",
    reason="smart_planner_not_invoked",
)

# ---------------------------------------------------------------------------
# Module-level budget (one counter per process)
# ---------------------------------------------------------------------------

_budget_lock = threading.Lock()
_budget: GptDailyBudget | None = None


def _get_budget() -> GptDailyBudget:
    global _budget  # noqa: PLW0603
    with _budget_lock:
        if _budget is None:
            _budget = GptDailyBudget(limit=_daily_budget())
    return _budget


# ---------------------------------------------------------------------------
# GPT prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a restaurant voice-ordering semantic parser. Your job is to extract
structured intent from a single customer utterance in a phone-ordering call.

Output ONLY valid JSON. No prose, no markdown, no explanation outside the JSON.

JSON schema:
{
  "decision": "add_items" | "correction" | "clarify" | "no_action" | "unclear",
  "confidence": 0.0-1.0,
  "items": [
    {
      "item_name": "string",
      "quantity": 1,
      "size": null,
      "variant": null,
      "modifiers": [{"name": "string", "operation": "add|remove|extra|light"}],
      "sides": [{"name": "string", "quantity": 1, "size": null, "variant": null}]
    }
  ],
  "correction": {"original_text": "string", "corrected_text": "string"} | null,
  "response": null,
  "reason": "short English phrase",
  "unresolved_spans": []
}

Hard rules:
- items[] is non-empty ONLY for "add_items" decision.
- correction is set ONLY for "correction" decision.
- response may be set ONLY for "clarify" — keep it under 20 words, no prices.
- NEVER invent menu items, sides, or modifiers not present in the utterance
  or allowed_options list.
- If allowed_options is provided, item/modifier/side names MUST come from that
  list (case-insensitive fuzzy match is allowed, but hallucination is not).
- quantity must be a positive integer clamped to [1, 20]; default 1.
- Do NOT include prices, tax, payment info, or PII in any field.
- Prefer "clarify" over a low-confidence "add_items" guess.
- Use previous_turns context to resolve pronouns and corrections.
"""

# Task-mode specific guidance appended to the user message.
_TASK_MODE_HINTS: dict[str, str] = {
    "compound_add_item": (
        "Task (compound_add_item): The customer said multiple things in one utterance. "
        "Extract ALL distinct items, each with their own modifiers and sides. "
        "Attach sides/drinks/modifiers to the nearest relevant parent item. "
        "Preserve side size/variant inside each side entry.\n"
        "Example: 'chicken burger with small coke and a hamburger with large fries' ->\n"
        "  items: [{item_name: 'Chicken Burger', sides: [{name: 'Coke', size: 'small'}]},\n"
        "          {item_name: 'Hamburger', sides: [{name: 'Fries', size: 'large'}]}]\n"
        "Rules:\n"
        "- Do NOT merge two parent items.\n"
        "- Do NOT invent menu items.\n"
        "- If one span cannot be matched, put it in unresolved_spans.\n"
        "- Use decision='add_items' with all items in items[]."
    ),
    "correction": (
        "Task (correction): The customer is correcting something they said or "
        "that was just added. Use last_cart_diff as the likely correction target. "
        "Use decision='correction' with correction.original_text=what was wrong "
        "and correction.corrected_text=what they actually want. "
        "Do not duplicate the original item."
    ),
    "modifier_selection": (
        "Task (modifier_selection): The customer is selecting a modifier for the "
        "pending item. Their answer MUST map to one entry from allowed_options "
        "(fuzzy/phonetic match allowed). Use decision='add_items' with "
        "items[0].item_name=pending_item_name and the resolved modifier in "
        "items[0].modifiers[]. If no allowed_option matches, use decision='clarify'."
    ),
    "side_selection": (
        "Task (side_selection): The customer is selecting a side dish. Their "
        "answer MUST map to one or more entries from allowed_options. "
        "Use decision='add_items' with items[0].item_name=pending_item_name and "
        "the resolved side(s) in items[0].sides[]. "
        "If no allowed_option matches, use decision='clarify'."
    ),
    "checkout": (
        "Task (checkout): The customer wants to finalize their order. "
        "If cart is non-empty and no pending required selections remain, "
        "use decision='no_action' (the FSM handles checkout). "
        "If cart is empty, use decision='clarify' to ask what they'd like to order."
    ),
    "generic_repair": (
        "Task (generic_repair): The customer's intent is unclear. "
        "Use previous_turns and cart context to determine the most likely intent. "
        "Prefer 'clarify' over a risky guess."
    ),
}


def _build_user_message(
    *,
    transcript: str,
    state: str,
    local_intent: str,
    local_confidence: float,
    menu_context: list[str],
    cart_snapshot: list[str],
    last_cart_diff: list[str] | None,
    previous_turns: list[tuple[str, str]],
    task_mode: str | None = None,
    allowed_options: list[str] | None = None,
    pending_item_name: str | None = None,
    pending_group_name: str | None = None,
    reprompt_count: int = 0,
) -> str:
    """Build the compact user message sent to the LLM.

    Includes task-mode specific guidance when task_mode is provided.
    Never includes prices, PII, API keys, or full menu content.
    """
    lines: list[str] = [
        f"State: {state}",
        f"Customer said: {transcript!r}",
        f"Local intent: {local_intent} (confidence={local_confidence:.2f})",
    ]

    if pending_item_name:
        lines.append(f"Pending item: {pending_item_name!r}")

    if pending_group_name and reprompt_count > 0:
        lines.append(f"Reprompt count for {pending_group_name!r}: {reprompt_count}")

    if allowed_options:
        opts = ", ".join(allowed_options[:20])
        group_label = f" ({pending_group_name})" if pending_group_name else ""
        lines.append(f"Allowed options{group_label}: {opts}")

    if previous_turns:
        lines.append("Recent turns (bot -> customer):")
        for bot_text, user_text in previous_turns[-3:]:
            lines.append(f"  Bot: {bot_text!r}")
            lines.append(f"  Customer: {user_text!r}")

    if cart_snapshot:
        lines.append(f"Cart: {', '.join(cart_snapshot[:8])}")

    if last_cart_diff:
        lines.append(f"Last added to cart: {', '.join(last_cart_diff[:4])}")

    if menu_context:
        lines.append(f"Relevant menu items: {', '.join(menu_context[:12])}")

    # Append task-mode specific guidance at the end
    if task_mode and task_mode in _TASK_MODE_HINTS:
        lines.append("")  # blank separator
        lines.append(_TASK_MODE_HINTS[task_mode])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPT output parser
# ---------------------------------------------------------------------------

def _parse_plan(raw: str, *, trigger_reason: str, model: str, latency_ms: float) -> SmartTurnPlan:
    """Parse LLM JSON output into a SmartTurnPlan. Returns no_action on failure."""
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("response is not a JSON object")

        decision = str(data.get("decision", "unclear"))
        if decision not in {"add_items", "correction", "clarify", "no_action", "unclear"}:
            decision = "unclear"

        raw_conf = data.get("confidence", 0.0)
        try:
            confidence = float(raw_conf)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        # Parse items[]
        items: list[SmartTurnItem] = []
        for raw_item in (data.get("items") or []):
            if not isinstance(raw_item, dict):
                continue
            item_name = str(raw_item.get("item_name") or "").strip()
            if not item_name:
                continue
            try:
                qty = max(1, min(20, int(raw_item.get("quantity") or 1)))
            except (TypeError, ValueError):
                qty = 1
            modifiers: list[SmartTurnModifier] = []
            for rm in (raw_item.get("modifiers") or []):
                if isinstance(rm, dict):
                    mname = str(rm.get("name") or "").strip()
                    mop = str(rm.get("operation") or "add")
                    if mop not in {"add", "remove", "extra", "light"}:
                        mop = "add"
                    if mname:
                        modifiers.append(SmartTurnModifier(name=mname, operation=mop))
            sides: list[SmartTurnSide] = []
            for rs in (raw_item.get("sides") or []):
                if isinstance(rs, dict):
                    sname = str(rs.get("name") or "").strip()
                    try:
                        sqty = max(1, min(10, int(rs.get("quantity") or 1)))
                    except (TypeError, ValueError):
                        sqty = 1
                    ssize = str(rs["size"]).strip() if rs.get("size") else None
                    svariant = str(rs["variant"]).strip() if rs.get("variant") else None
                    if sname:
                        sides.append(SmartTurnSide(name=sname, quantity=sqty, size=ssize, variant=svariant))
            items.append(SmartTurnItem(
                item_name=item_name,
                quantity=qty,
                size=str(raw_item["size"]) if raw_item.get("size") else None,
                variant=str(raw_item["variant"]) if raw_item.get("variant") else None,
                modifiers=tuple(modifiers),
                sides=tuple(sides),
            ))

        # Parse correction
        correction: SmartTurnCorrection | None = None
        raw_corr = data.get("correction")
        if isinstance(raw_corr, dict):
            orig = str(raw_corr.get("original_text") or "").strip()
            corr = str(raw_corr.get("corrected_text") or "").strip()
            if orig and corr:
                correction = SmartTurnCorrection(
                    original_text=orig,
                    corrected_text=corr,
                )

        # response only logged — never applied to customer-facing output
        response: str | None = None
        if decision == "clarify" and data.get("response"):
            resp_raw = str(data["response"]).strip()
            # Clamp to 100 chars to avoid accidental prompt injection
            response = resp_raw[:100] if resp_raw else None

        reason = str(data.get("reason") or "").strip()[:200]

        # Parse unresolved_spans
        raw_unresolved = data.get("unresolved_spans") or []
        unresolved_spans = tuple(
            str(s).strip() for s in raw_unresolved
            if isinstance(s, str) and str(s).strip()
        )

        return SmartTurnPlan(
            decision=decision,
            confidence=confidence,
            items=tuple(items),
            correction=correction,
            response=response,
            reason=reason,
            trigger_reason=trigger_reason,
            latency_ms=latency_ms,
            gpt_called=True,
            model=model,
            unresolved_spans=unresolved_spans,
        )
    except Exception as exc:
        logger.warning("smart_turn_planner_parse_error: %s", exc)
        return SmartTurnPlan(
            decision="unclear",
            gpt_called=True,
            skipped_reason=None,
            reason=f"parse_error: {exc}",
            trigger_reason=trigger_reason,
            latency_ms=latency_ms,
            model=model,
        )


# ---------------------------------------------------------------------------
# Module-level executor for background/timeout calls
# ---------------------------------------------------------------------------

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor  # noqa: PLW0603
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="smart_turn_planner")
    return _executor


# ---------------------------------------------------------------------------
# Core GPT call (runs inside the executor thread)
# ---------------------------------------------------------------------------

def _call_gpt(
    *,
    user_message: str,
    model: str,
    timeout_s: float,
) -> str:
    """Make the OpenAI chat completion call. Returns the raw text content."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # Lazy import so the module works when openai is not installed
    try:
        import openai
    except ModuleNotFoundError:
        raise RuntimeError("openai package not installed")

    client = openai.OpenAI(api_key=api_key, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    return content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_smart_turn(
    *,
    transcript: str,
    state: str,
    local_intent: str,
    local_confidence: float,
    menu_context: list[str],
    cart_snapshot: list[str],
    last_cart_diff: list[str] | None = None,
    previous_turns: list[tuple[str, str]] | None = None,
    trigger_reason: str = "",
    model: str | None = None,
    # Task-mode context (all optional — safe when not provided)
    task_mode: str | None = None,
    allowed_options: list[str] | None = None,
    pending_item_name: str | None = None,
    pending_group_name: str | None = None,
    reprompt_count: int = 0,
) -> SmartTurnPlan | None:
    """Invoke the SmartTurnPlanner and return a plan, or None on failure.

    Never raises. Returns None when:
    - feature flag is disabled
    - OPENAI_API_KEY is not set
    - daily budget is exceeded
    - GPT call times out
    - response JSON is unparseable  (returns no_action plan, not None)

    Parameters
    ----------
    transcript:
        Normalized customer utterance.
    state:
        Current FSM state name (e.g. "WAITING_FOR_MODIFIER").
    local_intent:
        Intent predicted by local NLU model.
    local_confidence:
        Confidence score from local NLU (0.0–1.0).
    menu_context:
        Short list of relevant menu item names to include in the payload
        (≤12 items). Caller must filter — do NOT pass the full menu.
    cart_snapshot:
        Compact list of item names currently in the cart.
    last_cart_diff:
        Item names most recently added to the cart (for correction context).
    previous_turns:
        List of (bot_text, user_text) pairs, newest last (up to 3).
    trigger_reason:
        Why the planner was invoked — e.g. "compound_utterance".
    model:
        Override the OpenAI model. Defaults to OPENAI_MODEL_FAST env var or
        gpt-4o-mini.
    """
    if not _is_enabled():
        return None

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.debug("smart_turn_planner skipped: missing_api_key")
        return SmartTurnPlan(
            decision="no_action",
            gpt_called=False,
            skipped_reason="missing_api_key",
            trigger_reason=trigger_reason,
        )

    budget = _get_budget()
    if not budget.try_consume():
        logger.info(
            "smart_turn_planner_skipped",
            extra={"skipped_reason": "daily_budget_exceeded"},
        )
        return SmartTurnPlan(
            decision="no_action",
            gpt_called=False,
            skipped_reason="daily_budget_exceeded",
            trigger_reason=trigger_reason,
        )

    resolved_model = model or os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini")
    timeout_ms = _timeout_ms()
    timeout_s = timeout_ms / 1000.0

    user_message = _build_user_message(
        transcript=transcript,
        state=state,
        local_intent=local_intent,
        local_confidence=local_confidence,
        menu_context=menu_context or [],
        cart_snapshot=cart_snapshot or [],
        last_cart_diff=last_cart_diff,
        previous_turns=previous_turns or [],
        task_mode=task_mode,
        allowed_options=allowed_options,
        pending_item_name=pending_item_name,
        pending_group_name=pending_group_name,
        reprompt_count=reprompt_count,
    )

    t0 = time.monotonic()
    try:
        executor = _get_executor()
        future = executor.submit(
            _call_gpt,
            user_message=user_message,
            model=resolved_model,
            timeout_s=timeout_s,
        )
        raw = future.result(timeout=timeout_s)
        latency_ms = (time.monotonic() - t0) * 1000.0

        plan = _parse_plan(
            raw,
            trigger_reason=trigger_reason,
            model=resolved_model,
            latency_ms=latency_ms,
        )
        logger.info(
            "smart_turn_planner_result",
            extra={
                "smart_planner_invoked": True,
                "smart_planner_trigger_reason": trigger_reason,
                "smart_planner_task_mode": task_mode,
                "smart_planner_decision": plan.decision,
                "smart_planner_confidence": plan.confidence,
                "smart_planner_latency_ms": plan.latency_ms,
                "smart_planner_items_count": len(plan.items),
                "smart_planner_has_correction": plan.correction is not None,
                "allowed_options_count": len(allowed_options or []),
                "last_cart_diff_present": last_cart_diff is not None,
                "state_before": state,
                "local_intent": local_intent,
                "local_confidence": local_confidence,
            },
        )
        return plan

    except FuturesTimeoutError:
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.warning(
            "smart_turn_planner_timeout",
            extra={
                "smart_planner_invoked": True,
                "smart_planner_trigger_reason": trigger_reason,
                "smart_planner_latency_ms": latency_ms,
                "smart_planner_timed_out": True,
            },
        )
        return None

    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.warning(
            "smart_turn_planner_error: %s",
            exc,
            extra={
                "smart_planner_invoked": True,
                "smart_planner_trigger_reason": trigger_reason,
                "smart_planner_latency_ms": latency_ms,
                "smart_planner_error": str(exc),
            },
        )
        return None
