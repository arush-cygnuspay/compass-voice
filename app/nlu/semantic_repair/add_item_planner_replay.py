# app/nlu/semantic_repair/add_item_planner_replay.py
"""Phase 4 Offline Replay + Evaluation Harness — core library.

Provides the importable engine for replaying captured turn logs or fixture
turns through the Phase 4 Add-Item Planner decision path without mutating any
production state (cart, session, FSM, or order).

Usage (from CLI or tests)
--------------------------
    from app.nlu.semantic_repair.add_item_planner_replay import (
        Phase4AddItemPlannerReplayHarness,
        PlannerReplayInputTurn,
        PlannerReplayResult,
        PlannerReplaySummaryBuilder,
        BUILT_IN_FIXTURES,
        parse_jsonl_row,
    )

    harness = Phase4AddItemPlannerReplayHarness(config=cfg)
    for result in harness.replay_fixtures(BUILT_IN_FIXTURES, mode="shadow"):
        print(result)

Safety invariants
------------------
* GPT is NEVER called unless `use_live_gpt=True` is passed at construction.
* `actual_applied` is always False — the harness never applies plans.
* No API key, PII, or cart mutation in output.
* All candidate item objects are synthetic (name-only) — no real menu loaded.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannerReplayInputTurn:
    """A single turn ready for Phase 4 add-item planner replay.

    All fields are optional except `user_text`.
    Missing fields degrade gracefully: empty candidate_items -> NO_GPT route.
    """

    user_text: str
    state_before: str = "IDLE"
    source_turn_id: str | None = None
    response_key_before: str | None = None
    local_intent: str | None = None
    local_confidence: float | None = None
    local_slots: tuple[dict, ...] = ()
    top_intents: tuple[dict, ...] = ()
    # Pre-built candidate items for the planner context.
    # Each dict: {"id": ..., "name": ..., "modifier_groups": [...], "side_groups": [...]}
    candidate_items: tuple[dict, ...] = ()
    cart_item_names: tuple[str, ...] = ()
    session_id: str | None = None
    turn_index: int | None = None


@dataclass(frozen=True, slots=True)
class PlannerReplayResult:
    """Structured result for a single replayed turn.

    `actual_applied` is always False — the harness never applies results.
    `would_apply` reflects what WOULD happen in inline mode if this were live.
    """

    replay_id: str
    source_turn_id: str | None
    user_text: str
    state_before: str
    mode: str                       # "disabled" | "shadow" | "inline"
    response_key_before: str | None = None
    local_intent: str | None = None
    local_confidence: float | None = None
    local_slots: list[dict] = field(default_factory=list)
    route_mode: str = "no_gpt"      # from AddItemPlannerResult
    route_reason: str = ""
    gpt_called: bool = False
    decision: str = "skipped"       # "add_items"|"clarify"|"no_repair"|"unclear"|"skipped"|"error"
    items_count: int = 0            # number of GPT items returned
    unresolved_count: int = 0
    confidence: float | None = None
    validator_passed: bool = False
    validator_reject_reason: str | None = None
    safe_to_apply: bool = False     # from apply gate
    would_apply: bool = False       # True only when mode=="inline" AND safe_to_apply AND decision=="add_items"
    actual_applied: bool = False    # ALWAYS False — replay never mutates state
    error: str | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (no PII, no API keys)."""
        return {
            "replay_id": self.replay_id,
            "source_turn_id": self.source_turn_id,
            "user_text": self.user_text,
            "state_before": self.state_before,
            "mode": self.mode,
            "response_key_before": self.response_key_before,
            "local_intent": self.local_intent,
            "local_confidence": self.local_confidence,
            "local_slots": list(self.local_slots),
            "route_mode": self.route_mode,
            "route_reason": self.route_reason,
            "gpt_called": self.gpt_called,
            "decision": self.decision,
            "items_count": self.items_count,
            "unresolved_count": self.unresolved_count,
            "confidence": self.confidence,
            "validator_passed": self.validator_passed,
            "validator_reject_reason": self.validator_reject_reason,
            "safe_to_apply": self.safe_to_apply,
            "would_apply": self.would_apply,
            "actual_applied": False,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# JSONL input parser
# ---------------------------------------------------------------------------


def parse_jsonl_row(
    row: dict[str, Any],
    *,
    filter_state: str | None = None,
) -> PlannerReplayInputTurn | None:
    """Convert a gpt_repair_turns.jsonl record to a PlannerReplayInputTurn.

    Returns None when the row is malformed, missing required fields, or
    filtered out by `filter_state`.

    The nested JSONL structure expected:
        {
          "session_id": "...",
          "turn_index": 1,
          "state_before": "IDLE",
          "normalized_text": "chicken burger with mozzarella",
          "response_key": "ask_for_item",
          "local": {"intent": "...", "confidence": 0.72, "slots": [], "top_intents": []},
          "allowed": {"choices": ["..."], ...},
        }
    """
    try:
        state = str(row.get("state_before") or "").strip().lower()
        if filter_state and state != filter_state.lower().strip():
            return None

        user_text = str(
            row.get("normalized_text") or row.get("customer_text") or ""
        ).strip()
        if not user_text:
            return None  # silence turn — skip

        local_block = row.get("local") or {}

        local_intent = str(local_block.get("intent") or "").strip() or None
        local_confidence_raw = local_block.get("confidence")
        local_confidence: float | None = None
        if local_confidence_raw is not None:
            try:
                local_confidence = float(local_confidence_raw)
            except (TypeError, ValueError):
                pass

        raw_slots = local_block.get("slots") or []
        local_slots: tuple[dict, ...] = ()
        if isinstance(raw_slots, list):
            local_slots = tuple(s for s in raw_slots if isinstance(s, dict))

        raw_top = local_block.get("top_intents") or []
        top_intents: tuple[dict, ...] = ()
        if isinstance(raw_top, list):
            top_intents = tuple(t for t in raw_top if isinstance(t, dict))

        session_id = str(row.get("session_id") or "").strip() or None
        turn_index_raw = row.get("turn_index")
        turn_index: int | None = None
        if turn_index_raw is not None:
            try:
                turn_index = int(turn_index_raw)
            except (TypeError, ValueError):
                pass

        source_id = (
            f"{session_id}:{turn_index}"
            if session_id and turn_index is not None
            else session_id or str(turn_index) if session_id or turn_index else None
        )

        return PlannerReplayInputTurn(
            user_text=user_text,
            state_before=state,
            source_turn_id=source_id,
            response_key_before=str(row.get("response_key") or "").strip() or None,
            local_intent=local_intent,
            local_confidence=local_confidence,
            local_slots=local_slots,
            top_intents=top_intents,
            session_id=session_id,
            turn_index=turn_index,
        )

    except Exception:
        return None


# ---------------------------------------------------------------------------
# No-op GPT client (used when use_live_gpt=False and no mock provided)
# ---------------------------------------------------------------------------


class _NoOpClient:
    """Mock OpenAI client that always returns a safe no_repair response.

    Prevents any real API calls in offline replay mode.
    """

    _SAFE_RESPONSE = json.dumps({
        "decision": "no_repair",
        "items": [],
        "unresolved": [],
        "confidence": 0.0,
        "reason_code": "no_match",
        "safe_to_apply": False,
    })

    class _Choice:
        class _Message:
            def __init__(self, content: str) -> None:
                self.content = content

        def __init__(self, content: str) -> None:
            self.message = _NoOpClient._Choice._Message(content)

    class _Completion:
        def __init__(self, content: str) -> None:
            self.choices = [_NoOpClient._Choice(content)]

    class _Completions:
        def __init__(self, content: str) -> None:
            self._content = content

        def create(self, **_kwargs: Any) -> "_NoOpClient._Completion":
            return _NoOpClient._Completion(self._content)

    class _Chat:
        def __init__(self, content: str) -> None:
            self.completions = _NoOpClient._Completions(content)

    def __init__(self) -> None:
        self.chat = _NoOpClient._Chat(self._SAFE_RESPONSE)


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


class Phase4AddItemPlannerReplayHarness:
    """Replay turns through the Phase 4 add-item planner without production side effects.

    Parameters
    ----------
    config:
        SemanticRepairConfig. `add_item_planner_mode` is overridden per-call
        by the `mode` argument to `replay_turn()`.
    use_live_gpt:
        If False (default), GPT is never called — the service uses a _NoOpClient
        that returns a safe no_repair response. Set to True with a real API key
        for staging.
    mock_client:
        Inject a test double for the OpenAI client (used in unit tests).
    """

    def __init__(
        self,
        config: "Any | None" = None,
        *,
        use_live_gpt: bool = False,
        mock_client: "Any | None" = None,
    ) -> None:
        from app.config.semantic_repair import get_semantic_repair_config

        if config is None:
            config = get_semantic_repair_config()
        self._base_config = config
        self._use_live_gpt = use_live_gpt
        self._mock_client = mock_client

    def _make_service(self, mode: str) -> "Any":
        """Create a GptAddItemPlannerService with the requested mode."""
        from app.config.semantic_repair import SemanticRepairConfig
        from app.nlu.semantic_repair.add_item_planner_service import GptAddItemPlannerService

        cfg = SemanticRepairConfig(
            phase=self._base_config.phase,
            model=self._base_config.model,
            timeout_seconds=float(
                getattr(self._base_config, "add_item_planner_timeout_ms", 1800) / 1000.0
            ),
            add_item_planner_mode=mode,
            add_item_planner_timeout_ms=getattr(
                self._base_config, "add_item_planner_timeout_ms", 1800
            ),
            add_item_planner_min_confidence=getattr(
                self._base_config, "add_item_planner_min_confidence", 0.75
            ),
            add_item_planner_max_item_candidates=getattr(
                self._base_config, "add_item_planner_max_item_candidates", 10
            ),
            add_item_planner_max_option_candidates=getattr(
                self._base_config, "add_item_planner_max_option_candidates", 20
            ),
        )

        client: Any
        if self._mock_client is not None:
            client = self._mock_client
        elif not self._use_live_gpt:
            client = _NoOpClient()
        else:
            client = None  # service will lazy-init from OPENAI_API_KEY

        svc = GptAddItemPlannerService(config=cfg, mock_client=client)
        return svc

    def replay_turn(self, turn: PlannerReplayInputTurn, *, mode: str) -> PlannerReplayResult:
        """Replay a single turn through the Phase 4 add-item planner.

        Parameters
        ----------
        turn : PlannerReplayInputTurn
        mode : "disabled" | "shadow" | "inline"

        Returns
        -------
        PlannerReplayResult — never raises; errors are captured in `result.error`.
        """
        replay_id = f"{turn.source_turn_id or 'turn'}-{mode}-{uuid.uuid4().hex[:6]}"
        t_start = time.perf_counter()

        try:
            svc = self._make_service(mode)

            local_slots_list = [dict(s) for s in turn.local_slots]
            top_intents_list = [dict(t) for t in turn.top_intents]
            candidate_items_list = [dict(c) for c in turn.candidate_items]
            cart_item_names_list = list(turn.cart_item_names)

            result = svc.run(
                user_text=turn.user_text,
                local_intent=turn.local_intent,
                local_confidence=turn.local_confidence or 0.0,
                local_slots=local_slots_list or None,
                top_k_intents=top_intents_list or None,
                candidate_items=candidate_items_list or None,
                cart_item_names=cart_item_names_list or None,
                previous_turns=None,
                state=turn.state_before,
            )

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            # would_apply: what WOULD happen if this were live in inline mode
            would_apply = (
                mode == "inline"
                and result.safe_to_apply
                and result.decision == "add_items"
            )

            return PlannerReplayResult(
                replay_id=replay_id,
                source_turn_id=turn.source_turn_id,
                user_text=turn.user_text,
                state_before=turn.state_before,
                mode=mode,
                response_key_before=turn.response_key_before,
                local_intent=turn.local_intent,
                local_confidence=turn.local_confidence,
                local_slots=local_slots_list,
                route_mode=result.route_mode,
                route_reason=result.route_reason,
                gpt_called=result.gpt_called,
                decision=result.decision,
                items_count=len(result.items),
                unresolved_count=len(result.unresolved),
                confidence=result.confidence,
                validator_passed=result.validator_passed,
                validator_reject_reason=result.validator_reject_reason,
                safe_to_apply=result.safe_to_apply,
                would_apply=would_apply,
                actual_applied=False,
                error=result.parse_error,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return PlannerReplayResult(
                replay_id=replay_id,
                source_turn_id=turn.source_turn_id,
                user_text=turn.user_text,
                state_before=turn.state_before,
                mode=mode,
                error=f"{type(exc).__name__}: {exc}"[:200],
                latency_ms=latency_ms,
            )

    def replay_fixtures(
        self,
        fixtures: list[PlannerReplayInputTurn],
        *,
        mode: str,
    ) -> Generator[PlannerReplayResult, None, None]:
        """Replay a list of fixture turns. Yields one PlannerReplayResult per turn."""
        for turn in fixtures:
            yield self.replay_turn(turn, mode=mode)

    def replay_jsonl(
        self,
        path: "str | Any",
        *,
        mode: str,
        max_turns: int = 0,
        filter_state: str | None = "idle",
        filter_response_key: str | None = None,
    ) -> Generator[PlannerReplayResult, None, None]:
        """Stream-replay turns from a gpt_repair_turns.jsonl file.

        Parameters
        ----------
        path:
            Path to the JSONL file.
        mode:
            GPT mode for this replay run.
        max_turns:
            Stop after this many successful turns (0 = unlimited).
        filter_state:
            Only replay turns where `state_before` matches this value.
            Defaults to "idle". Pass None to replay all states.
        filter_response_key:
            Optional additional filter on `response_key`.

        Yields
        ------
        PlannerReplayResult for each qualifying turn.
        """
        from pathlib import Path as _Path

        path = _Path(path)
        processed = 0

        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    yield _error_result(
                        error=f"JSONDecodeError on line {line_no}",
                        user_text="",
                        state_before=filter_state or "",
                        mode=mode,
                    )
                    continue

                if not isinstance(row, dict):
                    continue

                turn = parse_jsonl_row(row, filter_state=filter_state)
                if turn is None:
                    continue  # filtered out or malformed

                if filter_response_key and turn.response_key_before != filter_response_key:
                    continue

                result = self.replay_turn(turn, mode=mode)
                yield result

                processed += 1
                if max_turns and processed >= max_turns:
                    break


def _error_result(
    *,
    error: str,
    user_text: str,
    state_before: str,
    mode: str,
) -> PlannerReplayResult:
    return PlannerReplayResult(
        replay_id=f"err-{uuid.uuid4().hex[:8]}",
        source_turn_id=None,
        user_text=user_text,
        state_before=state_before,
        mode=mode,
        error=error,
    )


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


class PlannerReplaySummaryBuilder:
    """Aggregates PlannerReplayResult objects into a structured summary.

    Usage::

        builder = PlannerReplaySummaryBuilder(mode="shadow")
        for result in harness.replay_fixtures(BUILT_IN_FIXTURES, mode="shadow"):
            builder.add(result)
        summary = builder.to_dict()
        print(builder.to_markdown())
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._total = 0
        self._gpt_called = 0
        self._would_apply_count = 0
        self._error_count = 0
        self._route_modes: Counter[str] = Counter()
        self._decisions: Counter[str] = Counter()
        self._reject_reasons: Counter[str] = Counter()
        self._latencies_ms: list[float] = []
        self._would_apply_examples: list[dict] = []
        self._rejected_examples: list[dict] = []
        self._error_examples: list[dict] = []

    def add(self, result: PlannerReplayResult) -> None:
        """Register one replay result."""
        self._total += 1
        self._route_modes[result.route_mode] += 1

        if result.latency_ms is not None:
            self._latencies_ms.append(result.latency_ms)

        if result.error and not result.gpt_called:
            self._error_count += 1
            if len(self._error_examples) < 5:
                self._error_examples.append({
                    "user_text": result.user_text[:80],
                    "state": result.state_before,
                    "error": result.error[:120],
                })
            return

        if result.gpt_called:
            self._gpt_called += 1

        self._decisions[result.decision] += 1

        if result.error:
            self._error_count += 1
            if len(self._error_examples) < 5:
                self._error_examples.append({
                    "user_text": result.user_text[:80],
                    "state": result.state_before,
                    "error": result.error[:120],
                })

        if result.gpt_called and not result.validator_passed and result.validator_reject_reason:
            reason = result.validator_reject_reason or "unknown"
            self._reject_reasons[reason] += 1
            if len(self._rejected_examples) < 5:
                self._rejected_examples.append({
                    "user_text": result.user_text[:80],
                    "state": result.state_before,
                    "decision": result.decision,
                    "reject_reason": reason,
                })

        if result.would_apply:
            self._would_apply_count += 1
            if len(self._would_apply_examples) < 5:
                self._would_apply_examples.append({
                    "user_text": result.user_text[:80],
                    "state": result.state_before,
                    "decision": result.decision,
                    "items_count": result.items_count,
                    "confidence": result.confidence,
                })

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable summary dict."""
        p50, p95, p_max = _percentiles(self._latencies_ms)
        return {
            "mode": self._mode,
            "total_turns": self._total,
            "gpt_called": self._gpt_called,
            "would_apply_count": self._would_apply_count,
            "error_count": self._error_count,
            "route_modes": dict(self._route_modes),
            "decision_distribution": dict(self._decisions),
            "reject_reasons": dict(self._reject_reasons),
            "latency_ms": {"p50": p50, "p95": p95, "max": p_max},
            "would_apply_examples": self._would_apply_examples,
            "rejected_examples": self._rejected_examples,
            "error_examples": self._error_examples,
        }

    def to_markdown(self) -> str:
        """Render a human-readable markdown summary."""
        d = self.to_dict()
        total = max(d["total_turns"], 1)
        gpt = d["gpt_called"]

        lines: list[str] = [
            "# Phase 4 Add-Item Planner Replay Summary",
            "",
            f"**Mode**: `{d['mode']}`  ",
            f"**Total turns replayed**: {d['total_turns']}",
            "",
            "## Routing",
            "",
            "| Route Mode | Count | % |",
            "|------------|-------|---|",
        ]
        for rm, cnt in sorted(d["route_modes"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{rm}` | {cnt} | {100 * cnt / total:.1f}% |")

        lines += [
            "",
            "## GPT Calls",
            "",
            f"- **GPT called**: {gpt} ({100 * gpt / total:.1f}%)",
            f"- **Would apply (offline)**: {d['would_apply_count']}",
            f"- **Errors**: {d['error_count']}",
        ]

        if d["decision_distribution"]:
            lines += ["", "### Decision Distribution", ""]
            for dec, cnt in sorted(d["decision_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"- `{dec}`: {cnt} ({100 * cnt / max(gpt, 1):.1f}%)")

        if d["reject_reasons"]:
            lines += ["", "### Validator Reject Reasons", ""]
            for reason, cnt in sorted(d["reject_reasons"].items(), key=lambda x: -x[1]):
                lines.append(f"- `{reason}`: {cnt}")

        lat = d["latency_ms"]
        if lat["p50"] or lat["p95"] or lat["max"]:
            lines += [
                "",
                "## Latency (ms)",
                "",
                f"- p50: {lat['p50']:.1f}",
                f"- p95: {lat['p95']:.1f}",
                f"- max: {lat['max']:.1f}",
            ]

        if d["would_apply_examples"]:
            lines += ["", "## Would-Apply Examples (GPT improves local behavior)", ""]
            for ex in d["would_apply_examples"]:
                conf_str = (
                    f" (conf={ex['confidence']:.2f})" if ex.get("confidence") is not None else ""
                )
                lines.append(
                    f"- `{ex['user_text']}` -> {ex['items_count']} item(s){conf_str}"
                )

        if d["rejected_examples"]:
            lines += ["", "## Rejected Examples (needs prompt or routing improvement)", ""]
            for ex in d["rejected_examples"]:
                lines.append(
                    f"- `{ex['user_text']}` -> rejected: `{ex['reject_reason']}` "
                    f"(decision={ex['decision']})"
                )

        if d["error_examples"]:
            lines += ["", "## Errors", ""]
            for ex in d["error_examples"]:
                lines.append(f"- `{ex['user_text']}`: {ex['error']}")

        lines += ["", "---", "", "*Generated by Phase 4 Add-Item Planner Replay Harness*"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    n = len(s)
    p50 = s[int(n * 0.50)]
    p95 = s[min(int(n * 0.95), n - 1)]
    return p50, p95, s[-1]


# ---------------------------------------------------------------------------
# Built-in fixtures
# ---------------------------------------------------------------------------

BUILT_IN_FIXTURES: list[PlannerReplayInputTurn] = [
    # 1. Multi-modifier across two items — canonical Phase 4 example
    PlannerReplayInputTurn(
        source_turn_id="fixture:1",
        user_text="chicken burger with mozzarella, onions, mayo, and coke",
        state_before="IDLE",
        local_intent="add_item",
        local_confidence=0.51,
        candidate_items=(
            {
                "id": "chicken_burger",
                "name": "Chicken Burger",
                "modifier_groups": [
                    {"name": "Cheese choices", "choices": ["American Cheese", "Mozzarella Cheese"]},
                    {"name": "Toppings", "choices": ["Onions", "Lettuce", "Mayo"]},
                ],
                "side_groups": [],
            },
            {
                "id": "coke",
                "name": "Coke",
                "modifier_groups": [],
                "side_groups": [],
            },
        ),
    ),
    # 2. Two quantities of same item with per-item modifier variation
    PlannerReplayInputTurn(
        source_turn_id="fixture:2",
        user_text="two chicken sandwiches, one with Swiss, one plain",
        state_before="IDLE",
        local_intent="add_item",
        local_confidence=0.55,
        candidate_items=(
            {
                "id": "chicken_sandwich",
                "name": "Chicken Sandwich",
                "modifier_groups": [
                    {"name": "Cheese", "choices": ["Swiss", "American"]},
                ],
                "side_groups": [],
            },
        ),
    ),
    # 3. Combo item with side selection — side_groups pattern
    PlannerReplayInputTurn(
        source_turn_id="fixture:3",
        user_text="cheeseburger combo with fries and large coke",
        state_before="IDLE",
        local_intent="add_item",
        local_confidence=0.60,
        candidate_items=(
            {
                "id": "cheeseburger_combo",
                "name": "Cheeseburger Combo",
                "modifier_groups": [],
                "side_groups": [
                    {"name": "Drink", "choices": ["Coke", "Sprite", "Water"]},
                    {"name": "Side", "choices": ["Fries", "Onion Rings"]},
                ],
            },
            {
                "id": "coke",
                "name": "Coke",
                "modifier_groups": [],
                "side_groups": [],
            },
        ),
    ),
    # 4. Negated modifier — "no onions" + "extra cheese"
    PlannerReplayInputTurn(
        source_turn_id="fixture:4",
        user_text="burger with no onions and extra cheese",
        state_before="IDLE",
        local_intent="add_item",
        local_confidence=0.72,
        local_slots=(
            {"n": "ITEM", "v": "burger"},
            {"n": "MODIFIER", "v": "onions"},
            {"n": "MODIFIER", "v": "cheese"},
        ),
        candidate_items=(
            {
                "id": "burger",
                "name": "Burger",
                "modifier_groups": [
                    {"name": "Toppings", "choices": ["Onions", "Lettuce", "Cheese"]},
                ],
                "side_groups": [],
            },
        ),
    ),
    # 5. Multi-item no modifiers — simple list
    PlannerReplayInputTurn(
        source_turn_id="fixture:5",
        user_text="Hawaiian pizza and two cokes",
        state_before="IDLE",
        local_intent="add_item",
        local_confidence=0.58,
        candidate_items=(
            {
                "id": "hawaiian_pizza",
                "name": "Hawaiian Pizza",
                "modifier_groups": [],
                "side_groups": [],
            },
            {
                "id": "coke",
                "name": "Coke",
                "modifier_groups": [],
                "side_groups": [],
            },
        ),
    ),
    # 6. Nonsense text — GPT should return no_repair/error; no candidates
    PlannerReplayInputTurn(
        source_turn_id="fixture:6",
        user_text="blarbqux frizzlestick supreme delight",
        state_before="IDLE",
        local_intent="unknown",
        local_confidence=0.10,
        candidate_items=(),
    ),
    # 7. Vague utterance with low confidence — no candidates resolved
    PlannerReplayInputTurn(
        source_turn_id="fixture:7",
        user_text="the thing with cheese and whatever",
        state_before="IDLE",
        local_intent="add_item",
        local_confidence=0.30,
        candidate_items=(),
    ),
]
