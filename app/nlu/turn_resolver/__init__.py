# app/nlu/turn_resolver/__init__.py
"""Unified GPT turn-resolution layer.

Provides a bucket-based routing framework that sits above the existing
specialized GPT services (option_resolver_service, add_item_planner_service).

Buckets
-------
  idle_menu_item_resolution  (Bucket 0)
      State == IDLE, local intent UNKNOWN / low confidence, text looks like
      a menu item query.  GPT assists with item identification.

  option_resolution  (Bucket 2)
      Waiting state (size/side/modifier), local option matcher failed.
      GPT resolves the selection from the available choices list.

  multi_item_add_planning  (Bucket 3)
      State == IDLE, multiple ITEM slots or compound markers present.
      GPT extracts a structured multi-item plan.

Safety contract
---------------
  * GPT never mutates cart, session state, FSM, or customer-facing response.
  * GPT result is always validated before ``apply_gpt=True`` is set.
  * ``FinalTurnDecision.source == "local"`` when GPT is disabled, skipped,
    or validation fails — the local deterministic path remains in effect.
"""
from app.nlu.turn_resolver.schemas import (
    GptTurnResolution,
    ResolvedItemPlan,
    ResolvedModifierPlan,
    ResolvedSidePlan,
    GPT_TURN_RESOLUTION_SKIPPED,
)
from app.nlu.turn_resolver.bucket_policy import (
    BUCKET_IDLE_ITEM,
    BUCKET_OPTION,
    BUCKET_MULTI_ITEM,
    pick_bucket,
)
from app.nlu.turn_resolver.final_turn_decision_resolver import (
    FinalTurnDecision,
    FINAL_DECISION_LOCAL,
)

__all__ = [
    # schemas
    "GptTurnResolution",
    "ResolvedItemPlan",
    "ResolvedModifierPlan",
    "ResolvedSidePlan",
    "GPT_TURN_RESOLUTION_SKIPPED",
    # bucket names
    "BUCKET_IDLE_ITEM",
    "BUCKET_OPTION",
    "BUCKET_MULTI_ITEM",
    "pick_bucket",
    # resolver
    "FinalTurnDecision",
    "FINAL_DECISION_LOCAL",
]
