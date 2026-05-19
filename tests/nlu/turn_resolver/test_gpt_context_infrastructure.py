# tests/nlu/turn_resolver/test_gpt_context_infrastructure.py
"""Tests for Priority 1: Runtime Turn Memory + GPT Context Infrastructure.

Covers:
  1.  turn_memory stores user and assistant entries
  2.  turn_memory bounded to max 6 entries
  3.  turn_memory keeps previous 3 exchanges in order
  4.  turn_memory clears on full reset but not reset_item_scope
  5.  GptContextBuilder includes previous turns
  6.  GptContextBuilder includes previous assistant prompt
  7.  GptContextBuilder includes allowed intents for idle
  8.  GptContextBuilder includes allowed intents for waiting_for_modifier
  9.  GptContextBuilder includes allowed response keys for waiting_for_modifier
  10. GptContextBuilder includes current modifier options
  11. GptContextBuilder includes current side options
  12. GptContextBuilder includes current size options
  13. GptContextBuilder does not include full menu
  14. GptContextBuilder caps menu_candidates to max 12
  15. GptContextBuilder caps allowed_options to configured max
  16. GptContextBuilder sanitizes phone/email/payment-link-like text
  17. PromptRegistry returns different prompts per task mode
  18. PromptRegistry returns generic prompt for unknown task
  19. AllowedIntentProvider terminal state returns no unsafe cart/payment intents
  20. AllowedOptionExtractor returns empty tuple safely when pending context missing
  21. (Integration) turn_memory_service appends structured entries
"""
from __future__ import annotations

import pytest

from app.state_machine.models.conversation_context import ConversationContext
from app.state_machine.models.turn_memory import TurnMemoryEntry
from app.state_machine.services.turn_memory_service import (
    append_assistant_turn_memory,
    append_user_turn_memory,
    get_recent_turns,
)
from app.nlu.turn_resolver.allowed_intent_provider import AllowedIntentProvider
from app.nlu.turn_resolver.allowed_response_key_provider import AllowedResponseKeyProvider
from app.nlu.turn_resolver.allowed_option_extractor import AllowedOptionExtractor
from app.nlu.turn_resolver.gpt_context_builder import GptContextBuilder
from app.nlu.turn_resolver.prompt_registry import (
    PromptRegistry,
    TASK_MODIFIER_OPTION_RESOLUTION,
    TASK_SIDE_OPTION_RESOLUTION,
    TASK_IDLE_ADD_ITEM_OR_MENU_QUERY,
    TASK_GENERIC_UNKNOWN_REPAIR,
)
from app.state_machine.models.pending_item_models import (
    PendingAddItem,
    PendingModifierGroup,
    PendingModifierChoice,
    PendingSideGroup,
    PendingSideChoice,
    PendingVariantChoice,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_context() -> ConversationContext:
    return ConversationContext()


def _make_modifier_group(group_id: str = "g1", name: str = "Sauces") -> PendingModifierGroup:
    choices = [
        PendingModifierChoice(
            modifier_id=f"m{i}",
            name=choice_name,
            group_id=group_id,
            normalized_name=choice_name.lower(),
            match_texts=(choice_name.lower(), f"{choice_name.lower()} sauce"),
        )
        for i, choice_name in enumerate(["Ranch", "BBQ", "Honey Mustard"])
    ]
    return PendingModifierGroup(
        group_id=group_id,
        name=name,
        is_required=False,
        min_selector=0,
        max_selector=1,
        choices=choices,
    )


def _make_side_group(group_id: str = "sg1", name: str = "Drinks") -> PendingSideGroup:
    choices = [
        PendingSideChoice(
            item_id=f"s{i}",
            name=side_name,
            pricing_mode="fixed",
            normalized_name=side_name.lower(),
            match_texts=(side_name.lower(),),
            top_variant_names=("small", "medium", "large"),
        )
        for i, side_name in enumerate(["Coke", "Sprite", "Water"])
    ]
    return PendingSideGroup(
        group_id=group_id,
        name=name,
        is_required=False,
        min_selector=0,
        max_selector=1,
        choices=choices,
    )


def _make_pending_item_with_modifier() -> PendingAddItem:
    mod_group = _make_modifier_group()
    return PendingAddItem(
        item_id="item1",
        item_name="Wings",
        modifier_groups=[mod_group],
        modifier_groups_by_id={"g1": mod_group},
    )


def _make_pending_item_with_side() -> PendingAddItem:
    side_group = _make_side_group()
    return PendingAddItem(
        item_id="item2",
        item_name="Burger",
        side_groups=[side_group],
        side_groups_by_id={"sg1": side_group},
    )


def _make_pending_item_with_variants() -> PendingAddItem:
    variants = [
        PendingVariantChoice(variant_id=f"v{i}", name=size, normalized_name=size.lower())
        for i, size in enumerate(["Small", "Medium", "Large"])
    ]
    return PendingAddItem(
        item_id="item3",
        item_name="Soda",
        item_variants=variants,
        item_variants_by_id={v.variant_id: v for v in variants},
    )


# ── Test 1: turn_memory stores user and assistant entries ─────────────────────

def test_turn_memory_stores_user_entry():
    ctx = _make_context()
    append_user_turn_memory(ctx, "I want a burger", "i want a burger")
    entries = get_recent_turns(ctx)
    assert len(entries) == 1
    assert entries[0].role == "user"
    assert "burger" in entries[0].text


def test_turn_memory_stores_assistant_entry():
    ctx = _make_context()
    append_assistant_turn_memory(ctx, "I've added a burger to your order.", "item_added_successfully")
    entries = get_recent_turns(ctx)
    assert len(entries) == 1
    assert entries[0].role == "assistant"
    assert entries[0].response_key == "item_added_successfully"


# ── Test 2: turn_memory bounded to max 6 entries ─────────────────────────────

def test_turn_memory_bounded_at_6():
    ctx = _make_context()
    for i in range(10):
        ctx.append_turn_memory("user", f"utterance {i}")
    assert len(ctx.turn_memory) <= 6


def test_turn_memory_service_bounded_at_6():
    ctx = _make_context()
    for i in range(10):
        append_user_turn_memory(ctx, f"utterance {i}")
    entries = get_recent_turns(ctx, max_entries=10)
    assert len(entries) <= 6


# ── Test 3: turn_memory keeps previous 3 exchanges in order ──────────────────

def test_turn_memory_preserves_order():
    ctx = _make_context()
    append_user_turn_memory(ctx, "first", state="idle")
    append_assistant_turn_memory(ctx, "first response", "resp_key_1")
    append_user_turn_memory(ctx, "second", state="idle")
    append_assistant_turn_memory(ctx, "second response", "resp_key_2")
    append_user_turn_memory(ctx, "third", state="idle")
    append_assistant_turn_memory(ctx, "third response", "resp_key_3")

    entries = get_recent_turns(ctx, max_entries=6)
    texts = [e.text for e in entries]
    assert texts[0] == "first"
    assert texts[1] == "first response"
    assert texts[4] == "third"
    assert texts[5] == "third response"


def test_turn_memory_oldest_dropped_when_full():
    ctx = _make_context()
    # Fill to exactly 6
    for i in range(6):
        append_user_turn_memory(ctx, f"turn {i}")
    # Add one more — oldest should be dropped
    append_user_turn_memory(ctx, "turn 6")
    entries = get_recent_turns(ctx, max_entries=6)
    assert len(entries) == 6
    texts = [e.text for e in entries]
    assert "turn 0" not in texts
    assert "turn 6" in texts


# ── Test 4: turn_memory clears on full reset but not reset_item_scope ─────────

def test_turn_memory_not_cleared_by_reset_item_scope():
    ctx = _make_context()
    append_user_turn_memory(ctx, "I want wings")
    ctx.reset_item_scope()
    entries = get_recent_turns(ctx)
    assert len(entries) == 1, "turn_memory must survive reset_item_scope"


def test_turn_memory_cleared_by_reset_session_scope():
    ctx = _make_context()
    append_user_turn_memory(ctx, "I want wings")
    ctx.reset_session_scope()
    entries = get_recent_turns(ctx)
    assert len(entries) == 0, "turn_memory must be cleared by reset_session_scope"


def test_turn_memory_cleared_by_reset_all():
    ctx = _make_context()
    append_user_turn_memory(ctx, "I want wings")
    ctx.reset_all()
    entries = get_recent_turns(ctx)
    assert len(entries) == 0


# ── Test 5: GptContextBuilder includes previous turns ────────────────────────

def test_context_builder_includes_previous_turns():
    ctx = _make_context()
    append_user_turn_memory(ctx, "I want wings", state="idle", intent="add_item")
    append_assistant_turn_memory(ctx, "What sauce would you like?", "ask_for_modifier", state="waiting_for_modifier")

    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="ranch please",
        normalized_text="ranch please",
        local_intent="select_modifier",
        local_confidence=0.85,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_MODIFIER_OPTION_RESOLUTION,
        state="waiting_for_modifier",
    )

    prev = packet["previous_turns"]
    assert len(prev) == 2
    assert prev[0]["role"] == "user"
    assert "wings" in prev[0]["text"]
    assert prev[1]["role"] == "assistant"


# ── Test 6: GptContextBuilder includes previous assistant prompt ──────────────

def test_context_builder_includes_previous_assistant_prompt():
    ctx = _make_context()
    append_user_turn_memory(ctx, "I want wings")
    append_assistant_turn_memory(ctx, "What sauce?", "ask_for_modifier", state="waiting_for_modifier")

    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="ranch",
        normalized_text="ranch",
        local_intent="select_modifier",
        local_confidence=0.9,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_MODIFIER_OPTION_RESOLUTION,
        state="waiting_for_modifier",
    )

    assert packet["previous_assistant_prompt"] == "ask_for_modifier"


# ── Test 7: GptContextBuilder includes allowed intents for idle ───────────────

def test_context_builder_allowed_intents_idle():
    ctx = _make_context()
    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="can I get a burger",
        normalized_text="can i get a burger",
        local_intent="add_item",
        local_confidence=0.7,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_IDLE_ADD_ITEM_OR_MENU_QUERY,
        state="idle",
    )

    intent_names = [ai["name"] for ai in packet["allowed_intents"]]
    assert "add_item" in intent_names
    assert "checkout" in intent_names
    assert "ask_menu" in intent_names
    # Payment-like intents must NOT appear in idle
    assert "payment_preference" not in intent_names


# ── Test 8: GptContextBuilder includes allowed intents for waiting_for_modifier

def test_context_builder_allowed_intents_waiting_for_modifier():
    ctx = _make_context()
    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="ranch",
        normalized_text="ranch",
        local_intent="select_modifier",
        local_confidence=0.8,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_MODIFIER_OPTION_RESOLUTION,
        state="waiting_for_modifier",
    )

    intent_names = [ai["name"] for ai in packet["allowed_intents"]]
    assert "select_modifier" in intent_names
    assert "skip_optional_modifier" in intent_names
    assert "cancel_pending_item" in intent_names
    # add_item must NOT be in waiting_for_modifier
    assert "add_item" not in intent_names


# ── Test 9: GptContextBuilder includes allowed response keys ──────────────────

def test_context_builder_allowed_response_keys_waiting_for_modifier():
    ctx = _make_context()
    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="ranch",
        normalized_text="ranch",
        local_intent="select_modifier",
        local_confidence=0.8,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_MODIFIER_OPTION_RESOLUTION,
        state="waiting_for_modifier",
    )

    keys = packet["allowed_response_keys"]
    assert "ask_for_modifier" in keys
    assert "modifier_selected" in keys
    assert "generic_clarification" in keys
    # Payment keys must NOT appear
    assert "pickup_sms_sent_end_call" not in keys


# ── Test 10: GptContextBuilder includes current modifier options ──────────────

def test_context_builder_includes_modifier_options():
    ctx = _make_context()
    ctx.current_item_id = "item1"
    ctx.current_item_name = "Wings"
    ctx.pending_add_item = _make_pending_item_with_modifier()
    ctx.current_modifier_group_index = 0

    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="ranch",
        normalized_text="ranch",
        local_intent="select_modifier",
        local_confidence=0.85,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_MODIFIER_OPTION_RESOLUTION,
        state="waiting_for_modifier",
    )

    options = packet["allowed_options"]
    assert len(options) == 3
    option_names = [o["name"] for o in options]
    assert "Ranch" in option_names
    assert "BBQ" in option_names
    assert "Honey Mustard" in option_names


# ── Test 11: GptContextBuilder includes current side options ──────────────────

def test_context_builder_includes_side_options():
    ctx = _make_context()
    ctx.current_item_id = "item2"
    ctx.current_item_name = "Burger"
    ctx.pending_add_item = _make_pending_item_with_side()
    ctx.current_side_group_index = 0

    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="coke please",
        normalized_text="coke please",
        local_intent="select_side",
        local_confidence=0.9,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_SIDE_OPTION_RESOLUTION,
        state="waiting_for_side",
    )

    options = packet["allowed_options"]
    option_names = [o["name"] for o in options]
    assert "Coke" in option_names
    assert "Sprite" in option_names
    assert "Water" in option_names


# ── Test 12: GptContextBuilder includes current size options ──────────────────

def test_context_builder_includes_size_options():
    ctx = _make_context()
    ctx.current_item_id = "item3"
    ctx.current_item_name = "Soda"
    ctx.pending_add_item = _make_pending_item_with_variants()

    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="large please",
        normalized_text="large please",
        local_intent="select_size",
        local_confidence=0.9,
        local_candidates=None,
        local_slots=None,
        task_mode="size_option_resolution",
        state="waiting_for_size",
    )

    options = packet["allowed_options"]
    option_names = [o["name"] for o in options]
    assert "Small" in option_names
    assert "Medium" in option_names
    assert "Large" in option_names


# ── Test 13: GptContextBuilder does not include full menu ─────────────────────

def test_context_builder_no_full_menu_without_candidates():
    ctx = _make_context()
    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="what do you have",
        normalized_text="what do you have",
        local_intent="ask_menu",
        local_confidence=0.6,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_IDLE_ADD_ITEM_OR_MENU_QUERY,
        state="idle",
    )

    assert "menu_candidates" not in packet or len(packet.get("menu_candidates", [])) == 0
    assert "full_menu" not in packet
    assert "menu" not in packet


# ── Test 14: GptContextBuilder caps menu_candidates to max 12 ────────────────

def test_context_builder_caps_menu_candidates():
    ctx = _make_context()
    # Supply 20 candidates — should be capped to 12
    candidates = [{"name": f"Item {i}", "item_id": f"id{i}"} for i in range(20)]

    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="burger",
        normalized_text="burger",
        local_intent="add_item",
        local_confidence=0.7,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_IDLE_ADD_ITEM_OR_MENU_QUERY,
        state="idle",
        menu_candidates=candidates,
    )

    assert len(packet["menu_candidates"]) <= 12


# ── Test 15: GptContextBuilder caps allowed_options to configured max ─────────

def test_context_builder_caps_allowed_options():
    ctx = _make_context()
    # Supply 20 options — should be capped to 12
    options = [{"index": i, "name": f"Option {i}", "modifier_id": f"m{i}"} for i in range(20)]

    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="ranch",
        normalized_text="ranch",
        local_intent="select_modifier",
        local_confidence=0.85,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_MODIFIER_OPTION_RESOLUTION,
        state="waiting_for_modifier",
        allowed_options=options,
    )

    assert len(packet["allowed_options"]) <= 12


# ── Test 16: GptContextBuilder sanitizes PII in text ─────────────────────────

def test_context_builder_sanitizes_phone_number():
    ctx = _make_context()
    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="call me at 5551234567",
        normalized_text="call me at 5551234567",
        local_intent="unknown",
        local_confidence=0.1,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_GENERIC_UNKNOWN_REPAIR,
        state="idle",
    )

    assert "5551234567" not in (packet["user_text"] or "")
    assert "[REDACTED" in (packet["user_text"] or "")


def test_context_builder_sanitizes_email():
    ctx = _make_context()
    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="send to user@example.com",
        normalized_text="send to user@example.com",
        local_intent="unknown",
        local_confidence=0.1,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_GENERIC_UNKNOWN_REPAIR,
        state="idle",
    )

    assert "user@example.com" not in (packet["user_text"] or "")
    assert "[REDACTED_EMAIL]" in (packet["user_text"] or "")


def test_context_builder_sanitizes_payment_link():
    ctx = _make_context()
    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="go to https://pay.example.com/order123",
        normalized_text="go to https://pay.example.com/order123",
        local_intent="unknown",
        local_confidence=0.1,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_GENERIC_UNKNOWN_REPAIR,
        state="idle",
    )

    assert "pay.example.com" not in (packet["user_text"] or "")
    assert "[REDACTED_URL]" in (packet["user_text"] or "")


# ── Test 17: PromptRegistry returns different prompts per task mode ───────────

def test_prompt_registry_different_prompts_per_mode():
    registry = PromptRegistry()

    modifier_sys = registry.get_system_prompt(TASK_MODIFIER_OPTION_RESOLUTION)
    side_sys = registry.get_system_prompt(TASK_SIDE_OPTION_RESOLUTION)
    assert modifier_sys != side_sys

    modifier_instr = registry.get_task_instructions(TASK_MODIFIER_OPTION_RESOLUTION)
    side_instr = registry.get_task_instructions(TASK_SIDE_OPTION_RESOLUTION)
    assert modifier_instr != side_instr

    modifier_contract = registry.get_output_contract(TASK_MODIFIER_OPTION_RESOLUTION)
    side_contract = registry.get_output_contract(TASK_SIDE_OPTION_RESOLUTION)
    assert modifier_contract != side_contract


def test_prompt_registry_modifier_mentions_allowed_options():
    registry = PromptRegistry()
    instr = registry.get_task_instructions(TASK_MODIFIER_OPTION_RESOLUTION)
    assert "allowed_options" in instr.lower() or "allowed" in instr.lower()


def test_prompt_registry_all_modes_return_nonempty_prompts():
    from app.nlu.turn_resolver.prompt_registry import _ALL_TASK_MODES
    registry = PromptRegistry()
    for mode in _ALL_TASK_MODES:
        assert registry.get_system_prompt(mode), f"Empty system prompt for {mode}"
        assert registry.get_task_instructions(mode), f"Empty task instructions for {mode}"
        assert registry.get_output_contract(mode), f"Empty output contract for {mode}"


# ── Test 18: PromptRegistry returns generic prompt for unknown task ───────────

def test_prompt_registry_unknown_mode_returns_generic():
    registry = PromptRegistry()
    generic_sys = registry.get_system_prompt(TASK_GENERIC_UNKNOWN_REPAIR)
    unknown_sys = registry.get_system_prompt("this_mode_does_not_exist_xyz")
    # Should return generic, not raise
    assert unknown_sys == generic_sys
    assert unknown_sys  # Non-empty

    assert not registry.is_known_task_mode("this_mode_does_not_exist_xyz")
    assert registry.is_known_task_mode(TASK_MODIFIER_OPTION_RESOLUTION)


# ── Test 19: AllowedIntentProvider terminal state is safe ─────────────────────

def test_allowed_intent_provider_terminal_state_no_payment_intents():
    provider = AllowedIntentProvider()

    for terminal_state in ("completed", "transferring_to_human_agent"):
        intents = provider.get_allowed_intents_for_state(terminal_state)
        names = {ai.name for ai in intents}
        # These must not appear in terminal states
        unsafe = {"checkout", "add_item", "payment_preference", "remove_item", "modify_item"}
        overlap = names & unsafe
        assert not overlap, f"Terminal state '{terminal_state}' has unsafe intents: {overlap}"


def test_allowed_intent_provider_unknown_state_returns_conservative():
    provider = AllowedIntentProvider()
    intents = provider.get_allowed_intents_for_state("totally_unknown_state_xyz")
    names = {ai.name for ai in intents}
    # Should only have safe, conservative intents
    assert "add_item" not in names
    assert "checkout" not in names


def test_allowed_intent_provider_normalizes_state_case():
    provider = AllowedIntentProvider()
    lower = provider.get_allowed_intents_for_state("idle")
    upper = provider.get_allowed_intents_for_state("IDLE")
    mixed = provider.get_allowed_intents_for_state("Idle")
    assert lower == upper == mixed


# ── Test 20: AllowedOptionExtractor returns empty tuple safely ────────────────

def test_option_extractor_empty_when_no_pending_item():
    extractor = AllowedOptionExtractor()
    ctx = _make_context()
    # No pending_add_item set
    assert extractor.extract(ctx, "waiting_for_modifier") == ()
    assert extractor.extract(ctx, "waiting_for_side") == ()
    assert extractor.extract(ctx, "waiting_for_size") == ()


def test_option_extractor_empty_for_unknown_state():
    extractor = AllowedOptionExtractor()
    ctx = _make_context()
    ctx.pending_add_item = _make_pending_item_with_modifier()
    result = extractor.extract(ctx, "some_unknown_state")
    assert result == ()


def test_option_extractor_returns_order_type_options():
    extractor = AllowedOptionExtractor()
    ctx = _make_context()
    result = extractor.extract(ctx, "waiting_for_order_type")
    assert len(result) == 2
    names = {o["name"] for o in result}
    assert "pickup" in names
    assert "delivery" in names


def test_option_extractor_handles_out_of_bounds_group_index():
    extractor = AllowedOptionExtractor()
    ctx = _make_context()
    ctx.pending_add_item = _make_pending_item_with_modifier()
    ctx.current_modifier_group_index = 99  # way out of bounds
    result = extractor.extract(ctx, "waiting_for_modifier")
    assert result == ()


# ── Test 21: turn_memory_service appends structured entries ───────────────────

def test_turn_memory_service_user_entry_has_fields():
    ctx = _make_context()
    append_user_turn_memory(
        ctx,
        text="I want a burger",
        normalized_text="i want a burger",
        state="idle",
        intent="add_item",
    )
    entries = get_recent_turns(ctx)
    assert len(entries) == 1
    e = entries[0]
    assert isinstance(e, TurnMemoryEntry)
    assert e.role == "user"
    assert e.normalized_text == "i want a burger"
    assert e.state == "idle"
    assert e.intent == "add_item"
    assert e.timestamp_utc is not None


def test_turn_memory_service_assistant_entry_has_fields():
    ctx = _make_context()
    append_assistant_turn_memory(
        ctx,
        response_text="What sauce would you like?",
        response_key="ask_for_modifier",
        state="waiting_for_modifier",
    )
    entries = get_recent_turns(ctx)
    assert len(entries) == 1
    e = entries[0]
    assert isinstance(e, TurnMemoryEntry)
    assert e.role == "assistant"
    assert e.response_key == "ask_for_modifier"
    assert e.state == "waiting_for_modifier"
    assert e.intent is None
    assert e.timestamp_utc is not None


def test_turn_memory_service_skips_empty_text():
    ctx = _make_context()
    append_user_turn_memory(ctx, "")
    append_user_turn_memory(ctx, "   ")
    assert len(ctx.turn_memory) == 0


def test_turn_memory_backward_compat_append():
    """ctx.append_turn_memory('bot', text) still works and normalises to 'assistant'."""
    ctx = _make_context()
    ctx.append_turn_memory("bot", "Hello!")
    entries = get_recent_turns(ctx)
    assert len(entries) == 1
    assert entries[0].role == "assistant"
    assert entries[0].text == "Hello!"


def test_turn_memory_get_turn_memory_backward_compat():
    """get_turn_memory() returns (role, text) tuples as before."""
    ctx = _make_context()
    ctx.append_turn_memory("user", "hi")
    ctx.append_turn_memory("bot", "hello")
    pairs = ctx.get_turn_memory(2)
    assert len(pairs) == 2
    roles = [p[0] for p in pairs]
    assert "user" in roles
    assert "assistant" in roles


def test_context_builder_build_metadata():
    ctx = _make_context()
    append_user_turn_memory(ctx, "I want wings")
    builder = GptContextBuilder()
    packet = builder.build(
        context=ctx,
        user_text="wings please",
        normalized_text="wings please",
        local_intent="add_item",
        local_confidence=0.9,
        local_candidates=None,
        local_slots=None,
        task_mode=TASK_IDLE_ADD_ITEM_OR_MENU_QUERY,
        state="idle",
    )
    meta = builder.build_metadata(packet)
    assert meta["gpt_context_built"] is True
    assert meta["gpt_task_mode"] == TASK_IDLE_ADD_ITEM_OR_MENU_QUERY
    assert meta["gpt_context_previous_turn_count"] == 1
    assert meta["gpt_context_allowed_intents_count"] > 0
    assert meta["gpt_context_allowed_response_keys_count"] > 0


def test_context_builder_allowed_response_key_provider_all_states():
    """AllowedResponseKeyProvider returns non-empty tuples for all known states."""
    provider = AllowedResponseKeyProvider()
    known_states = [
        "idle", "waiting_for_order_type", "waiting_for_modifier", "waiting_for_side",
        "waiting_for_size", "waiting_for_side_size", "confirming_item", "confirming_order",
        "waiting_for_pickup_sms_permission", "completed",
    ]
    for state in known_states:
        keys = provider.get_allowed_response_keys_for_state(state)
        assert len(keys) > 0, f"No response keys for state {state}"
