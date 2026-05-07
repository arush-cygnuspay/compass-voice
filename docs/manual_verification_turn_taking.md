# Manual Verification: Strict Turn-Taking & Barge-In Fix

Run these scenarios in order against a live Compass Voice instance with a real phone call.
Each scenario lists the exact log events to look for in the structured output.

---

## Prerequisites

```bash
# Enable debug-level structured logging
export BARGE_IN_ENABLED=true
export STRICT_TURN_TAKING=true
export MIN_BARGE_IN_AUDIO_MS=700
export MIN_BARGE_IN_WORDS=2
export POST_PLAYBACK_GUARD_MS=250
export USER_TURN_COMMIT_DELAY_MS=700
```

Start the server and place a call. Tail logs in a second terminal:

```bash
python -m app.api.twilio_server   # or however you start locally
```

---

## Scenario 1 — Split utterance produces ONE FSM turn

**Action:** After the greeting, say in one breath:
> "I want a chicken burger with cheese."

**Pass condition:** Exactly one `[user_turn_committed]` log entry followed by exactly one `[turn_processing_started]`. The FSM receives the full text, not two fragments.

**Logs to confirm:**

```
[stt_final_received]    transcript="I want a chicken burger" is_final=true speech_final=false
[user_turn_commit_scheduled]  delay_ms=700
[stt_final_received]    transcript="with cheese." is_final=true speech_final=false
[user_turn_commit_cancelled_or_merged]  reason=new_final_arrived
[user_turn_commit_scheduled]  delay_ms=700
[user_turn_committed]   text="I want a chicken burger with cheese." turn_id=1
[turn_processing_started]  turn_id=1  text="I want a chicken burger with cheese."
[turn_processing_finished] turn_id=1
[assistant_tts_started]
```

**Fail condition:** Two `[turn_processing_started]` events for the same utterance.

---

## Scenario 2 — Filler/noise during TTS is ignored

**Action:** While the bot is mid-sentence responding, say quietly:
> "uh"

**Pass condition:** TTS continues uninterrupted. No FSM turn. No `[barge_in_accepted]`.

**Logs to confirm:**

```
[barge_in_candidate]   text="uh"
[barge_in_rejected]    reason="filler_only"
```

No `[turn_processing_started]` or `[assistant_tts_finished]` interruption should follow.

---

## Scenario 3 — Meaningful correction interrupts TTS and is processed

**Action:** While the bot is speaking (you should hear it mid-sentence), say clearly:
> "no, change that to coke"

**Pass condition:** TTS stops within ~1 second, and the FSM processes your correction.

**Logs to confirm:**

```
[barge_in_candidate]   text="no change that to coke"  audio_ms=900  confidence=0.92
[barge_in_accepted]    reason="accepted"
[assistant_tts_finished]  reason="interrupted"
[turn_processing_started]  text="no change that to coke"
```

No `[barge_in_rejected]` should appear for this phrase.

---

## Scenario 4 — Echo/tail audio immediately after TTS is suppressed

**Action:** Stay completely silent. Watch what happens in the 250 ms immediately after the bot finishes speaking. (Bot audio can sometimes echo back into Deepgram.)

**Pass condition:** No FSM turn fires from the echo tail. Any detected transcript during the guard window is rejected.

**Logs to confirm (if any audio is detected):**

```
[barge_in_candidate]   text="..." audio_ms=<short>
[barge_in_rejected]    reason="inside_playback_guard_window"
```

---

## Scenario 5 — User speaks immediately after bot finishes (mark ack)

**Action:** Wait for the bot to finish completely (you'll hear it stop). Immediately say:
> "pickup please"

**Pass condition:** The FSM processes your turn normally — no rejection, no delay beyond the configured 250 ms response delay.

**Logs to confirm:**

```
[assistant_tts_finished]   reason="mark_ack"
[stt_final_received]   transcript="pickup please"
[user_turn_committed]  text="pickup please"  turn_id=2
[turn_processing_started]  turn_id=2  text="pickup please"
```

No `[barge_in_rejected]` should appear (bot has already finished; this is normal listening mode).

---

## Scenario 6 — Stale/duplicate STT event does not mutate session

**Action:** This is hard to trigger manually. Simulate by checking logs over a full call for any:

```
[stale_turn_event_ignored]  turn_id=<N>  last_committed=<M>  (where N <= M)
```

If you see this, it means a delayed STT packet arrived after a newer turn was already processed — and it was correctly dropped.

---

## Quick Log Grep Reference

After a test call, grep the log for these signals:

```bash
# All barge-in decisions
grep -E "barge_in_accepted|barge_in_rejected" app.log

# All committed turns (each should be one logical utterance)
grep "user_turn_committed" app.log

# Stale events dropped
grep "stale_turn_event_ignored" app.log

# Lock timeouts (should be zero in normal operation)
grep "turn_lock_timeout" app.log

# Pending queue overflows (should be rare)
grep "pending_queue_overflow_dropped" app.log
```

---

## Pass / Fail Checklist

| # | Scenario | Expected | Pass? |
|---|----------|----------|-------|
| 1 | Split utterance | 1 FSM turn for full sentence | |
| 2 | "uh" during TTS | Rejected as filler_only | |
| 3 | "no change that to coke" during TTS | Accepted, TTS interrupted | |
| 4 | Echo tail after TTS | Rejected via guard_window | |
| 5 | Speech after mark ack | Accepted as normal turn | |
| 6 | No stale turn mutations in logs | stale events dropped if any | |
