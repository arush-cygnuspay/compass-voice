# app/services/multi_item_order_planner.py
"""MultiItemOrderPlanner — menu-aware heuristic splitter for compound utterances.

Fixes a critical failure mode where the existing slot-based parser misses items
when NLU fails to tag them (e.g. typos like "cicken", size-adjacent items without
explicit connectors, variant descriptors misread as quantities).

Example failure this fixes
--------------------------
User:  "i want a grilled cicken sandwich a large fries small onion rings
        a Tuna Melt and a 6 piece wings"

Old behaviour (slot-based parse):
  - NLU misses "cicken" → only Tuna Melt detected
  - "6 piece wings" → quantity=6, item="wings" (wrong)
  - "Onion" attached as modifier to Tuna Melt

New behaviour (this planner):
  - Splits into 5 spans via article/size-word/connector boundaries
  - Fuzzy-matches "cicken" → "chicken" (SequenceMatcher ratio ≥ 0.82)
  - Resolves "6 piece wings" → variant="6 piece", quantity=1
  - Returns ParsedMultiItemPlan with 5 ParsedOrderItems

Design
------
* Pure functions — no I/O, no side effects, never raises.
* Does NOT mutate cart, context, or state.
* LLM is not used; all matching is deterministic.
* Uses difflib.SequenceMatcher for character-level typo correction.
* Conservative fuzzy threshold (FUZZY_MIN_RATIO = 0.80).
* Logs compound-turn events to COMPASS_CHAT_EVENTS_JSONL_PATH (default
  logs/current/chat_turn_events.jsonl) via a background thread — caller
  never blocks on I/O.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from app.menu.store import MenuStore
    from app.menu.models import MenuItem, PricingVariant

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

FUZZY_MIN_RATIO: float = 0.80          # SequenceMatcher threshold for typo match
FUZZY_TOKEN_RATIO: float = 0.72        # Per-token threshold for token-level match
MAX_SPAN_WORDS: int = 8                # Ignore spans longer than this word count
MIN_ITEM_SPAN_WORDS: int = 1           # Ignore spans shorter than this
VARIANT_PIECE_RE = re.compile(         # "6 piece", "12 piece", "24pc", etc.
    r"^(\d+)\s*(?:pc|pcs|piece|pieces|oz|ounce|ounces|ct|count)$",
    re.IGNORECASE,
)

# Words that attach a following element to the previous item (not new item)
_ATTACHMENT_WORDS: frozenset[str] = frozenset({
    "with", "extra", "more", "double", "no", "without", "light",
    "less", "hold", "remove", "add", "side", "on",
})
# Hard connectors that always signal a new item segment
_HARD_CONNECTORS: frozenset[str] = frozenset({"and", "plus", "also", "then"})
# Articles that signal a new item when NOT preceded by an attachment word
_ARTICLES: frozenset[str] = frozenset({"a", "an"})
# Size words that signal a new item when NOT preceded by an attachment word
_SIZE_WORDS: frozenset[str] = frozenset({"small", "medium", "large", "regular", "xl", "lg", "sm", "md"})
# Quantity words → integer value (used in span-level extraction)
_QUANTITY_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "dozen": 12,
}
# Filler prefixes stripped before processing (mirrors normalize_item_request_text)
_FILLER_PREFIXES: tuple[str, ...] = (
    "i want ", "i'd like ", "i would like ", "can i get ", "can i have ",
    "give me ", "let me get ", "let me have ", "i'll have ", "i'll take ",
    "i will have ", "add ", "also add ", "please add ", "get me ",
    "could i get ", "could i have ", "may i have ", "may i get ",
    "i want to order ", "id like ",
)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedOrderItem:
    """One successfully resolved item span from a compound utterance.

    Attributes
    ----------
    raw_span:       The raw text segment that produced this item.
    item_id:        Menu item ID (empty string when not found in menu).
    item_name:      Menu item display name.
    quantity:       Order quantity (always ≥ 1).
    size_id:        Variant/size ID when a size was detected (may be empty).
    size_name:      Size display label (e.g. "Large", "6 Piece").
    variant_id:     Same as size_id (alias used by some callers).
    variant_name:   Same as size_name (alias used by some callers).
    modifiers:      Modifier text tokens found in the span (informational only).
    sides:          Side text tokens found in the span (informational only).
    confidence:     Matching confidence 0.0–1.0.
    match_type:     "exact", "alias", "fuzzy", or "voice_label".
    """

    raw_span: str
    item_id: str
    item_name: str
    quantity: int = 1
    size_id: str = ""
    size_name: str = ""
    variant_id: str = ""
    variant_name: str = ""
    modifiers: tuple[str, ...] = ()
    sides: tuple[str, ...] = ()
    confidence: float = 1.0
    match_type: str = "exact"


@dataclass(frozen=True, slots=True)
class ParsedMultiItemPlan:
    """Result of plan_multi_item_order().

    Attributes
    ----------
    items:              Successfully resolved items.
    unresolved_spans:   Candidate spans that could not be matched to any menu item.
    confidence:         Overall plan confidence (mean of item confidences).
    reason:             Short human-readable explanation (e.g. "5_spans_4_resolved").
    is_compound:        True when the utterance contained ≥ 2 resolved items.
    raw_spans:          All candidate spans (resolved + unresolved), in order.
    """

    items: tuple[ParsedOrderItem, ...]
    unresolved_spans: tuple[str, ...]
    confidence: float
    reason: str
    is_compound: bool
    raw_spans: tuple[str, ...] = ()


# Empty singleton
_EMPTY_PLAN = ParsedMultiItemPlan(
    items=(),
    unresolved_spans=(),
    confidence=0.0,
    reason="no_spans",
    is_compound=False,
)

# ---------------------------------------------------------------------------
# Background JSONL event logger
# ---------------------------------------------------------------------------

_CHAT_EVENTS_LOG_PATH = os.getenv(
    "COMPASS_CHAT_EVENTS_JSONL_PATH",
    os.path.join("logs", "current", "chat_turn_events.jsonl"),
)

# Single daemon writer thread shared across the process lifetime
_log_queue: queue.Queue[dict | None] = queue.Queue(maxsize=1000)
_writer_started = False
_writer_lock = threading.Lock()


def _writer_loop(path: str) -> None:
    """Background writer daemon — one log line per record."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:  # pragma: no cover
        pass

    while True:
        try:
            record = _log_queue.get(timeout=5)
            if record is None:
                break
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        except queue.Empty:
            continue
        except Exception:  # pragma: no cover
            logger.debug("chat_events writer error", exc_info=True)


def _ensure_writer() -> None:
    global _writer_started
    with _writer_lock:
        if not _writer_started:
            t = threading.Thread(
                target=_writer_loop,
                args=(_CHAT_EVENTS_LOG_PATH,),
                daemon=True,
                name="chat_events_writer",
            )
            t.start()
            _writer_started = True


def _log_plan_event(
    *,
    transcript: str,
    plan: ParsedMultiItemPlan,
    session_id: str | None = None,
    state: str | None = None,
) -> None:
    """Non-blocking — enqueues a record; never raises."""
    try:
        _ensure_writer()
        record: dict[str, Any] = {
            "event": "multi_item_plan",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id or "",
            "state": state or "",
            "transcript": transcript,
            "is_compound": plan.is_compound,
            "raw_spans": list(plan.raw_spans),
            "resolved_count": len(plan.items),
            "unresolved_count": len(plan.unresolved_spans),
            "unresolved_spans": list(plan.unresolved_spans),
            "confidence": plan.confidence,
            "reason": plan.reason,
            "items": [
                {
                    "item_id": it.item_id,
                    "item_name": it.item_name,
                    "quantity": it.quantity,
                    "size_name": it.size_name,
                    "variant_name": it.variant_name,
                    "confidence": it.confidence,
                    "match_type": it.match_type,
                    "raw_span": it.raw_span,
                }
                for it in plan.items
            ],
        }
        _log_queue.put_nowait(record)
    except Exception:  # pragma: no cover
        pass  # Never block the call path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_multi_item_order(
    transcript: str,
    menu_store: "MenuStore | None",
    *,
    state: str | None = None,
    session_id: str | None = None,
    cart_snapshot: Any = None,  # reserved for future use
    smart_plan: Any = None,     # SmartTurnPlan | None, reserved for future use
) -> ParsedMultiItemPlan:
    """Parse a compound utterance into a structured multi-item plan.

    Parameters
    ----------
    transcript:
        Raw or lightly-normalized user utterance.
    menu_store:
        MenuStore for item lookup. When None, only exact-text heuristics
        are used (useful in unit tests without a live menu).
    state:
        Current FSM state (for logging only).
    session_id:
        Session identifier (for logging only).
    cart_snapshot:
        Reserved — not used in this implementation.
    smart_plan:
        SmartTurnPlan from upstream (reserved — not applied in this impl).

    Returns
    -------
    ParsedMultiItemPlan with resolved items and unresolved spans.
    Returns _EMPTY_PLAN (is_compound=False, items=()) when fewer than
    2 spans are found.  Never raises.
    """
    try:
        return _plan(transcript, menu_store, state=state, session_id=session_id)
    except Exception:
        logger.exception("multi_item_order_planner: unexpected error — falling through")
        return _EMPTY_PLAN


def resolve_quantity_and_variant(
    raw_span: str,
    matched_item: "MenuItem | None",
    *,
    quantity_override: int | None = None,
) -> tuple[int, str, str]:
    """Separate order quantity from variant/size descriptor.

    Handles the "6 piece wings" problem:
      - "6 piece wings" → quantity=1, variant_id="6_piece", variant_name="6 Piece"
      - "2 large fries"  → quantity=2, variant_id="large", variant_name="Large"
      - "3 burgers"      → quantity=3, variant_id="", variant_name=""

    Parameters
    ----------
    raw_span:
        The raw text for this item (after stripping leading articles/fillers).
    matched_item:
        The resolved MenuItem (None when not found).
    quantity_override:
        When set, use this quantity regardless of detected leading number.

    Returns
    -------
    (quantity, variant_id, variant_name) — quantity ≥ 1.
    """
    if quantity_override is not None:
        return max(1, quantity_override), "", ""

    text = raw_span.strip().lower()
    variants = _get_variants(matched_item)

    # --- 1. Try to match a variant label directly from the span ----------
    for variant in variants:
        v_label = variant.normalized_label.lower()
        if v_label and v_label in text:
            # Variant found — quantity is whatever's LEFT after removing variant
            remainder = text.replace(v_label, "").strip()
            qty = _parse_leading_quantity(remainder) or 1
            return qty, variant.variant_id, variant.label
        # Also check "6 piece" style: leading digit + unit
        m = VARIANT_PIECE_RE.match(v_label)
        if m:
            digit = m.group(1)
            if re.search(rf"\b{re.escape(digit)}\b", text):
                # Check the unit word too
                unit = v_label[len(digit):].strip()
                if unit and unit.rstrip("s") in text:
                    remainder = re.sub(
                        rf"\b{re.escape(digit)}\s*{re.escape(unit)}s?\b",
                        "",
                        text,
                    ).strip()
                    qty = _parse_leading_quantity(remainder) or 1
                    return qty, variant.variant_id, variant.label

    # --- 2. Try leading number + "piece/pc/oz/count" pattern -------------
    m = re.match(
        r"^(\d+)\s+(piece|pieces|pc|pcs|oz|ounce|ounces|ct|count)\b",
        text,
        re.IGNORECASE,
    )
    if m:
        # Looks like a variant descriptor — treat as variant=1 order
        variant_label = f"{m.group(1)} {m.group(2)}"
        # Find the best matching variant
        best_v = _best_variant_for_label(variant_label, variants)
        if best_v:
            return 1, best_v.variant_id, best_v.label
        # No variant in menu — still treat as variant, not quantity
        return 1, "", variant_label

    # --- 3. Leading size word → size variant ------------------------------
    size_match = re.match(r"^(small|medium|large|regular|xl|lg|sm|md)\b", text, re.IGNORECASE)
    if size_match:
        size_word = size_match.group(1).lower()
        best_v = _best_variant_for_label(size_word, variants)
        qty, _, _ = _leading_quantity_after(text, size_match.end())
        if best_v:
            return max(1, qty), best_v.variant_id, best_v.label
        return max(1, qty), "", size_word

    # --- 4. Plain leading quantity ----------------------------------------
    # Parse raw digit without the 1-99 range guard (we clamp below).
    raw_qty: int | None = None
    m_qty = re.match(r"^(\d+)\b", text)
    if m_qty:
        raw_qty = int(m_qty.group(1))
    else:
        for word, val in _QUANTITY_WORDS.items():
            if word in _ARTICLES:
                continue
            if re.match(rf"^{re.escape(word)}\b", text, re.IGNORECASE):
                raw_qty = val
                break
    qty = raw_qty if raw_qty is not None else 1
    qty = max(1, min(qty, 99))
    return qty, "", ""


# ---------------------------------------------------------------------------
# Internal: span splitting
# ---------------------------------------------------------------------------


def _strip_filler(text: str) -> str:
    """Strip leading order-filler phrases; normalize whitespace."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    changed = True
    while changed and t:
        changed = False
        for prefix in _FILLER_PREFIXES:
            if t.startswith(prefix):
                t = t[len(prefix):].strip()
                changed = True
                break
    return t


def _tok(word: str) -> str:
    """Lowercase and strip trailing punctuation from a word token."""
    return word.lower().rstrip(".,!?;:")


def _split_spans(text: str) -> list[str]:
    """Split transcript into candidate item spans using boundary heuristics.

    Boundary signals (in priority order):
    1. Hard connectors: "and", "plus", "also", "then" — always split.
    2. Comma — always split.
    3. Bare article "a"/"an" NOT immediately preceded by an attachment word
       and when current span already has a content word (non-filler).
    4. Bare size word "small"/"medium"/"large"/etc. NOT immediately preceded
       by an attachment word and when current span already has a content word.

    This correctly segments:
      "a grilled cicken sandwich a large fries small onion rings a Tuna Melt
       and a 6 piece wings"
    → ["a grilled cicken sandwich", "a large fries", "small onion rings",
       "a tuna melt", "a 6 piece wings"]
    """
    # Split by commas first, then process each comma-chunk for further splits
    comma_parts = re.split(r",", text)
    raw_spans: list[str] = []

    for comma_part in comma_parts:
        tokens = comma_part.strip().split()
        if not tokens:
            continue

        current: list[str] = []

        for i, tok in enumerate(tokens):
            t = _tok(tok)
            prev_t = _tok(tokens[i - 1]) if i > 0 else ""

            # 1. Hard connector → flush current, skip connector token
            if t in _HARD_CONNECTORS and i > 0:
                if current:
                    raw_spans.append(" ".join(current))
                current = []
                continue

            # 2. Article boundary: "a"/"an" not after attachment word
            if (
                t in _ARTICLES
                and i > 0
                and current
                and prev_t not in _ATTACHMENT_WORDS
                and prev_t not in _HARD_CONNECTORS
            ):
                # Only split if current span has a content word
                if _has_content_word(current):
                    raw_spans.append(" ".join(current))
                    current = [tok]
                    continue

            # 3. Size word boundary: "small"/"large"/etc. not after attachment word
            if (
                t in _SIZE_WORDS
                and i > 0
                and current
                and prev_t not in _ATTACHMENT_WORDS
                and prev_t not in _HARD_CONNECTORS
                and prev_t not in _ARTICLES
            ):
                if _has_content_word(current):
                    raw_spans.append(" ".join(current))
                    current = [tok]
                    continue

            current.append(tok)

        if current:
            raw_spans.append(" ".join(current))

    return [s.strip() for s in raw_spans if s.strip()]


def _has_content_word(tokens: list[str]) -> bool:
    """Return True if the token list has at least one non-filler word."""
    non_filler = _ARTICLES | _SIZE_WORDS | _ATTACHMENT_WORDS | _HARD_CONNECTORS
    return any(_tok(t) not in non_filler for t in tokens)


# ---------------------------------------------------------------------------
# Internal: item matching (exact → alias → fuzzy)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace (no heavy pre-processing)."""
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _strip_leading_articles_and_quantity(text: str) -> tuple[int | None, str]:
    """Strip leading articles / quantity words; return (quantity, remainder).

    Examples:
      "a grilled chicken sandwich" → (None, "grilled chicken sandwich")
      "2 large fries"              → (2,    "large fries")
      "the tuna melt"              → (None, "tuna melt")
    """
    t = text.strip()
    # Strip "the", "an", "a" articles (not quantity-preserving)
    t = re.sub(r"^(?:the|an?)\s+", "", t, flags=re.IGNORECASE).strip()

    # Leading digit
    m = re.match(r"^(\d+)\s+", t)
    if m:
        return int(m.group(1)), t[m.end():].strip()

    # Leading quantity word (but NOT articles — already stripped)
    for word, qty in _QUANTITY_WORDS.items():
        if word in _ARTICLES:
            continue
        if re.match(rf"^{re.escape(word)}\s+", t, re.IGNORECASE):
            return qty, t[len(word):].strip()

    return None, t


def _find_best_match(
    span_core: str,
    menu_store: "MenuStore | None",
) -> tuple["MenuItem | None", float, str]:
    """Match a normalized span core against the menu.

    Returns (matched_item, confidence, match_type).
    match_type: "exact" | "alias" | "voice_label" | "fuzzy" | "none"
    """
    if not span_core or not menu_store:
        return None, 0.0, "none"

    norm = _normalize(span_core)
    if not norm:
        return None, 0.0, "none"

    # 1. Exact name match
    item = menu_store.find_item_exact(norm)
    if item is not None:
        return item, 1.0, "exact"

    # 2. Alias match
    alias_ids = menu_store.find_item_ids_by_alias(norm)
    if alias_ids:
        try:
            item = menu_store.get_item(alias_ids[0])
            return item, 1.0, "alias"
        except (KeyError, Exception):
            pass

    # 3. Voice label match
    vl_ids = menu_store.find_item_ids_by_voice_label(norm)
    if vl_ids:
        try:
            item = menu_store.get_item(vl_ids[0])
            return item, 1.0, "voice_label"
        except (KeyError, Exception):
            pass

    # 4. Fuzzy match across all discoverable items
    return _fuzzy_match(norm, menu_store)


def _fuzzy_match(
    norm_span: str,
    menu_store: "MenuStore",
) -> tuple["MenuItem | None", float, str]:
    """Character-level fuzzy match via SequenceMatcher.

    Also attempts token-level matching for partial overlaps
    (e.g. "tuna melt" ⊆ "classic tuna melt").
    """
    best_item: "MenuItem | None" = None
    best_ratio = 0.0

    for menu_item in menu_store.iter_discoverable_items():
        labels: list[str] = [menu_item.normalized_name]
        labels.extend(menu_item.normalized_aliases)
        labels.extend(menu_item.voice_labels)

        for label in labels:
            if not label:
                continue
            ratio = SequenceMatcher(None, norm_span, label).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = menu_item

            # Token containment: if all tokens of norm_span appear in label
            span_tokens = set(norm_span.split())
            label_tokens = set(label.split())
            if span_tokens and span_tokens <= label_tokens:
                # Subset match — score proportional to overlap
                containment_score = len(span_tokens) / max(len(label_tokens), 1)
                token_ratio = 0.65 + 0.35 * containment_score
                if token_ratio > best_ratio:
                    best_ratio = token_ratio
                    best_item = menu_item

            # Token containment in reverse: norm_span contains all label tokens
            if label_tokens and label_tokens <= span_tokens:
                containment_score = len(label_tokens) / max(len(span_tokens), 1)
                token_ratio = 0.65 + 0.35 * containment_score
                if token_ratio > best_ratio:
                    best_ratio = token_ratio
                    best_item = menu_item

    if best_ratio >= FUZZY_MIN_RATIO and best_item is not None:
        return best_item, best_ratio, "fuzzy"

    # Last attempt: single-token fuzzy — try each token of the span
    # against each word of every item name (catches "cicken"→"chicken")
    return _token_level_fuzzy(norm_span, menu_store, threshold=FUZZY_TOKEN_RATIO)


def _token_level_fuzzy(
    norm_span: str,
    menu_store: "MenuStore",
    threshold: float = FUZZY_TOKEN_RATIO,
) -> tuple["MenuItem | None", float, str]:
    """Per-token fuzzy match for single-character typos like 'cicken'→'chicken'."""
    span_tokens = norm_span.split()
    if not span_tokens:
        return None, 0.0, "none"

    best_item: "MenuItem | None" = None
    best_score = 0.0

    for menu_item in menu_store.iter_discoverable_items():
        labels: list[str] = [menu_item.normalized_name]
        labels.extend(menu_item.normalized_aliases)

        for label in labels:
            if not label:
                continue
            label_tokens = label.split()
            if not label_tokens:
                continue

            # For each span token, find best matching label token
            total_score = 0.0
            for stok in span_tokens:
                best_tok_ratio = max(
                    SequenceMatcher(None, stok, ltok).ratio()
                    for ltok in label_tokens
                )
                total_score += best_tok_ratio

            # Normalised score: average per-token match × coverage
            avg_score = total_score / len(span_tokens)
            # Penalise length mismatch (too many or too few label tokens)
            len_penalty = min(len(span_tokens), len(label_tokens)) / max(
                len(span_tokens), len(label_tokens)
            )
            combined = avg_score * (0.6 + 0.4 * len_penalty)

            if combined > best_score:
                best_score = combined
                best_item = menu_item

    if best_score >= threshold and best_item is not None:
        return best_item, best_score, "fuzzy"

    return None, 0.0, "none"


# ---------------------------------------------------------------------------
# Internal: variant / size extraction
# ---------------------------------------------------------------------------


def _get_variants(item: "MenuItem | None") -> "list[PricingVariant]":
    if item is None:
        return []
    pricing = getattr(item, "pricing", None)
    if pricing is None:
        return []
    return list(getattr(pricing, "variants", None) or [])


def _best_variant_for_label(
    label: str,
    variants: "list[PricingVariant]",
) -> "PricingVariant | None":
    """Find the best-matching variant for a label string."""
    norm_label = _normalize(label)
    if not norm_label or not variants:
        return None

    best: "PricingVariant | None" = None
    best_ratio = 0.0

    for v in variants:
        v_norm = _normalize(v.normalized_label)
        ratio = SequenceMatcher(None, norm_label, v_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = v

    if best_ratio >= 0.75:
        return best
    return None


def _extract_size_from_span(
    span_text: str,
    item: "MenuItem | None",
) -> tuple[str, str, str]:
    """Extract size/variant info from the full span text.

    Returns (size_id, size_name, cleaned_span_for_quantity_parse).
    """
    t = _normalize(span_text)
    variants = _get_variants(item)

    # Try each variant label as substring of the span
    for v in variants:
        v_norm = _normalize(v.normalized_label)
        if v_norm and v_norm in t:
            cleaned = t.replace(v_norm, "").strip()
            return v.variant_id, v.label, cleaned

    # Try size words
    for size_w in sorted(_SIZE_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(size_w)}\b", t):
            best_v = _best_variant_for_label(size_w, variants)
            cleaned = re.sub(rf"\b{re.escape(size_w)}\b", "", t).strip()
            if best_v:
                return best_v.variant_id, best_v.label, cleaned
            return "", size_w, cleaned

    # Try "N piece" / "N pc" / "N oz" patterns
    m = re.search(
        r"\b(\d+)\s*(piece|pieces|pc|pcs|oz|ounce|ounces|ct|count)\b",
        t,
        re.IGNORECASE,
    )
    if m:
        variant_label = f"{m.group(1)} {m.group(2)}"
        best_v = _best_variant_for_label(variant_label, variants)
        cleaned = t[: m.start()].strip() + " " + t[m.end():].strip()
        cleaned = cleaned.strip()
        if best_v:
            return best_v.variant_id, best_v.label, cleaned
        return "", variant_label, cleaned

    return "", "", t


def _parse_leading_quantity(text: str) -> int | None:
    """Extract leading quantity from text (digit or word). Return None if none."""
    t = text.strip()
    if not t:
        return None
    m = re.match(r"^(\d+)\b", t)
    if m:
        v = int(m.group(1))
        return v if 1 <= v <= 99 else None
    for word, qty in _QUANTITY_WORDS.items():
        if word in _ARTICLES:
            continue
        if re.match(rf"^{re.escape(word)}\b", t, re.IGNORECASE):
            return qty
    return None


def _leading_quantity_after(text: str, offset: int) -> tuple[int, int, str]:
    """After `offset` chars, try to parse a leading quantity. Returns (qty, end, rest)."""
    remainder = text[offset:].strip()
    qty = _parse_leading_quantity(remainder)
    if qty is not None:
        m = re.match(r"^(\d+|\w+)\b", remainder)
        end = offset + (m.end() if m else 0)
        rest = remainder[m.end():].strip() if m else remainder
        return qty, end, rest
    return 1, offset, remainder


# ---------------------------------------------------------------------------
# Internal: plan builder
# ---------------------------------------------------------------------------


def _plan(
    transcript: str,
    menu_store: "MenuStore | None",
    *,
    state: str | None,
    session_id: str | None,
) -> ParsedMultiItemPlan:
    """Core planner logic (factored out so plan_multi_item_order can wrap it)."""
    if not transcript or not transcript.strip():
        return _EMPTY_PLAN

    # Pre-process
    cleaned = _strip_filler(transcript)
    if not cleaned:
        return _EMPTY_PLAN

    # Split into candidate spans
    raw_spans = _split_spans(cleaned)
    if len(raw_spans) < 2:
        return _EMPTY_PLAN

    # Validate span count and word limits
    valid_spans = [
        s for s in raw_spans
        if MIN_ITEM_SPAN_WORDS <= len(s.split()) <= MAX_SPAN_WORDS
    ]
    if len(valid_spans) < 2:
        return _EMPTY_PLAN

    # Resolve each span
    items: list[ParsedOrderItem] = []
    unresolved: list[str] = []

    for span in valid_spans:
        parsed = _resolve_span(span, menu_store)
        if parsed is not None:
            items.append(parsed)
        else:
            unresolved.append(span)

    total = len(valid_spans)
    resolved = len(items)

    if resolved < 2:
        # Not enough items resolved to constitute a multi-item plan
        return _EMPTY_PLAN

    # Overall confidence: mean of item confidences
    confidence = sum(it.confidence for it in items) / resolved

    reason = f"{total}_spans_{resolved}_resolved"
    if unresolved:
        reason += f"_{len(unresolved)}_unresolved"

    plan = ParsedMultiItemPlan(
        items=tuple(items),
        unresolved_spans=tuple(unresolved),
        confidence=confidence,
        reason=reason,
        is_compound=True,
        raw_spans=tuple(valid_spans),
    )

    _log_plan_event(
        transcript=transcript,
        plan=plan,
        session_id=session_id,
        state=state,
    )

    return plan


def _resolve_span(
    span: str,
    menu_store: "MenuStore | None",
) -> ParsedOrderItem | None:
    """Try to resolve a single span to a ParsedOrderItem. Returns None on failure."""
    if not span.strip():
        return None

    # 1. Extract size/variant from the full span text before stripping
    size_id, size_name, span_for_qty = _extract_size_from_span(span, None)

    # 2. Strip leading article + quantity word to get core item text
    detected_qty, core_text = _strip_leading_articles_and_quantity(span)

    # Also try without the size word for matching purposes
    _, core_without_size = _strip_leading_articles_and_quantity(span_for_qty)
    if not core_without_size:
        core_without_size = core_text

    # 3. Try to match core text to menu item
    matched_item, confidence, match_type = _find_best_match(core_text, menu_store)
    if matched_item is None and core_without_size != core_text:
        matched_item, confidence, match_type = _find_best_match(core_without_size, menu_store)

    if matched_item is None:
        return None

    # 4. Now that we have the item, re-extract size/variant with item variants
    size_id, size_name, _ = _extract_size_from_span(span, matched_item)

    # 5. Resolve quantity vs variant
    qty, variant_id, variant_name = resolve_quantity_and_variant(
        span,
        matched_item,
        quantity_override=None,
    )

    # If extract_size_from_span found a size and resolve_quantity_and_variant didn't
    # find a variant, use the size info as the variant
    if size_name and not variant_name:
        variant_id = size_id
        variant_name = size_name

    # Detected quantity overrides if resolve_quantity_and_variant returned 1
    if detected_qty is not None and detected_qty > 1 and qty == 1 and not variant_name:
        qty = detected_qty

    # Clamp quantity to safe range
    qty = max(1, min(qty, 99))

    return ParsedOrderItem(
        raw_span=span,
        item_id=matched_item.item_id,
        item_name=matched_item.name,
        quantity=qty,
        size_id=size_id or variant_id,
        size_name=size_name or variant_name,
        variant_id=variant_id or size_id,
        variant_name=variant_name or size_name,
        confidence=round(confidence, 3),
        match_type=match_type,
    )
