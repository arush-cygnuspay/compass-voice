import re

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

# Used by the compound-number parser below.  Values come exclusively from
# NUMBER_WORDS so updates stay in lockstep.
_COMPOUND_TENS = {word: value for word, value in NUMBER_WORDS.items() if value in {20, 30, 40, 50, 60, 70, 80, 90}}
_COMPOUND_ONES = {word: value for word, value in NUMBER_WORDS.items() if 1 <= value <= 9}

SPECIAL_QUANTITIES = {
    "a": 1,
    "an": 1,
    "single": 1,
    "couple": 2,
}

UNIT_WORDS = (
    "dozens",
    "dozen",
    "pieces",
    "piece",
    "pcs",
    "pc",
    "orders",
    "order",
)

# Weight-style units used in deli/butcher menus.  These DO NOT collapse to
# integer quantities by default — they describe the *amount of an item*,
# not the *number of items*.  ``extract_weight_quantity`` returns the
# parsed weight in ounces so the caller (cart builder, checkout summary)
# can apply the correct pricing.
WEIGHT_UNIT_TO_OUNCES: dict[str, float] = {
    "oz": 1.0,
    "ozs": 1.0,
    "ounce": 1.0,
    "ounces": 1.0,
    "lb": 16.0,
    "lbs": 16.0,
    "pound": 16.0,
    "pounds": 16.0,
}
_WEIGHT_UNIT_PATTERN = "|".join(
    sorted(WEIGHT_UNIT_TO_OUNCES.keys(), key=len, reverse=True)
)

INCREMENTAL_PATTERNS = (
    r"\bone more\b",
    r"\banother one\b",
    r"\banother\b",
)

VAGUE_PATTERNS = (
    r"\bsome\b",
    r"\ba few\b",
    r"\bseveral\b",
)

UNIT_PATTERN = "|".join(sorted(UNIT_WORDS, key=len, reverse=True))
_LEADING_QUANTITY_PATTERN = re.compile(
    r"^(?P<token>\d+|a|an|single|couple|one|two|three|four|five|six|seven|eight|nine|ten)\b(?P<rest>.*)$"
)

_HALF_POUND_PATTERN = re.compile(r"\bhalf\s+(?:a\s+)?(?:lb|pound)s?\b")
_QUARTER_POUND_PATTERN = re.compile(r"\bquarter\s+(?:lb|pound)s?\b")
_DIGIT_WEIGHT_PATTERN = re.compile(
    rf"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{_WEIGHT_UNIT_PATTERN})\b"
)
_WORD_WEIGHT_PATTERN = re.compile(
    rf"\b(?P<word>[a-z]+(?:\s+[a-z]+)?)\s+(?P<unit>{_WEIGHT_UNIT_PATTERN})\b"
)


def _parse_compound_number_word(text: str) -> int | None:
    """Parse English compound number words up to 999 (e.g. "forty nine"
    → 49, "two hundred" → 200, "one hundred and five" → 105).
    Returns None if no parse is possible.
    """
    if not text:
        return None

    cleaned = re.sub(r"[-,]", " ", text.lower()).strip()
    if not cleaned:
        return None

    tokens = [tok for tok in cleaned.split() if tok and tok != "and"]
    if not tokens:
        return None

    if len(tokens) == 1:
        token = tokens[0]
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
        if token in SPECIAL_QUANTITIES:
            return SPECIAL_QUANTITIES[token]
        return None

    # Pattern: <ones> hundred [<tens-or-ones>...]
    if len(tokens) >= 2 and tokens[1] == "hundred":
        ones = _COMPOUND_ONES.get(tokens[0])
        if ones is None:
            return None
        total = ones * 100
        rest = tokens[2:]
        if not rest:
            return total
        sub = _parse_compound_number_word(" ".join(rest))
        if sub is None:
            return None
        return total + sub

    # Pattern: <tens> <ones>
    if len(tokens) == 2 and tokens[0] in _COMPOUND_TENS and tokens[1] in _COMPOUND_ONES:
        return _COMPOUND_TENS[tokens[0]] + _COMPOUND_ONES[tokens[1]]

    return None


def _first_numeric_token(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def _first_number_word(text: str) -> int | None:
    # Prefer the longest compound match first ("forty nine" before
    # "forty") to avoid silently dropping the ones digit.
    compound = _first_compound_number(text)
    if compound is not None:
        return compound

    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return value

    return None


def _first_compound_number(text: str) -> int | None:
    if not text:
        return None

    # Match runs like "two hundred and fifty seven", "forty nine",
    # "thirty-two".  We accept hyphens and the conjunction "and".
    pattern = re.compile(
        r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
        r"forty|fifty|sixty|seventy|eighty|ninety|hundred|and)[\s-]?){2,}",
        flags=re.IGNORECASE,
    )

    best: int | None = None
    for match in pattern.finditer(text):
        parsed = _parse_compound_number_word(match.group(0))
        if parsed is not None:
            best = parsed if best is None else max(best, parsed)
    return best


def extract_weight_quantity(text: str) -> dict | None:
    """Parse weight-style quantities like ``"half pound"``,
    ``"quarter lb"``, ``"6 oz"``, ``"two pounds"``.

    Returns ``{"value": float, "unit": "oz"|"lb", "ounces": float}`` or
    ``None`` if the text contains no weight phrase.  Pricing logic still
    lives downstream — this helper is purely a parser.
    """
    if not text:
        return None

    normalized = text.lower().strip()
    if not normalized:
        return None

    if _HALF_POUND_PATTERN.search(normalized):
        return {"value": 0.5, "unit": "lb", "ounces": 8.0}

    if _QUARTER_POUND_PATTERN.search(normalized):
        return {"value": 0.25, "unit": "lb", "ounces": 4.0}

    digit_match = _DIGIT_WEIGHT_PATTERN.search(normalized)
    if digit_match:
        value = float(digit_match.group("value"))
        unit_raw = digit_match.group("unit")
        ounces = WEIGHT_UNIT_TO_OUNCES[unit_raw] * value
        unit = "oz" if unit_raw in {"oz", "ozs", "ounce", "ounces"} else "lb"
        return {"value": value, "unit": unit, "ounces": ounces}

    word_match = _WORD_WEIGHT_PATTERN.search(normalized)
    if word_match:
        word_raw = word_match.group("word")
        unit_raw = word_match.group("unit")
        # Strip leading articles before parsing.
        cleaned_word = re.sub(r"^(?:a|an)\s+", "", word_raw).strip()
        value = normalize_quantity(cleaned_word)
        if value is not None and value > 0:
            ounces = WEIGHT_UNIT_TO_OUNCES[unit_raw] * value
            unit = "oz" if unit_raw in {"oz", "ozs", "ounce", "ounces"} else "lb"
            return {"value": float(value), "unit": unit, "ounces": ounces}

    return None


def _extract_dozen_quantity(text: str) -> int | None:
    if re.search(r"\bhalf dozen\b", text):
        return 6

    digit_match = re.search(r"\b(\d+)\s+dozen\b", text)
    if digit_match:
        return int(digit_match.group(1)) * 12

    for word, value in {**NUMBER_WORDS, **SPECIAL_QUANTITIES}.items():
        if re.search(rf"\b{re.escape(word)}\s+dozen\b", text):
            return value * 12

    if re.search(r"\ba dozen\b", text):
        return 12

    return None


def _extract_unit_quantity(text: str) -> int | None:
    digit_match = re.search(rf"\b(\d+)\s*(?:{UNIT_PATTERN})\b", text)
    if digit_match:
        return int(digit_match.group(1))

    for word, value in {**NUMBER_WORDS, **SPECIAL_QUANTITIES}.items():
        if re.search(rf"\b{re.escape(word)}\s+(?:{UNIT_PATTERN})\b", text):
            if re.search(rf"\b{re.escape(word)}\s+dozen\b", text):
                continue
            return value

    return None


def extract_leading_quantity_phrase(text: str) -> tuple[int, str, str] | None:
    normalized = (text or "").lower().strip()
    if not normalized:
        return None

    match = _LEADING_QUANTITY_PATTERN.match(normalized)
    if match is None:
        return None

    token = match.group("token").strip()
    rest = (match.group("rest") or "").strip(" .,!?:;-")
    if not token:
        return None

    value: int | None = None
    if token.isdigit():
        value = int(token)
    elif token in NUMBER_WORDS:
        value = NUMBER_WORDS[token]
    elif token in SPECIAL_QUANTITIES:
        value = SPECIAL_QUANTITIES[token]

    if value is None or value <= 0:
        return None

    return value, rest, token


def normalize_quantity(text: str) -> int | None:
    """
    Normalize quantity expressions into an integer.

    Examples:
    - "2" -> 2
    - "two" -> 2
    - "2 pcs" -> 2
    - "half dozen" -> 6
    - "one dozen" -> 12
    """

    if not text:
        return None

    text = text.lower().strip()

    dozen_quantity = _extract_dozen_quantity(text)
    if dozen_quantity is not None:
        return dozen_quantity

    unit_quantity = _extract_unit_quantity(text)
    if unit_quantity is not None:
        return unit_quantity

    if text.isdigit():
        return int(text)

    if text in NUMBER_WORDS:
        return NUMBER_WORDS[text]

    if text in SPECIAL_QUANTITIES:
        return SPECIAL_QUANTITIES[text]

    numeric_token = _first_numeric_token(text)
    if numeric_token is not None:
        return numeric_token

    number_word = _first_number_word(text)
    if number_word is not None:
        return number_word

    return None


def detect_quantity(text: str) -> dict | None:
    """
    Returns:
    {
        "type": "exact" | "incremental" | "vague",
        "value": int | None
    }
    """

    normalized = (text or "").lower().strip()
    if not normalized:
        return None

    value = normalize_quantity(normalized)
    if value is not None:
        quantity_type = "exact"
        if any(re.search(pattern, normalized) for pattern in INCREMENTAL_PATTERNS):
            quantity_type = "incremental"
        return {
            "type": quantity_type,
            "value": value,
        }

    if any(re.search(pattern, normalized) for pattern in INCREMENTAL_PATTERNS):
        return {
            "type": "incremental",
            "value": 1,
        }

    if any(re.search(pattern, normalized) for pattern in VAGUE_PATTERNS):
        return {
            "type": "vague",
            "value": None,
        }

    return None
