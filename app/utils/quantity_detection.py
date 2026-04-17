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
}

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


def _first_numeric_token(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def _first_number_word(text: str) -> int | None:
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return value

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
