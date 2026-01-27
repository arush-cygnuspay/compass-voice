from app.nlu.intent_patterns.quantity import *


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
}

SPECIAL_QUANTITIES = {
    "a": 1,
    "an": 1,
    "single": 1,
    "couple": 2,
}

def normalize_quantity(text: str) -> int | None:
    """
    Normalize quantity expressions into an integer.

    Examples:
    - "2" -> 2
    - "two" -> 2
    - "a" -> 1
    - "just one" -> 1
    - "only two" -> 2
    - "one more" -> 1
    - "another one" -> 1
    """

    if not text:
        return None

    text = text.lower().strip()

    # ----------------------------------
    # 1️⃣ Pure digit
    # ----------------------------------
    if text.isdigit():
        return int(text)

    # ----------------------------------
    # 2️⃣ Exact word match
    # ----------------------------------
    if text in NUMBER_WORDS:
        return NUMBER_WORDS[text]

    if text in SPECIAL_QUANTITIES:
        return SPECIAL_QUANTITIES[text]

    # ----------------------------------
    # 3️⃣ Extract first numeric token
    # ----------------------------------
    tokens = re.findall(r"\b\d+\b", text)
    if tokens:
        return int(tokens[0])

    # ----------------------------------
    # 4️⃣ Extract first number word
    # ----------------------------------
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return value

    for word, value in SPECIAL_QUANTITIES.items():
        if re.search(rf"\b{word}\b", text):
            return value

    # ----------------------------------
    # 5️⃣ No usable quantity
    # ----------------------------------
    return None

def detect_quantity(text: str) -> dict | None:
    """
    Returns:
    {
        "type": "exact" | "incremental" | "vague",
        "value": int | None
    }
    """
    if PURE_QUANTITY_PAT.match(text):
        return {"type": "exact", "value": normalize_quantity(text)}

    if INCREMENTAL_QUANTITY_PAT.search(text):
        return {"type": "incremental", "value": normalize_quantity(text)}

    if SOFT_QUANTITY_PAT.search(text) or QUANTITY_NOUN_PAT.search(text):
        return {"type": "exact", "value": normalize_quantity(text)}

    if VAGUE_QUANTITY_PAT.search(text):
        return {"type": "vague", "value": None}

    return None
