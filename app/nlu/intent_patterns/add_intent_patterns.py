import re

# Strong ADD verbs (high confidence)
ADD_STRONG_VERB_PAT = re.compile(
    r"""
    \b(
        add
        | give\s+me
        | get\s+me
        | order
        | put\s+(?:in|on)
        | place\s+an?\s+order\s+for
    )\b
    """,
    re.I | re.X
)

# Desire / choice phrasing (context-sensitive)
ADD_DESIRE_PAT = re.compile(
    r"""
    \b(
        i\s+(?:want|need|wanna)
        | i\s+would\s+like(?:\s+to\s+order)?
        | i'?d\s+like(?:\s+to\s+order)?
        | i'?ll\s+(?:take|have|get)
        | i'?ll\s+go\s+with
        | i\s+(?:choose|am\s+choosing)
        | i'?m\s+going\s+with
        | let\s+me\s+(?:get|have)
        | can\s+i\s+(?:get|have)
        | could\s+i\s+(?:get|have)
        | may\s+i\s+(?:get|have)
    )\b
    """,
    re.I | re.X
)

# Confirmation / selection phrases (follow-up only)
ADD_CONFIRMATION_PAT = re.compile(
    r"""
    \b(
        make\s+it
        | let\s+it\s+be
        | that'?ll\s+be
        | just
        | only
    )\b
    """,
    re.I | re.X
)

# Quantity + noun (dangerous but necessary)
QUANTITY_ITEM_PAT = re.compile(
    r"""
    \b(
        one|two|three|four|five|
        \d+
    )\s+\w+
    """,
    re.I | re.X
)

# Bare noun / implicit add (last resort)
# E.g. "coke", "sprite please", "chicken burger"
BARE_ITEM_PAT = re.compile(
    r"^\s*\w+(?:\s+\w+)*\s*(?:please)?\s*$",
    re.I
)

# Multi-item patterns (split AFTER intent)
# These should not decide intent, only entity splitting:
# Examples
# “burger and fries”
# “pizza, coke, and fries”
# “burger with fries”
#
# Rule:
# Intent first → then split items

ITEM_SEPARATOR_PAT = re.compile(
    r"\b(?:and|with|plus|,)\b",
    re.I
)

FILLER_PAT = re.compile(
    r"^(?:i\s+(?:want|need|wanna)|"
    r"i\s+would\s+like|"
    r"i'?d\s+like|"
    r"i'?ll\s+(?:take|have|get)|"
    r"let\s+me\s+(?:get|have)|"
    r"can\s+i\s+(?:get|have)|"
    r"may\s+i\s+(?:get|have)|"
    r"give\s+me|"
    r"add)\s+",
    re.I,
)


# def strip_fillers(text: str) -> str:
#     prev = None
#     while prev != text:
#         prev = text
#         text = FILLER_PAT.sub("", text).strip()
#     return text


# IF strong_add_verb:
#     intent = ADD_PRIMARY_ITEM
#
# ELIF desire_phrase:
#     IF state == ASKING_FOR_SIDE:
#         intent = ADD_FOLLOWUP_ITEM
#     ELSE:
#         intent = ADD_PRIMARY_ITEM
#
# ELIF confirmation_phrase:
#     IF state == ASKING_FOR_SIDE:
#         intent = ADD_FOLLOWUP_ITEM
#     ELSE:
#         intent = CONFIRM_OR_CLARIFY
#
# ELIF quantity_item OR bare_item:
#     IF state in (ASKING_FOR_SIDE, EXPECTING_ITEM):
#         intent = ADD_FOLLOWUP_ITEM
#     ELSE:
#         intent = NEED_CONFIRMATION


