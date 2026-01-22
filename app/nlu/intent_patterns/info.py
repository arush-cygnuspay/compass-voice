import re

ASK_CATEGORY_LIST_PAT = re.compile(
    r"""
    ^(
        what\s+(?:kind|types|all)\s+of\s+\w+
        | what\s+\w+\s+do\s+you\s+have
        | what\s+are\s+the\s+\w+
        | which\s+\w+\s+(?:do\s+you\s+have|you\s+have|you\s+got)
        | show\s+(?:me\s+)?\w+
        | list\s+(?:all\s+)?\w+
        | tell\s+me\s+the\s+\w+
        | do\s+you\s+have\s+(?:any\s+)?\w+
        | have\s+you\s+got\s+(?:any\s+)?\w+
        | do\s+you\s+serve\s+\w+
        | you\s+serve\s+\w+
        | do\s+you\s+sell\s+\w+
        | you\s+sell\s+\w+
        | do\s+you\s+offer\s+\w+
        | you\s+offer\s+\w+
        | what\s+options\s+do\s+i\s+have\s+for\s+\w+
    )
    """,
    re.I | re.X
)


ASK_ITEM_INFO_PAT = re.compile(
    r"""
    ^(
        tell\s+me\s+about\s+\w+
        | what\s+is\s+\w+
        | describe\s+\w+
        | what'?s\s+in\s+\w+
        | does\s+\w+\s+have
        | how\s+is\s+\w+
        | is\s+\w+\s+(?:good|spicy|available)
    )
    """,
    re.I | re.X
)

ASK_VIEW_PAT = re.compile(
    r"""
    \b(
        i\s+want\s+to\s+(?:see|know|check|view)
        | i'?d\s+like\s+to\s+(?:see|know|check)
    )\b
    """,
    re.I | re.X
)

BARE_CATEGORY_HINT_PAT = re.compile(
    r"""
    ^\s*\w+s\s*$
    """,
    re.I
)

# Patterns that CAPTURE the entity
CATEGORY_ENTITY_PAT = re.compile(
    r"""
    (?:
        do\s+you\s+have(?:\s+any)?
        | have\s+you\s+got(?:\s+any)?
        | you\s+got(?:\s+any)?
        | got(?:\s+any)?
        | do\s+you\s+serve
        | you\s+serve
        | do\s+you\s+sell
        | you\s+sell
        | do\s+you\s+offer
        | you\s+offer
        | show\s+(?:me\s+)?
        | list\s+(?:all\s+)?
        | tell\s+me\s+the
        | what\s+(?:kind|types|all)\s+of
        | what\s+\w+\s+do\s+you\s+have
    )
    \s+(?P<entity>[\w\s]+)$
    """,
    re.I | re.X
)


BARE_ENTITY_PAT = re.compile(r"^[a-z\s]+s$", re.I)

SOFT_MENU_INQUIRY_PAT = re.compile(
    r"""
    ^(
        you\s+(?:got|serve|sell|offer)(?:\s+any)?\s+\w+
        | got(?:\s+any)?\s+\w+
        | any\s+\w+
    )$
    """,
    re.I | re.X
)

WHICH_CATEGORY_PAT = re.compile(
    r"""
    ^which\s+(?P<entity>[\w\s]+?)\s+
    (?:
        do\s+you\s+have
        | you\s+have
        | you\s+got
        | do\s+you\s+got      # rare but heard in ESL
        | you\s+serve
        | you\s+sell
        | available
    )$
    """,
    re.I | re.X
)


ASK_PRICE_PAT = re.compile(
    r"""
    \b(
        how\s+much\s+(?:is|are|does)\b
        | what'?s\s+the\s+price\s+of\b
        | price\s+of\b
        | how\s+much\s+does\b
        | cost\s+of\b
    )
    """,
    re.I | re.X
)



