#
import re


ASK_OPTIONS_CORE_PAT = re.compile(
    r"""
    ^(
        what\s+options\s+do\s+(?:you|i)\s+have
        | what\s+are\s+(?:my\s+)?options
        | show\s+(?:me\s+)?options
        | give\s+me\s+options
        | what\s+options\s+are\s+available
        | what\s+choices\s+do\s+i\s+have
        | show\s+available\s+options
        | what\s+can\s+i\s+choose\s+from
    )
    """,
    re.I | re.X
)

ASK_OPTIONS_FOR_ENTITY_PAT = re.compile(
    r"""
    ^(
        what\s+options\s+do\s+you\s+have\s+for\s+\w+
        | what\s+\w+\s+options\s+do\s+you\s+have
        | what\s+are\s+the\s+options\s+for\s+\w+
        | show\s+(?:me\s+)?\w+\s+options
        | what\s+choices\s+are\s+there\s+for\s+\w+
    )
    """,
    re.I | re.X
)

ASK_OPTIONS_FOLLOWUP_PAT = re.compile(
    r"""
    ^(
        what\s+else\s+do\s+you\s+have
        | any\s+other\s+options
        | what\s+are\s+the\s+other\s+options
        | more\s+options
        | anything\s+else\s+available
    )$
    """,
    re.I | re.X
)

ASK_OPTIONS_CASUAL_PAT = re.compile(
    r"""
    ^(
        what\s+do\s+you\s+have
        | what\s+all\s+do\s+you\s+have
        | what\s+are\s+the\s+choices
        | what\s+can\s+i\s+get
    )$
    """,
    re.I | re.X
)


