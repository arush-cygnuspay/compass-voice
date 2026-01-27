import re

PURE_QUANTITY_PAT = re.compile(
    r"""
    ^(
        \d+ |
        one | two | three | four | five |
        six | seven | eight | nine | ten |
        a | an |
        single |
        couple |
        few
    )$
    """,
    re.I | re.X
)

QUANTITY_NOUN_PAT = re.compile(
    r"""
    \b(
        \d+ |
        one | two | three | four | five |
        six | seven | eight | nine | ten |
        a | an |
        couple | few
    )
    \s+
    (?P<item>\w+)
    """,
    re.I | re.X
)

SOFT_QUANTITY_PAT = re.compile(
    r"""
    \b(
        just |
        only |
        make\s+it |
        give\s+me |
        add
    )\s+
    (
        \d+ |
        one | two | three | four | five |
        six | seven | eight | nine | ten |
        a | an
    )
    """,
    re.I | re.X
)

INCREMENTAL_QUANTITY_PAT = re.compile(
    r"""
    \b(
        one\s+more |
        two\s+more |
        another\s+one |
        add\s+one\s+more |
        add\s+another
    )\b
    """,
    re.I | re.X
)

VAGUE_QUANTITY_PAT = re.compile(
    r"""
    \b(
        few |
        some |
        several |
        many |
        a\s+lot
    )\b
    """,
    re.I | re.X
)

MINIMIZE_QUANTITY_PAT = re.compile(
    r"""
    \b(
        just\s+one |
        only\s+one |
        make\s+it\s+one |
        give\s+me\s+one |
        add\s+one
    )\b
    """,
    re.I | re.X
)
