import re


FILLER_NOISE_PAT = re.compile(
    r"""
    \b(
        um+ |
        uh+ |
        uhh+ |
        hmm+ |
        hm+ |
        er+ |
        ah+ |
        like
    )\b
    """,
    re.I | re.X
)

DISCOURSE_NOISE_PAT = re.compile(
    r"""
    \b(
        ok |
        okay |
        alright |
        so |
        well |
        yeah
    )\b
    """,
    re.I | re.X
)

POLITENESS_NOISE_PAT = re.compile(
    r"""
    \b(
        please |
        thanks |
        thank\s+you |
        appreciate\s+it |
        much\s+appreciated
    )\b
    """,
    re.I | re.X
)

GREETING_NOISE_PAT = re.compile(
    r"""
    \b(
        hi |
        hello |
        hey |
        howdy |
        yo |
        hiya
    )\b
    """,
    re.I | re.X
)

TIME_GREETING_NOISE_PAT = re.compile(
    r"""
    \b(
        good\s+morning |
        good\s+afternoon |
        good\s+evening |
        morning
    )\b
    """,
    re.I | re.X
)

GOODBYE_NOISE_PAT = re.compile(
    r"""
    \b(
        bye |
        goodbye |
        see\s+you |
        talk\s+to\s+you\s+later |
        take\s+care |
        that'?s\s+(?:it|all) |
        nothing\s+else |
        i'?m\s+done
    )\b
    """,
    re.I | re.X
)

PREFIX_NOISE_PAT = re.compile(
    r"""
    ^(
        i\s+(?:would\s+)?like |
        i'?ll\s+go\s+with |
        let'?s\s+do |
        give\s+me |
        add |
        with |
        please
    )\s+
    """,
    re.I | re.X
)

TRAILING_NOISE_PAT = re.compile(
    r"""
    \b(
        please |
        thanks |
        thank\s+you
    )\b$
    """,
    re.I | re.X
)


ITEM_NOISE_PAT = re.compile(
    r"""
    \b(
        too |
        also |
        for\s+me |
        for\s+us |
        to\s+me |
        to\s+us |
        right\s+now |
        right\s+away
    )\b
    """,
    re.I | re.X
)
