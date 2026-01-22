# app/nlu/intent_patterns/common.py
import re

# YES_STRONG_PAT (VERY HIGH confidence)
YES_STRONG_PAT = re.compile(
    r"""
    ^(
        y
        | yes
        | yeah
        | yep
        | yup
        | yah
    )\b
    """,
    re.I | re.X
)


YES_ACTION_PAT = re.compile(
    r"""
    ^(
        go\s+ahead
        | proceed
        | do\s+it
        | continue
        | carry\s+on
        | move\s+forward
        | confirm
        | please\s+do
        | do\s+that
    )\b
    """,
    re.I | re.X
)

YES_SOFT_PAT = re.compile(
    r"""
    ^(
        ok
        | okay
        | alright
        | all\s+right
        | sure
        | fine
        | perfect
        | great
        | awesome
        | sounds\s+good
        | looks\s+good
        | works(\s+for\s+me)?
        | that'?s\s+(fine|okay|good|correct)
        | no\s+problem
    )\b
    """,
    re.I | re.X
)

#--------------------------------
# No/Negation Patterns
#--------------------------------

NO_STRONG_PAT = re.compile(
    r"""
    ^(
        n
        | no
        | nope
        | nah
        | na
    )\b
    """,
    re.I | re.X
)


NO_CANCEL_PAT = re.compile(
    r"""
    ^(
        cancel
        | cancel\s+that
        | stop
        | stop\s+it
        | never\s?mind
        | forget\s+it
        | scratch\s+that
        | undo\s+that
    )\b
    """,
    re.I | re.X
)

NO_NEGATION_PAT = re.compile(
    r"""
    ^(
        don'?t
        | do\s+not
        | no\s+need
        | no\s+thanks?
        | i\s+don'?t\s+(?:want|need)
        | i'?d\s+rather\s+not
        | not\s+really
        | i\s+don'?t\s+think\s+so
    )
    """,
    re.I | re.X
)


NO_REVERSAL_PAT = re.compile(
    r"""
    ^(
        actually\s+no
        | wait\s+no
        | sorry\s+no
        | i\s+mean\s+no
        | change\s+that
        | let\s+me\s+change\s+that
        | i\s+changed\s+my\s+mind
    )\b
    """,
    re.I | re.X
)

NO_DEFER_PAT = re.compile(
    r"""
    ^(
        later
        | maybe\s+later
        | not\s+now
        | not\s+right\s+now
        | another\s+time
        | hold\s+on
        | wait
    )\b
    """,
    re.I | re.X
)
