# app/nlu/intent_patterns/order.py
import re

"""
Precedence rules (MANDATORY)

This order prevents catastrophic mistakes:
1. ORDER_STATUS_PAT
2. ORDER_CONFIRM_EXPLICIT_PAT
3. END_ADDING_STRONG_PAT
4. END_ADDING_SOFT_PAT   (context-locked)
5. ORDER_META_CLARIFY_PAT
"""


# Stop asking me for more items
END_ADDING_STRONG_PAT = re.compile(
    r"""
    ^(
        that'?s\s+all
        | that'?s\s+it
        | that\s+is\s+all
        | that\s+is\s+it
        | nothing\s+else
        | no\s+more(\s+items?)?
        | i'?m\s+done
        | finished
        | done\s+ordering
        | done\s+with\s+my\s+order
    )\b
    """,
    re.I | re.X
)

ORDER_CONFIRM_EXPLICIT_PAT = re.compile(
    r"""
    ^(
        place\s+(?:my\s+)?order
        | confirm\s+(?:my\s+)?order
        | checkout
        | check\s+out
        | proceed\s+(?:to\s+checkout)?
        | go\s+ahead\s+and\s+place\s+(?:it|the\s+order)?
        | submit\s+order
        | finalize\s+order
        | complete\s+order
    )\b
    """,
    re.I | re.X
)

END_ADDING_SOFT_PAT = re.compile(
    r"""
    ^(
        no
        | nope
        | nah
        | no\s+thanks?
        | no\s+thank\s+you
        | i'?m\s+good
        | i'?m\s+good\s+thanks
        | all\s+good
        | that'?s\s+enough
        | that\s+should\s+be\s+enough
    )\b
    """,
    re.I | re.X
)

ORDER_STATUS_PAT = re.compile(
    r"""
    ^(
        is\s+my\s+order\s+(?:placed|confirmed)
        | has\s+my\s+order\s+been\s+(?:placed|confirmed)
        | did\s+my\s+order\s+go\s+through
        | did\s+it\s+go\s+through
        | is\s+it\s+confirmed
        | did\s+you\s+place\s+the\s+order
        | has\s+it\s+been\s+ordered
    )
    """,
    re.I | re.X
)

ORDER_META_CLARIFY_PAT = re.compile(
    r"""
    ^(
        right\??
        | correct\??
        | that'?s\s+correct\s+right\??
        | is\s+that\s+all\??
        | so\s+that'?s\s+it\??
        | just\s+one\??
        | only\s+that\??
    )$
    """,
    re.I | re.X
)


