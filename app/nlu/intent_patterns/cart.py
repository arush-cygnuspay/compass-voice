import re

SHOW_CART_PAT = re.compile(
    r"""
    ^(
        # existing
        what('?s|\s+is)\s+(?:in|on)\s+my\s+(?:cart|order)
        | what\s+did\s+i\s+order
        | what\s+do\s+i\s+have
        | what\s+have\s+i\s+ordered

        # 🔥 NEW (missing cases)
        | what('?s|\s+is)\s+my\s+order(\s+so\s+far)?
        | my\s+order(\s+so\s+far)?
        | show\s+(?:me\s+)?my\s+order(\s+so\s+far)?
        | what\s+is\s+in\s+my\s+order(\s+so\s+far)?

        # existing
        | show\s+(?:me\s+)?my\s+(?:cart|order|items)
        | list\s+(?:my\s+)?(?:cart|order|items)
        | display\s+(?:my\s+)?(?:cart|order)
        | read\s+(?:back\s+)?my\s+order
        | go\s+over\s+my\s+order
        | tell\s+me\s+what\s+(?:is|’s)?\s+in\s+my\s+(?:cart|order)
        | i\s+want\s+to\s+see\s+my\s+(?:cart|order)
        | let\s+me\s+see\s+my\s+(?:cart|order)
        | can\s+you\s+show\s+(?:me\s+)?my\s+(?:cart|order)
    )
    """,
    re.I | re.X
)


SHOW_TOTAL_PAT = re.compile(
    r"""
    ^(
        what('?s|\s+is)\s+(?:my\s+)?(?:total|bill|amount|price)
        | how\s+much\s+(?:is\s+it|do\s+i\s+owe|does\s+it\s+cost)
        | what\s+is\s+the\s+(?:total|bill|amount)
        | (?:order\s+)?total\s+(?:so\s+far|right\s+now)?
        | (?:current\s+)?total
        | bill\s+so\s+far
        | (?:my\s+)?order\s+total
        | total\??
        | bill\??
        | price\??
        | cost\??
    )
    """,
    re.I | re.X
)


CART_ITEM_QUERY_PAT = re.compile(
    r"""
    ^(
        do\s+i\s+have\s+\w+
        | did\s+i\s+order\s+\w+
        | did\s+i\s+add\s+\w+
        | am\s+i\s+getting\s+\w+
        | is\s+there\s+\w+\s+in\s+my\s+(?:cart|order)
        | are\s+there\s+\w+\s+in\s+my\s+(?:cart|order)
        | how\s+many\s+\w+\s+(?:do\s+i\s+have|are\s+in\s+my\s+(?:cart|order))
        | how\s+many\s+did\s+i\s+order
        | did\s+i\s+already\s+order\s+\w+
    )
    """,
    re.I | re.X
)


CLEAR_CART_PAT = re.compile(
    r"""
    ^(
        clear\s+(?:my\s+)?(?:cart|order)
        | empty\s+(?:my\s+)?(?:cart|order)
        | delete\s+(?:my\s+)?(?:cart|order)
        | remove\s+(?:everything|all\s+items)
        | cancel\s+the\s+(?:whole\s+)?order
        | delete\s+the\s+(?:whole\s+)?order
        | start\s+over
        | begin\s+again
        | reset\s+(?:my\s+)?order
    )
    """,
    re.I | re.X
)

