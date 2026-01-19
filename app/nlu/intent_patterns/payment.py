# app/nlu/intent_patterns/payment.py
import re

PAYMENT_REQUEST_PAT = re.compile(
    r"""
    \b(
        pay\s+now |
        proceed\s+to\s+pay |
        send\s+(me\s+)?(the\s+)?payment\s+link |
        i\s+(want|need|would\s+like)\s+to\s+pay |
        let\s+me\s+pay |
        checkout
    )\b
    """,
    re.I | re.X,
)


PAYMENT_DONE_PAT = re.compile(
    r"""
    \b(
        ive\s+paid |
        i\s+paid |
        i(\s+|')?(have\s+)?paid |
        i\s+have\s+done\s+payment |
        paid\s+already |
        already\s+paid |

        payment\s+(is\s+)?(done|complete|completed|successful) |
        transaction\s+(is\s+)?completed |
        finished\s+payment |
        completed\s+payment |
        payment\s+successful |
        transaction\s+completed |

        # voice / conversational
        i(\s+|')?m\s+done |
        all\s+done |
        im\s+done |
        im\s+finished
        done\s+(with\s+)?payment
    )\b
    """,
    re.I | re.X,
)



