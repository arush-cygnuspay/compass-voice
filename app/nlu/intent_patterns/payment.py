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
        i(\s+|')?ve\s+paid |
        i\s+paid |
        paid\s+already |
        already\s+paid |
        payment\s+(is\s+)?done |
        payment\s+completed |
        payment\s+successful |
        transaction\s+completed |
        finished\s+payment |
        completed\s+payment
    )\b
    """,
    re.I | re.X,
)


