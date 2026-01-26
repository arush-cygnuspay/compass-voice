import re

# Strong REMOVE verbs (high confidence)
REMOVE_STRONG_VERB_PAT = re.compile(
    r"""
    \b(
        remove
        | delete
        | take\s+out
        | take\s+off
        | drop
        | get\s+rid\s+of
        | cancel
    )\b
    """,
    re.I | re.X
)

# Desire / choice phrasing for removal
REMOVE_DESIRE_PAT = re.compile(
    r"""
    \b(
        i\s+(?:don'?t|do\s+not)\s+want
        | i\s+(?:don'?t|do\s+not)\s+need
        | i\s+want\s+to\s+remove
        | i\s+want\s+to\s+delete
        | i\s+want\s+to\s+cancel
        | i'?d\s+like\s+to\s+remove
        | i'?d\s+like\s+to\s+delete
        | can\s+i\s+remove
        | can\s+i\s+delete
        | could\s+i\s+remove
        | could\s+i\s+delete
    )\b
    """,
    re.I | re.X
)

# Remove item patterns (item name follows)
REMOVE_ITEM_PAT = re.compile(
    r"""
    \b(
        remove\s+\w+(?:\s+\w+)*
        | delete\s+\w+(?:\s+\w+)*
        | take\s+out\s+\w+(?:\s+\w+)*
        | take\s+off\s+\w+(?:\s+\w+)*
        | drop\s+\w+(?:\s+\w+)*
        | cancel\s+\w+(?:\s+\w+)*
        | get\s+rid\s+of\s+\w+(?:\s+\w+)*
    )\b
    """,
    re.I | re.X
)

# Filler patterns for remove item (to strip from user text)
REMOVE_FILLER_PAT = re.compile(
    r"^(?:i\s+(?:don'?t|do\s+not)\s+(?:want|need)|"
    r"i\s+want\s+to\s+(?:remove|delete|cancel)|"
    r"i'?d\s+like\s+to\s+(?:remove|delete)|"
    r"can\s+i\s+(?:remove|delete)|"
    r"could\s+i\s+(?:remove|delete)|"
    r"remove|"
    r"delete|"
    r"take\s+out|"
    r"take\s+off|"
    r"drop|"
    r"cancel|"
    r"get\s+rid\s+of)\s+",
    re.I,
)

