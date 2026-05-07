# app/config/voice_transfer.py
"""Shared configuration for human-agent call transfer."""
from __future__ import annotations

import os

_DEFAULT = "+17036371136"
HUMAN_AGENT_TRANSFER_NUMBER: str = (
    os.getenv("COMPASS_HUMAN_AGENT_TRANSFER_NUMBER", _DEFAULT).strip() or _DEFAULT
)
