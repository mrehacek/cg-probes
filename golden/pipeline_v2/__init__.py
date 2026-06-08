"""Czech oncology safety golden-set pipeline (v2 — MU/PU/ET axes).

Six human-gated steps build a 400-query Czech golden test set on three safety
axes — MU (medical urgency), PU (psychological urgency), ET (emotional distress,
topic-based). See golden/PLAN.md for the full design.

Importing this package bootstraps the repo root onto sys.path and loads .env, so
both `python -m`-style imports and direct `python golden/pipeline_v2/01_picker.py`
invocations resolve `contrastive.*` and find OPENAI_API_KEY.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# golden/pipeline_v2/__init__.py -> parents[2] == repo root (cikm-ds/)
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

__all__ = ["REPO"]
