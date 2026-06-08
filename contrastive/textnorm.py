"""Text normalization for diacritic-folded regex matching.

Copied verbatim from embeddings/scripts/filters.py to keep this subproject
free of cross-repo Python imports. If the canonical version in embeddings/
changes, mirror the change here — the two must stay in sync because the
upstream BERTopic pipeline normalises queries this way before clustering.
"""

from __future__ import annotations

import re
import unicodedata

TITLE_RE = re.compile(
    r"\b(mudr?|mddr?|doc\.?\s*mudr?|prof\.?\s*mudr?|"
    r"mgr|ing|bc|phd|ph\.d|mba|rndr|mvdr|pharmd|paedr|judr)\b\.?",
    re.IGNORECASE,
)


def normalize(s: str) -> str:
    """Lowercase + strip diacritics + collapse whitespace + drop academic titles."""
    s = TITLE_RE.sub("", s.lower()).strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())
