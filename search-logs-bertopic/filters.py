"""Shared query-filter helpers — the noise-removal layer before embedding.

All filters operate on the normalized query form (lowercase + NFD-strip
diacritics), matching the `query_key` convention in the unified CSV.

Layers:
  * `is_title_query`       — regex on academic/medical titles (MUDr., PhD, …)
  * `is_employee_name`     — exact/fuzzy match against the employee list
  * `is_extreme_navigational` — curated list of hospital brand/navigational
                                queries that dominate click counts and would
                                swamp any topic model
  * `is_digit_query`       — pure digits (phone numbers, ids)
  * `is_spam_query`        — SQL/XSS injection probes + SEO pharma spam
  * `mark_internal_shorthand` — DataFrame-level: short queries seen only on
                                site1_internal (staff/power-user shorthand
                                like `Uro`, `Mam`, `gynek`, `aro` — useless
                                for question-variant generation)

`EmployeeFilter` holds the expensive state (employee name set + fuzzy
matcher) so callers pay that cost once. `NavigationalFilter` holds the
curated list so the notebook can inject extras at runtime.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


# ── Normalization ────────────────────────────────────────────────────────────

TITLE_RE = re.compile(
    r"\b(mudr?|mddr?|doc\.?\s*mudr?|prof\.?\s*mudr?|"
    r"mgr|ing|bc|phd|ph\.d|mba|rndr|mvdr|pharmd|paedr|judr)\b\.?",
    re.IGNORECASE,
)


def normalize(s: str) -> str:
    """Lowercase + strip diacritics + collapse whitespace. Strips academic
    titles too so 'MUDr. Novak' and 'novak' normalize the same."""
    s = TITLE_RE.sub("", s.lower()).strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def is_title_query(q: str) -> bool:
    """True if the query contains an academic/medical title."""
    return bool(TITLE_RE.search(q))


# ── Employee filter ──────────────────────────────────────────────────────────


class EmployeeFilter:
    """Match queries against the employee list (first/last/fullname).

    Exact match on full normalized string or single tokens ≥4 chars, plus a
    fuzzy RapidFuzz pass at cutoff 88 for short (<=3-word) queries.
    """

    def __init__(self, employees_json: Path | str):
        with open(employees_json, encoding="utf-8") as f:
            data = json.load(f)["data"]
        variants: set[str] = set()
        for emp in data:
            fn = normalize(emp.get("firstname") or "")
            ln = normalize(emp.get("lastname") or "")
            if fn:
                variants.add(fn)
            if ln:
                variants.add(ln)
            if fn and ln:
                variants.add(f"{fn} {ln}")
                variants.add(f"{ln} {fn}")
        variants.discard("")
        self.variants = variants
        self._variants_list = list(variants)

    def __call__(self, q: str) -> bool:
        n = normalize(q)
        if n in self.variants:
            return True
        for tok in n.split():
            if len(tok) >= 4 and tok in self.variants:
                return True
        if len(n.split()) <= 3:
            hit = process.extractOne(
                n, self._variants_list, scorer=fuzz.ratio, score_cutoff=88
            )
            return hit is not None
        return False


# ── Extreme navigational filter ──────────────────────────────────────────────


# Curated after looking at the top-50 by clicks in the aggregated CSV. These
# are hospital / site brand queries that dominate totals and carry no topical
# information (patients searching the hospital name or "modra kniha" are
# navigating, not asking a question we can learn from).
#
# Keys are already normalized (casefold + diacritics-stripped + title-stripped)
# to match what `normalize()` produces. Match is exact.
DEFAULT_EXTREME_NAVIGATIONAL: frozenset[str] = frozenset({
    # hospital identity
})


class NavigationalFilter:
    """Drop extreme-navigational brand/hospital queries.

    The default set is a curated list from the top-clicked queries; pass
    `extra` to add domain-specific ones from the notebook, or pass
    `override` to replace the default entirely.
    """

    def __init__(
        self,
        extra: set[str] | None = None,
        override: set[str] | None = None,
    ):
        base = set(override) if override is not None else set(DEFAULT_EXTREME_NAVIGATIONAL)
        if extra:
            base |= set(extra)
        # Normalize the input list so callers don't have to.
        self.blocklist = {normalize(x) for x in base if x}

    def __call__(self, q: str) -> bool:
        return normalize(q) in self.blocklist


# ── Digit / spam / shorthand filters ─────────────────────────────────────────


_DIGITS_ONLY = re.compile(r"^\d{3,}$")


def is_digit_query(q: str) -> bool:
    """True if the query is just digits (phone numbers, ids, ZIP codes)."""
    return bool(_DIGITS_ONLY.match((q or "").strip()))


# Spam patterns observed in site1_internal logs:
#   - SQL injection probes: '-1 OR 5*5=25 --, (select(0)from(select(sleep(15)))v)
#   - Blind SQLi scanner: 'cktql"a="b"f1z2x  (multiple consecutive quote chars)
#   - SEO pharma spam: CONTACT colaship.shop [ BUY COCAINE ONLINE ] DARKNET STORE
_SPAM = re.compile(
    r"""(
        ['"`]{2,}                                               # ≥2 consecutive quote chars (injection probe)
      | \bselect\s*\(                                           # SQL select(
      | \bsleep\s*\(\s*\d                                       # SQL sleep(
      | \bor\s+\d+\s*[*=]\s*\d+                                 # 'OR 5*5=25', 'OR 1=1'
      | \b(?:buy|contact)\s+(?:\w+\s+){0,3}(?:online|here)\b    # SEO pharma 'BUY X ONLINE'
      | \b(?:cocaine|mdma|xanax|valium|adderall|oxycontin
          |ritalin|suboxone|darknet)\b
      | \b(?:colaship|pharmacypills)\.shop\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def is_spam_query(q: str) -> bool:
    """True if the query looks like SQL/XSS scanner output or SEO pharma spam."""
    return bool(_SPAM.search(q or ""))


def mark_internal_shorthand(
    df: pd.DataFrame,
    *,
    max_len: int = 4,
    key_col: str = "query_key",
    ga4_cols: tuple[str, ...] = ("clicks_site1_ga4",),
    internal_col: str = "clicks_site1_internal",
) -> pd.Series:
    """Flag staff/power-user shorthand queries on site1_internal.

    Heuristic: short query (`len(query_key) ≤ max_len`) that has been clicked
    on the internal search but never via GA4. Real medical short tokens
    (`picc`, `tnm`, `ecog`, `egfr`, `mrsa`, `ercp`, `mri`) all show GA4
    traffic and pass through; staff abbreviations (`Uro`, `Mam`, `gynek`,
    `aro`, `Orl`) don't and get dropped. They cluster densely (everyone
    typing `Urologie` agrees on the embedding) but are unusable for the
    downstream question-variant generation task.

    Returns a boolean Series aligned to `df.index`.
    """
    short = df[key_col].fillna("").str.len() <= max_len
    no_ga4 = pd.Series(True, index=df.index)
    for c in ga4_cols:
        if c in df.columns:
            no_ga4 &= df[c].fillna(0) == 0
    has_internal = df[internal_col].fillna(0) > 0 if internal_col in df.columns else False
    return short & no_ga4 & has_internal


# ── Combined convenience ─────────────────────────────────────────────────────


def build_default_filter_pipeline(
    employees_json: Path | str,
    extra_navigational: set[str] | None = None,
):
    """Return a dict of callable filters ready for the topic-modelling pipe.

    Usage:
        fp = build_default_filter_pipeline('data/employees_2026-02-06.json')
        df['is_title']    = df['query'].map(fp['title'])
        df['is_employee'] = df['query'].map(fp['employee'])
        df['is_nav']      = df['query'].map(fp['nav'])
        df['drop']        = df[['is_title','is_employee','is_nav']].any(axis=1)
    """
    return {
        "title":    is_title_query,
        "employee": EmployeeFilter(employees_json),
        "nav":      NavigationalFilter(extra=extra_navigational),
    }
