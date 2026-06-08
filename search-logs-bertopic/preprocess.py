"""Apply filters.py + click threshold to the aggregated CSV.

Outputs embed_input.parquet (the rows we'll embed) and a filter_report.json
counting drops per filter so we can audit and tune the click threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from filters import (  # noqa: E402
    EmployeeFilter,
    NavigationalFilter,
    is_digit_query,
    is_spam_query,
    is_title_query,
    mark_internal_shorthand,
)

DATA = ROOT / "data"
OUT = ROOT / "cache" / "preprocess"

INPUT_CSV = DATA / "search_queries_unified_2025-2026_aggregated.csv"
EMPLOYEES_JSON = DATA / "employees_2026-02-06.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--min-clicks",
        type=int,
        default=1,
        help="Drop queries with clicks < this (default: 1, keep everything)",
    )
    ap.add_argument("--limit", type=int, default=None, help="For iteration")
    args = ap.parse_args()

    if not INPUT_CSV.exists():
        sys.exit(f"input not found: {INPUT_CSV}")
    if not EMPLOYEES_JSON.exists():
        sys.exit(f"employees json not found: {EMPLOYEES_JSON}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    if args.limit:
        df = df.head(args.limit)
    n0 = len(df)
    print(f"[preprocess] loaded {n0:,} rows from {INPUT_CSV.name}")

    # Order matters for auditability: cheapest first.
    df["drop_clicks"] = df["clicks"].fillna(0) < args.min_clicks
    df["drop_digits"] = df["query"].astype(str).map(is_digit_query)
    df["drop_spam"] = df["query"].astype(str).map(is_spam_query)
    df["drop_title"] = df["query"].astype(str).map(is_title_query)

    nav = NavigationalFilter()
    df["drop_nav"] = df["query"].astype(str).map(nav)

    emp = EmployeeFilter(EMPLOYEES_JSON)
    df["drop_employee"] = df["query"].astype(str).map(emp)

    df["drop_shorthand"] = mark_internal_shorthand(df)

    drop_cols = [c for c in df.columns if c.startswith("drop_")]
    df["drop_any"] = df[drop_cols].any(axis=1)

    report = {
        "input_rows": n0,
        "min_clicks": args.min_clicks,
        "per_filter_drops": {c: int(df[c].sum()) for c in drop_cols},
        "total_dropped": int(df["drop_any"].sum()),
        "total_kept": int((~df["drop_any"]).sum()),
        "kept_total_clicks": int(df.loc[~df["drop_any"], "clicks"].fillna(0).sum()),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "filter_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    keep = ~df["drop_any"]
    keep_cols = ["query_key", "query", "clicks", "n_sources"] + [
        c for c in df.columns if c.startswith("clicks_")
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df.loc[keep, keep_cols].to_parquet(OUT / "embed_input.parquet", index=False)
    df.loc[~keep, ["query_key", "query", "clicks", *drop_cols]].to_parquet(
        OUT / "dropped_queries.parquet", index=False
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
