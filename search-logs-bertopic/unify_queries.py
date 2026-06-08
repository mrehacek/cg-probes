"""Unify search-query exports from /data into a single long-format CSV.

Sources handled:
  * site1 internal site-search log       — .xls  (columns in Czech)
  * GA4 organic-search exports (site1)   — .csv  (9-line header)
  * GA4 device-breakdown exports         — .csv  (6-line header, extra device column)

Output schema (one row per (source_file, query_key)):
  query_key            casefold + diacritics-stripped — dedup key within a source
  query                representative original casing/diacritics (highest-count variant)
  source               'site1_internal' | 'site1_ga4'
  site                 'site1'
  source_file          filename it came from
  clicks               internal search count OR GA4 clicks
  impressions          GA4 only
  ctr                  GA4 only
  position             GA4 only
  found                internal only (# results returned)
  lang                 internal only
  device               GA4 device-breakdown only
  date_start           YYYY-MM-DD (from GA4 header or internal last_searched year)
  date_end             YYYY-MM-DD
  last_searched        internal only — latest timestamp of any variant (YYYY-MM-DD)
  last_year            internal only — year of last_searched (Int64)

Aggregated schema (--aggregate) adds per-source pivot columns on top of the
legacy totals, so the topic-modelling pipeline can see which channel drove
each query:

  clicks_site1_ga4,    clicks_site1_internal
  impressions_site1_ga4    (XLS has no impressions)
  site1_int_last_searched, site1_int_last_year
  clicks, impressions, ctr, position, found, sources, sites, n_sources

Czech-aware dedup: variants that only differ in case or diacritics collapse
into a single row; clicks are summed, the most-frequent written form wins.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = DATA_DIR / "search_queries_unified_2025-2026.csv"
OUT_PATH_AGG = DATA_DIR / "search_queries_unified_2025-2026_aggregated.csv"

# ─── Hacking / junk filters (ported from pipeline.ipynb) ──────────────────────
BAD_PATTERN = re.compile(r"^\s*$|^\{.*\}$|^[^\w\s]{0,3}$")
MIN_LEN = 3


def normalize_key(s: str) -> str:
    """Dedup key: casefold + strip diacritics + collapse whitespace."""
    s = unicodedata.normalize("NFD", s.casefold())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def clean_query(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if len(s) < MIN_LEN:
        return None
    if BAD_PATTERN.match(s):
        return None
    return s


# ─── GA4 header parsing ───────────────────────────────────────────────────────
DATE_RE = re.compile(r"(20\d{6})")


def _ga4_dates(path: Path) -> tuple[str | None, str | None]:
    """Pull start/end YYYYMMDD from the CSV header comments."""
    dates: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#") and not line.strip() == "":
                break
            dates += DATE_RE.findall(line)
    def fmt(d: str) -> str:
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    if len(dates) >= 2:
        return fmt(dates[0]), fmt(dates[1])
    if len(dates) == 1:
        return fmt(dates[0]), fmt(dates[0])
    return None, None


def _ga4_header_rows(path: Path) -> int:
    """Count leading comment/blank rows before the CSV header line."""
    n = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                n += 1
            else:
                break
    return n


# ─── Dedup within a single source ─────────────────────────────────────────────
def _dedup(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Collapse rows that share a query_key. Sum numerics, keep best name."""
    if df.empty:
        return df
    # representative original = variant with the highest clicks in each group
    df = df.sort_values("clicks", ascending=False)
    rep = df.drop_duplicates(subset="query_key", keep="first")[["query_key", "query"]]
    agg = {c: "sum" for c in numeric_cols if c in df.columns}
    # for ctr/position take click-weighted mean rather than sum
    for c in ("ctr", "position"):
        if c in df.columns:
            agg.pop(c, None)
    grouped = df.groupby("query_key", as_index=False).agg(agg)
    # click-weighted averages for rate metrics
    for c in ("ctr", "position"):
        if c in df.columns:
            w = df.groupby("query_key").apply(
                lambda g: (g[c] * g["clicks"]).sum() / g["clicks"].sum()
                if g["clicks"].sum() else g[c].mean()
            )
            grouped[c] = grouped["query_key"].map(w)
    out = rep.merge(grouped, on="query_key")
    # carry over first-value scalar columns (source, site, dates, lang, etc.)
    for c in df.columns:
        if c in out.columns or c in ("query_key", "query"):
            continue
        first = df.drop_duplicates("query_key", keep="first").set_index("query_key")[c]
        out[c] = out["query_key"].map(first)
    return out


# ─── Loaders ──────────────────────────────────────────────────────────────────
_XLS_DATE_RE = re.compile(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})")


def _parse_xls_date(s: str) -> pd.Timestamp | None:
    """Parse 'D. M. YYYY HH:MM' → Timestamp. Returns NaT on failure."""
    if not isinstance(s, str):
        return pd.NaT
    m = _XLS_DATE_RE.search(s)
    if not m:
        return pd.NaT
    d, mo, y = (int(x) for x in m.groups())
    try:
        return pd.Timestamp(year=y, month=mo, day=d)
    except (ValueError, OverflowError):
        return pd.NaT


def load_internal_xls(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, engine="calamine")
    raw = raw.rename(columns={
        "Hledaný výraz":     "query",
        "Jazyk":             "lang",
        "Hledáno":           "clicks",
        "Nalezeno":          "found",
        "Naposledy hledáno": "last_searched_raw",
    })
    raw = raw.dropna(subset=["query"])
    raw["query"] = raw["query"].astype(str).map(clean_query)
    raw = raw.dropna(subset=["query"])
    raw["query_key"] = raw["query"].map(normalize_key)
    raw["clicks"] = pd.to_numeric(raw["clicks"], errors="coerce").fillna(0).astype(int)
    raw["found"] = pd.to_numeric(raw.get("found"), errors="coerce").astype("Int64")
    raw["source"] = "site1_internal"
    raw["site"] = "site1"
    raw["source_file"] = path.name

    # Parse last_searched. One timestamp per raw row (i.e. per case/diacritic
    # variant). After dedup we take the MAX across variants so `last_searched`
    # reflects the most recent activity of any variant.
    ts = raw["last_searched_raw"].map(_parse_xls_date)
    raw["last_searched"] = ts.dt.strftime("%Y-%m-%d")
    raw["last_year"] = ts.dt.year.astype("Int64")
    years_all = ts.dt.year.dropna()
    raw["date_start"] = f"{int(years_all.min())}-01-01" if len(years_all) else None
    raw["date_end"] = f"{int(years_all.max())}-12-31" if len(years_all) else None

    cols = ["query_key", "query", "source", "site", "source_file",
            "clicks", "found", "lang", "last_searched", "last_year",
            "date_start", "date_end"]
    # _dedup's "carry-over first value" logic uses sort-by-clicks order, which
    # is wrong for last_searched (we want max, not highest-clicks variant's
    # value). Compute it separately then merge back.
    per_key_last = (
        raw.sort_values("last_searched")
           .groupby("query_key", as_index=False)
           .agg(last_searched=("last_searched", "max"),
                last_year=("last_year", "max"))
    )
    deduped = _dedup(raw[cols], numeric_cols=["clicks", "found"])
    deduped = deduped.drop(columns=["last_searched", "last_year"], errors="ignore")
    return deduped.merge(per_key_last, on="query_key", how="left")


def load_ga4(path: Path, site: str) -> pd.DataFrame:
    skip = _ga4_header_rows(path)
    df = pd.read_csv(path, skiprows=skip, encoding="utf-8", on_bad_lines="skip")
    # Two layouts: plain (5 cols) and device-breakdown (6+ cols with device column)
    device_col = next((c for c in df.columns if "zařízen" in c or "zariaden" in c), None)
    df.columns = [c.strip() for c in df.columns]
    query_col = df.columns[0]
    df = df.rename(columns={query_col: "query"})
    # Column name varies by locale (cs/sk) — take positional.
    if device_col:
        df = df.rename(columns={device_col: "device"})
        metric_cols = [c for c in df.columns if c not in ("query", "device")]
    else:
        df["device"] = None
        metric_cols = [c for c in df.columns if c not in ("query", "device")]
    # Positional rename of the four GA4 metrics
    rename_pos = dict(zip(metric_cols[:4], ["clicks", "impressions", "ctr", "position"]))
    df = df.rename(columns=rename_pos)

    df["query"] = df["query"].astype(str).map(clean_query)
    df = df.dropna(subset=["query"])
    df["query_key"] = df["query"].map(normalize_key)
    for c in ("clicks", "impressions"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ("ctr", "position"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    date_start, date_end = _ga4_dates(path)
    df["source"] = f"{site}_ga4"
    df["site"] = site
    df["source_file"] = path.name
    df["date_start"] = date_start
    df["date_end"] = date_end

    cols = ["query_key", "query", "source", "site", "source_file", "device",
            "clicks", "impressions", "ctr", "position", "date_start", "date_end"]
    return _dedup(df[cols], numeric_cols=["clicks", "impressions"])


# ─── Orchestration ────────────────────────────────────────────────────────────
def discover(data_dir: Path) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for p in sorted(data_dir.iterdir()):
        name = p.name.lower()
        if name.endswith(".xls") and "search_log" in name:
            jobs.append(("internal_xls", p))
        elif name.endswith(".csv") and "ga4" in name:
            jobs.append(("ga4:site1", p))
    return jobs


def _pick_canonical_files(unified: pd.DataFrame) -> pd.DataFrame:
    """Drop overlapping GA4 exports: within each (site, source) keep the file
    with the widest date span (tiebreak: most total clicks). Needed before any
    cross-source sum — otherwise overlapping periods would double-count.
    Returns the filtered DataFrame and prints which files were dropped.
    """
    keep_files: set[str] = set()
    for (site, source), g in unified.groupby(["site", "source"]):
        files = g["source_file"].unique()
        if len(files) == 1:
            keep_files.add(files[0])
            continue
        stats = []
        for f in files:
            sub = g[g["source_file"] == f]
            ds = pd.to_datetime(sub["date_start"].iloc[0], errors="coerce")
            de = pd.to_datetime(sub["date_end"].iloc[0], errors="coerce")
            span = (de - ds).days if pd.notna(ds) and pd.notna(de) else 0
            stats.append((f, span, int(sub["clicks"].sum())))
        stats.sort(key=lambda s: (s[1], s[2]), reverse=True)
        winner = stats[0][0]
        keep_files.add(winner)
        for f, span, clk in stats[1:]:
            print(f"  dropping overlapping file: {f}  (span={span}d, clicks={clk:,}) — "
                  f"kept {winner}", file=sys.stderr)
    return unified[unified["source_file"].isin(keep_files)].copy()


_ALL_SOURCES = ("site1_ga4", "site1_internal")


def aggregate_cross_source(unified: pd.DataFrame) -> pd.DataFrame:
    """Collapse by query_key across all sources. Sums clicks/impressions,
    click-weighted ctr/position, records which sources contributed, and pivots
    per-source clicks/impressions into separate columns so the topic-modelling
    pipeline can see the breakdown (patient-facing Google organic vs
    staff-facing internal site search, etc.).
    """
    df = _pick_canonical_files(unified)
    # Representative original spelling: highest-clicks variant overall.
    df = df.sort_values("clicks", ascending=False)
    rep = df.drop_duplicates("query_key", keep="first")[["query_key", "query"]]

    num_cols = [c for c in ("clicks", "impressions", "found") if c in df.columns]
    grouped = df.groupby("query_key", as_index=False)[num_cols].sum(min_count=1)

    # Click-weighted rate metrics
    for c in ("ctr", "position"):
        if c in df.columns:
            w = df.groupby("query_key").apply(
                lambda g: (g[c] * g["clicks"]).sum() / g["clicks"].sum()
                if g["clicks"].sum() else g[c].mean()
            )
            grouped[c] = grouped["query_key"].map(w)

    # Which sources / sites contributed
    srcs = df.groupby("query_key")["source"].agg(lambda s: ",".join(sorted(set(s))))
    sites = df.groupby("query_key")["site"].agg(lambda s: ",".join(sorted(set(s))))
    n_src = df.groupby("query_key")["source"].nunique()

    out = rep.merge(grouped, on="query_key")
    out["sources"] = out["query_key"].map(srcs)
    out["sites"] = out["query_key"].map(sites)
    out["n_sources"] = out["query_key"].map(n_src)

    # ── Per-source pivot (clicks + impressions) ───────────────────────────────
    for src in _ALL_SOURCES:
        sub = df[df["source"] == src]
        clicks_by_key = sub.groupby("query_key")["clicks"].sum()
        out[f"clicks_{src}"] = out["query_key"].map(clicks_by_key).fillna(0).astype("Int64")
        if "impressions" in sub.columns and sub["impressions"].notna().any():
            imps_by_key = sub.groupby("query_key")["impressions"].sum()
            out[f"impressions_{src}"] = out["query_key"].map(imps_by_key).fillna(0).astype("Int64")

    # ── internal-only freshness carry-over ────────────────────────────────────
    if "last_searched" in df.columns:
        internal = df[df["source"] == "site1_internal"]
        if len(internal):
            last = internal.groupby("query_key")["last_searched"].max()
            out["site1_int_last_searched"] = out["query_key"].map(last)
            if "last_year" in internal.columns:
                last_year = internal.groupby("query_key")["last_year"].max()
                out["site1_int_last_year"] = out["query_key"].map(last_year).astype("Int64")

    return out.sort_values("clicks", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aggregate", action="store_true",
                    help="Also write a cross-source aggregated CSV "
                         "(one row per query_key, totals summed across sources). "
                         "Drops overlapping GA4 exports to avoid double-counting.")
    ap.add_argument("--only-aggregated", action="store_true",
                    help="Only write the aggregated output, not the long-format one.")
    ap.add_argument("-o", "--output", type=Path, default=OUT_PATH,
                    help=f"Long-format output path (default: {OUT_PATH.name}).")
    ap.add_argument("--output-aggregated", type=Path, default=OUT_PATH_AGG,
                    help=f"Aggregated output path (default: {OUT_PATH_AGG.name}).")
    args = ap.parse_args()

    jobs = discover(DATA_DIR)
    if not jobs:
        print(f"No input files found in {DATA_DIR}", file=sys.stderr)
        return 1
    frames: list[pd.DataFrame] = []
    for kind, path in jobs:
        print(f"[{kind}] {path.name} ...", file=sys.stderr)
        try:
            if kind == "internal_xls":
                frames.append(load_internal_xls(path))
            else:
                frames.append(load_ga4(path, site=kind.split(":", 1)[1]))
        except Exception as e:
            print(f"  skipped: {e}", file=sys.stderr)
    unified = pd.concat(frames, ignore_index=True)
    unified = unified.sort_values(
        ["site", "source", "clicks"], ascending=[True, True, False]
    ).reset_index(drop=True)

    if not args.only_aggregated:
        unified.to_csv(args.output, index=False, encoding="utf-8")
        print(f"\nWrote {len(unified):,} rows → {args.output}", file=sys.stderr)
        print(unified.groupby(["site", "source"]).size().to_string(), file=sys.stderr)

    if args.aggregate or args.only_aggregated:
        print("\nAggregating across sources...", file=sys.stderr)
        agg = aggregate_cross_source(unified)
        agg.to_csv(args.output_aggregated, index=False, encoding="utf-8")
        print(f"Wrote {len(agg):,} unique queries → {args.output_aggregated}",
              file=sys.stderr)
        print(f"  total clicks: {int(agg['clicks'].sum()):,}", file=sys.stderr)
        print(f"  queries in >1 source: {(agg['n_sources'] > 1).sum():,}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
