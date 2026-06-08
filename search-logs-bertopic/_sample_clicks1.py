"""One-off: sample 500 random clicks==1 queries for human review.

Helps decide whether the clicks==1 long tail (~50,590 rows) is junk to drop
or contains rare-but-valuable patient queries (emergencies, edge symptoms).
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
agg = pd.read_csv(ROOT / "data/search_queries_unified_2025-2026_aggregated.csv",
                  low_memory=False)
ones = agg[agg["clicks"] == 1].copy()
print(f"Total clicks==1 rows: {len(ones):,}")
print()
for c in ["clicks_site1_ga4", "clicks_site1_internal"]:
    if c in ones.columns:
        print(f"  >0 in {c:>25}: {(ones[c].fillna(0) > 0).sum():>6,}")
print()

sample = ones.sample(500, random_state=42).reset_index(drop=True)


def src_short(row):
    parts = []
    if row.get("clicks_site1_ga4", 0) > 0:
        parts.append("G")
    if row.get("clicks_site1_internal", 0) > 0:
        parts.append("I")
    return "".join(parts) or "?"


sample["src"] = sample.apply(src_short, axis=1)

# CSV for spreadsheet review
out_csv = ROOT / "cache/clicks1_sample_500.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
sample[["src", "query"]].to_csv(out_csv, index=False, encoding="utf-8")

# TXT for fast eyeballing (sorted by source so patterns are visible)
out_txt = ROOT / "cache/clicks1_sample_500.txt"
lines = ["src\tquery"]
for _, r in sample.sort_values(["src", "query"]).iterrows():
    lines.append(f"{r['src']}\t{r['query']}")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print(f"Wrote {out_csv.relative_to(ROOT.parent)}")
print(f"Wrote {out_txt.relative_to(ROOT.parent)}")
print()
print("src legend: G=site1-GA4 (organic), I=site1-internal (site search)")
print("Multi-letter = multi-source. Sole-source clicks=1 from organic search means a real")
print("Google query that fired once and got 1 click; clicks=1 on internal-only means one")
print("hospital visitor typed it into the search box once.")
print()
print(sample["src"].value_counts().to_string())
