import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "x:/projects/onkoradce/cikm-ds/search-logs-bertopic/cache/preprocess"
kept = pd.read_parquet(f"{ROOT}/embed_input.parquet")
drop = pd.read_parquet(f"{ROOT}/dropped_queries.parquet")
print(f"kept rows: {len(kept):,}    dropped rows: {len(drop):,}")
print(f"kept cols: {list(kept.columns)}")
print(f"NaN in query: {kept['query'].isna().sum()}; NaN in query_key: {kept['query_key'].isna().sum()}")
print()
print("--- top 20 kept by clicks ---")
for _, r in kept.nlargest(20, "clicks").iterrows():
    print(f"  {r['clicks']:>5}  {r['query']!r}")
print()
print("--- random 25 kept (seed=7) ---")
for _, r in kept.sample(25, random_state=7).iterrows():
    print(f"  {r['clicks']:>5}  {r['query']!r}")
print()
print("--- top 15 dropped by clicks (should be brand/employee/title) ---")
for _, r in drop.nlargest(15, "clicks").iterrows():
    reasons = [c for c in drop.columns if c.startswith("drop_") and r[c]]
    print(f"  {r['clicks']:>5}  {r['query']!r:<55}  {reasons}")
print()
print("--- random 25 dropped (seed=7) ---")
for _, r in drop.sample(25, random_state=7).iterrows():
    reasons = [c for c in drop.columns if c.startswith("drop_") and r[c]]
    print(f"  {r['clicks']:>5}  {r['query']!r:<55}  {reasons}")
