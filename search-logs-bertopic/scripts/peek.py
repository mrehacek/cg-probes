import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_csv(r"x:/projects/onkoradce/cikm-ds/search-logs-bertopic/scripts/out/spacy_person_filtered_sample.csv")
print("rows:", len(df))
print("--- first 40 ---")
for _, r in df.head(40).iterrows():
    print(f"  {r['query']!r:<55}  ->  {r['all_entities']}")
print("--- random 40 (seed=1) ---")
for _, r in df.sample(40, random_state=1).iterrows():
    print(f"  {r['query']!r:<55}  ->  {r['all_entities']}")
