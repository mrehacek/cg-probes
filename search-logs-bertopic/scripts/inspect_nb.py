import json, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8")
nb = json.loads(pathlib.Path(
    r"x:/projects/onkoradce/cikm-ds/search-logs-bertopic/bertopic_query_clusters.ipynb"
).read_text(encoding="utf-8"))
print(f"cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    head = src[:300].replace("\n", " / ")
    print(f"{i:2d} [{c['cell_type'][:4]}] {head}")
