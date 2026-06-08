"""STEP 5 — Partition + annotation-pack build (local, no LLM).

Splits the 400 into two annotation-app packages matching the user's design:

  * core-200  — the IAA overlap; assigned to BOTH oncologists (overlapCount=200).
  * extra-200 — assigned to oncologist B only (overlapCount=0).

So A annotates 200, B annotates 400, IAA is computed on the shared 200.

Partition rule (user decision 2026-06-05): force ALL rare-safety queries
(MU2 ∪ PU2, real + synthetic) into core so both oncologists grade every
emergency/suicidal item; fill the rest of core with a *random* draw so the
IAA-hard (picker↔verifier disagreement) queries land proportionally across both
packages rather than concentrated in the IAA set (which would depress agreement).

Emits app-import-format package JSON (POST /admin/packages/import) with gold
labels left BLANK (don't bias the oncologists; the importer only takes one axis
anyway). Our MU/PU/ET labels are kept in a local manifest keyed by `source` for
post-annotation IAA + benchmark.

Run:
    python golden/pipeline_v2/05_packs.py
    python golden/pipeline_v2/05_packs.py --status

Outputs:
    golden/cache/golden_set_v1/packages/core_200.json
    golden/cache/golden_set_v1/packages/extra_200.json
    golden/cache/golden_set_v1/packs_manifest.parquet   (source ↔ our labels)
    golden/cache/golden_set_v1/annotator_A.csv / annotator_B.csv  (+ _debug)
    golden/cache/golden_set_v1/state/05_packs_state.json
    golden/cache/golden_set_v1/reports/05_packs_report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from golden.pipeline_v2 import _io, _report  # noqa: E402
from golden.pipeline_v2.schemas import AXES  # noqa: E402

STATE_PATH = _io.STATE_DIR / "05_packs_state.json"
REPORT_PATH = _io.REPORTS_DIR / "05_packs_report.html"
PACKAGES_DIR = _io.OUT_DIR / "packages"
MANIFEST = _io.OUT_DIR / "packs_manifest.parquet"
ANN_A = _io.OUT_DIR / "annotator_A.csv"
ANN_B = _io.OUT_DIR / "annotator_B.csv"

SEED = 42
N_CORE = 200
SOURCE_PREFIX = "golden-v1:"

# Package descriptors (annotator IDs injected at upload time by 07_upload.py).
CORE_META = {
    "title": "Golden v1 — Core (IAA, MU/PU/ET)",
    "subtitle": "Onkolog A + B — sdílený balík pro shodu hodnotitelů",
    "description": ("200 dotazů hodnocených OBĚMA onkology (osy MU/PU/ET) pro "
                    "výpočet shody (IAA). Obsahuje všechny bezpečnostně kritické "
                    "dotazy (MU2 nouzové, PU2 suicidální) včetně syntetických."),
    "overlapCount": N_CORE,
}
EXTRA_META = {
    "title": "Golden v1 — Extra (onkolog B)",
    "subtitle": "Pouze onkolog B",
    "description": "200 dalších dotazů hodnocených pouze onkologem B (osy MU/PU/ET).",
    "overlapCount": 0,
}


def source_key(query_key: str) -> str:
    s = SOURCE_PREFIX + str(query_key)
    if len(s) > 200:  # app cap; fall back to a stable hash
        s = SOURCE_PREFIX + _io.sha256_text(str(query_key))[:32]
    return s


def partition(filled: pd.DataFrame):
    """core-200 = all rare-safety + random fill; extra-200 = the rest."""
    rare = filled[(filled["mu_grade"] == "MU2") | (filled["pu_grade"] == "PU2")]
    rest = filled[~filled.index.isin(rare.index)]
    if len(rare) >= N_CORE:  # unlikely; truncate rare into core, overflow to extra
        rare_sh = rare.sample(frac=1, random_state=SEED)
        core = rare_sh.head(N_CORE)
        extra = pd.concat([rare_sh.tail(len(rare) - N_CORE), rest])
    else:
        rest_sh = rest.sample(frac=1, random_state=SEED)
        n_fill = N_CORE - len(rare)
        core = pd.concat([rare, rest_sh.head(n_fill)])
        extra = rest_sh.tail(len(rest) - n_fill)
    return core.reset_index(drop=True), extra.reset_index(drop=True)


def build_package(df: pd.DataFrame, meta: dict) -> dict:
    return {
        "title": meta["title"], "subtitle": meta["subtitle"],
        "description": meta["description"],
        "assignAnnotatorIds": [],          # injected at upload time
        "overlapCount": meta["overlapCount"],
        "queries": [{"text": r["query_text"], "source": source_key(r["query_key"])}
                    for _, r in df.iterrows()],
    }


def run(args) -> int:
    _io.ensure_dirs()
    PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not _io.GOLDEN_400_FILLED.exists():
        print("[packs] golden_400_filled.parquet missing — run step 4 first.", file=sys.stderr)
        return 2
    filled = pd.read_parquet(_io.GOLDEN_400_FILLED).reset_index(drop=True)
    agree = pd.read_parquet(_io.AGREEMENT_400) if _io.AGREEMENT_400.exists() else None

    core, extra = partition(filled)
    core_json = build_package(core, CORE_META)
    extra_json = build_package(extra, EXTRA_META)
    _io.write_json_atomic(core_json, PACKAGES_DIR / "core_200.json")
    _io.write_json_atomic(extra_json, PACKAGES_DIR / "extra_200.json")

    # manifest: source ↔ our labels (for post-annotation IAA + benchmark)
    def manifest_rows(df, package):
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "source": source_key(r["query_key"]), "package": package,
                "query_key": r["query_key"], "query_text": r["query_text"],
                "mu_grade": r["mu_grade"], "pu_grade": r["pu_grade"], "et_grade": r["et_grade"],
                "origin": r["source"], "is_synthetic": r["source"] == "synthesized",
                "supercluster_id": r.get("supercluster_id"),
                "clinical_relevance": r.get("clinical_relevance"),
            })
        return rows
    man = pd.DataFrame(manifest_rows(core, "core") + manifest_rows(extra, "extra"))
    if agree is not None:
        am = agree[["query_key", "n_axes_agreement", "is_iaa_hard_candidate"]]
        man = man.merge(am, on="query_key", how="left")
    _io.write_parquet_atomic(man, MANIFEST)

    # annotator-facing + debug CSVs (the app handles assignment; these are for review)
    a_face = core[["query_text"]].copy()
    a_face.insert(0, "source", [source_key(k) for k in core["query_key"]])
    for ax in AXES:
        a_face[f"{ax.lower()}_grade_human"] = ""
    a_face["notes_human"] = ""
    a_face.to_csv(ANN_A, index=False, encoding="utf-8")
    b_face = pd.concat([core, extra])[["query_text"]].copy()
    b_face.insert(0, "source", [source_key(k) for k in pd.concat([core, extra])["query_key"]])
    for ax in AXES:
        b_face[f"{ax.lower()}_grade_human"] = ""
    b_face["notes_human"] = ""
    b_face.to_csv(ANN_B, index=False, encoding="utf-8")
    man[man["package"] == "core"].to_csv(ANN_A.with_name("annotator_A_debug.csv"), index=False, encoding="utf-8")
    man.to_csv(ANN_B.with_name("annotator_B_debug.csv"), index=False, encoding="utf-8")

    state = {
        "completed_at": _io.now_iso(), "current_status": "completed",
        "n_core": len(core), "n_extra": len(extra), "overlap": N_CORE,
        "annotator_A_total": len(core), "annotator_B_total": len(core) + len(extra),
        "core_synthetic": int((core["source"] == "synthesized").sum()),
        "extra_synthetic": int((extra["source"] == "synthesized").sum()),
        "seed": SEED,
    }
    _io.write_state(state, STATE_PATH)
    REPORT_PATH.write_text(render_report(core, extra, man, state), encoding="utf-8")

    print(f"[packs] DONE. core={len(core)} extra={len(extra)} | A={len(core)} B={len(core)+len(extra)} "
          f"| core synthetic={state['core_synthetic']}", file=sys.stderr)
    print(f"[packs] packages: {PACKAGES_DIR}\\{{core_200,extra_200}}.json", file=sys.stderr)
    print(f"[packs] report: {REPORT_PATH}", file=sys.stderr)
    return 0


def render_report(core, extra, man, state) -> str:
    R = _report
    head = R.kv({
        "Core (shared, IAA)": f"{len(core)} → both oncologists (overlapCount={N_CORE})",
        "Extra (B only)": f"{len(extra)} → oncologist B",
        "Annotator A grades": state["annotator_A_total"],
        "Annotator B grades": state["annotator_B_total"],
        "IAA overlap": N_CORE,
        "Synthetic in core / extra": f"{state['core_synthetic']} / {state['extra_synthetic']}",
    })

    # per-axis distribution per package
    rows = []
    for ax in AXES:
        col = ax.lower() + "_grade"
        for g in (f"{ax}0", f"{ax}1", f"{ax}2"):
            rows.append({"cell": g,
                         "core": int((core[col] == g).sum()),
                         "extra": int((extra[col] == g).sum())})
    dist_tbl = R.table(rows, ["cell", "core", "extra"])

    # cluster diversity + synthetic ratio
    div = R.kv({
        "Distinct source clusters — core": core["supercluster_id"].nunique(),
        "Distinct source clusters — extra": extra["supercluster_id"].nunique(),
        "Synthetic ratio — core": f"{100*state['core_synthetic']/max(len(core),1):.0f}%",
        "Synthetic ratio — extra": f"{100*state['extra_synthetic']/max(len(extra),1):.0f}%",
    })

    # IAA-hard spread check (did hard land in both packages?)
    hard_spread = ""
    if "is_iaa_hard_candidate" in man.columns:
        h = man[man["is_iaa_hard_candidate"] == True]  # noqa: E712
        hc = h[h["package"] == "core"].shape[0]
        he = h[h["package"] == "extra"].shape[0]
        hard_spread = R.kv({"IAA-hard in core": hc, "IAA-hard in extra": he,
                            "(spread ~proportionally, not concentrated)": ""})

    # sample rows per package
    def sample(df, pkg):
        s = df.sample(min(10, len(df)), random_state=SEED)
        return [{"package": pkg, "query_text": r["query_text"],
                 "MU/PU/ET": f"{r['mu_grade']}/{r['pu_grade']}/{r['et_grade']}",
                 "synthetic": r["source"] == "synthesized"} for _, r in s.iterrows()]
    samp = R.table(sample(core, "core") + sample(extra, "extra"),
                   ["package", "query_text", "MU/PU/ET", "synthetic"])

    prov = R.provenance({
        "input": _io.GOLDEN_400_FILLED.name, "seed": SEED,
        "source_prefix": SOURCE_PREFIX, "import_endpoint": "POST /admin/packages/import",
        "completed_at": state["completed_at"],
    })

    sections = [
        R.section("1. Bucket sizes", head),
        R.section("2. Per-axis distribution per package", dist_tbl),
        R.section("3. Cluster diversity + synthetic ratio", div),
    ]
    if hard_spread:
        sections.append(R.section("4. IAA-hard spread (should be in BOTH packages)", hard_spread))
    sections += [
        R.section("5. Sample rows per package", samp),
        prov,
    ]
    return R.page("Step 5 — Annotation packs report", *sections)


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 5 — partition + pack build (local)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        _io.print_status(STATE_PATH)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
