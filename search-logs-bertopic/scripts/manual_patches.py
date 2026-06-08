"""Manual cluster-relevance overrides after a P1-05 sweep.

Loads cluster_manifest.jsonl, applies a curated set of override entries
(documented reasons in code), writes the manifest back with the same schema.

Run after `cluster_cards.py` if you want to ship corrections without paying
for another full LLM sweep. Re-runnable; idempotent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "cache" / "clusters" / "cluster_manifest.jsonl"
SUMMARY = ROOT / "cache" / "clusters" / "cluster_manifest_summary.csv"

# (cluster_id, new clinical_relevance, short reason in EN for the audit trail)
PATCHES: list[tuple[int, str, str]] = [
    # The LLM under-recognized that oncological context is assumed for queries
    # coming from X. These overrides correct that.
    (736,  "oncology-core",      "Modrá kniha čos = Czech oncology society reference book; core oncology content."),
    (1306, "oncology-adjacent",  "World Cancer Day / cancer-news queries; patient interest in oncology calendar/news."),
    (1715, "oncology-adjacent",  "Kate Middleton cancer queries — patient interest in public-figure cancer cases."),
    (1008, "oncology-adjacent",  "dexter/sinister Latin anatomical terms — patients decoding pathology reports."),
    (1780, "oncology-adjacent",  "Proteomics & acute-phase proteins — biomedical patient/family research."),
    (1986, "oncology-adjacent",  "Medical abbreviation lookups (l.sin., lg sil) — patients decoding doctors' notes."),
    # Patient-services clusters labelled non-clinical that are actually navigational
    # (hospital patient programmes, fundraising events, info points).
    (1585, "navigational",       "'Tancem pro život' is a Czech cancer-patient dance/fundraising programme."),
    (835,  "navigational",       "Art-therapy workshops for patients — patient-services navigational."),
    (1245, "navigational",       "Hospital art gallery and exhibition pages — navigational."),
    (1356, "navigational",       "Cultural events / programmes for patients — navigational."),
    (1656, "navigational",       "Czech data-inbox / SUKL drug-database lookups — administrative navigational."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"missing: {args.manifest}")

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {int(r["topic_id"]): r for r in rows}

    n_patched = 0
    for cid, new_rel, reason in PATCHES:
        if cid not in by_id:
            print(f"  ! missing cluster {cid} — skipping")
            continue
        r = by_id[cid]
        old_rel = r["clinical_relevance"]
        if old_rel == new_rel:
            print(f"  · {cid:>4}  already {new_rel}  ({r['czech_label']})")
            continue
        print(f"  → {cid:>4}  {old_rel:<18} → {new_rel:<18}  ({r['czech_label']})")
        if not args.dry_run:
            r["clinical_relevance"] = new_rel
            r["rationale_cs"] = f"[manual_override] {reason} (was: {old_rel}). " + r["rationale_cs"]
            n_patched += 1

    if args.dry_run:
        print("[dry-run] no changes written")
        return 0

    # Stable order by topic_id
    out = sorted(by_id.values(), key=lambda r: int(r["topic_id"]))
    with open(args.manifest, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] patched {n_patched} clusters; wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
