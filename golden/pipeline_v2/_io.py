"""Slim I/O helpers for the MU/PU/ET golden-set pipeline.

Holds: repo-relative input/output paths (this pipeline reads from
search-logs-bertopic/, NOT the sibling embeddings/ repo that contrastive.paths
targets), the supercluster loader, prompt parsing, hashing for provenance, and
atomic write + state-file helpers shared by every step.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# golden/pipeline_v2/_io.py -> parents[2] == repo root (cikm-ds/)
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

# --- Inputs (read-only) ------------------------------------------------------

_CLASSIFICATION = REPO / "search-logs-bertopic" / "cache" / "classification"
SUPERCLUSTERS_JSONL = _CLASSIFICATION / "superclusters_l2_constrained.jsonl"
QUERIES_CSV = _CLASSIFICATION / "queries_classified_l2.csv"
AXES_SNAPSHOT = REPO / "axes-snapshot-20250604.md"

# --- Prompts -----------------------------------------------------------------

PROMPTS_DIR = REPO / "golden" / "prompts"
RUBRICS_DIR = PROMPTS_DIR / "rubrics_v2"
PICKER_PROMPT = PROMPTS_DIR / "cluster_picker_v1.md"
VERIFIER_PROMPT = PROMPTS_DIR / "cluster_verifier_v1.md"
SYNTH_PROMPT = PROMPTS_DIR / "backfill_synthesizer_v1.md"

# --- Outputs (gitignored under golden/cache/) --------------------------------

OUT_DIR = REPO / "golden" / "cache" / "golden_set_v1"
STATE_DIR = OUT_DIR / "state"
REPORTS_DIR = OUT_DIR / "reports"

PICKS_RAW = OUT_DIR / "picks_raw.parquet"
GOLDEN_400_PICKED = OUT_DIR / "golden_400_picked.parquet"
VERIFIER_400 = OUT_DIR / "verifier_400.parquet"
AGREEMENT_400 = OUT_DIR / "agreement_400.parquet"
GOLDEN_400_FILLED = OUT_DIR / "golden_400_filled.parquet"


def ensure_dirs() -> None:
    for d in (OUT_DIR, STATE_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- Time --------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Cluster loading ---------------------------------------------------------

def load_superclusters() -> list[dict]:
    """Read the supercluster JSONL into a list of dicts (one per line).

    Each record has: supercluster_id, clinical_relevance, czech_label,
    summary_cs, n_queries, top_words, urgency_potential, emotional_potential,
    and queries[] (each {query_key, query, clicks, cluster_id}).
    """
    if not SUPERCLUSTERS_JSONL.exists():
        raise FileNotFoundError(f"superclusters not found: {SUPERCLUSTERS_JSONL}")
    out: list[dict] = []
    with SUPERCLUSTERS_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def cluster_member_queries(
    cluster: dict, *, dedup: bool = True, limit: int | None = None
) -> list[str]:
    """Verbatim member-query strings for one cluster, most-clicked first.

    Dedups on query_key (keeping the first/highest-click surface form) so the
    picker isn't shown the same query twice. `limit` caps the count (None = all);
    the picker model has a 1M-token window so the default is to include all.
    """
    items = sorted(
        cluster.get("queries", []),
        key=lambda q: q.get("clicks", 0),
        reverse=True,
    )
    seen: set[str] = set()
    texts: list[str] = []
    for q in items:
        key = q.get("query_key") or q.get("query", "")
        if dedup and key in seen:
            continue
        seen.add(key)
        texts.append(q.get("query", ""))
        if limit is not None and len(texts) >= limit:
            break
    return texts


# --- Prompt parsing ----------------------------------------------------------

def split_prompt(text: str) -> tuple[str, str]:
    """Split a `[SYSTEM] ... [USER] ...` prompt file into (system, user).

    Leading `#` comment lines before the first marker are ignored. Mirrors the
    convention in golden/llm_annotator/prompts/axis_judge.txt.
    """
    if "[SYSTEM]" not in text or "[USER]" not in text:
        raise ValueError("prompt must contain both [SYSTEM] and [USER] markers")
    after_system = text.split("[SYSTEM]", 1)[1]
    system, user = after_system.split("[USER]", 1)
    return system.strip(), user.strip()


def load_prompt(path: Path) -> tuple[str, str]:
    return split_prompt(path.read_text(encoding="utf-8"))


# --- Hashing (provenance + prompt-gating) ------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Atomic writes -----------------------------------------------------------

def write_json_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_parquet_atomic(df, path: Path) -> None:
    """Atomic parquet write (df.to_parquet to tmp, then os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


# --- State files -------------------------------------------------------------

def read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(state: dict, path: Path) -> None:
    write_json_atomic(state, path)


def print_status(path: Path) -> None:
    """Human-readable dump of a step state file (for `--status`)."""
    state = read_state(path)
    if state is None:
        print(f"[status] no state file at {path}")
        return
    print(f"[status] {path}")
    for k, v in state.items():
        if isinstance(v, list) and len(v) > 8:
            print(f"  {k}: <{len(v)} items> {v[:8]} ...")
        else:
            print(f"  {k}: {v}")
