"""P4 — embed the labeled set (and optionally the real corpus) per cell.

A "cell" is (embedder, instruction-mode[, axis]). For instruction-following
embedders the axis instruction changes the vector, so `per_axis` mode produces
one parquet per axis; `generic` produces one shared parquet. `openai3large` is
raw-only (generic). Train and eval texts are embedded together per cell so the
probe and the benchmark share identical vectors.

Texts embedded:
  * per-axis cell A: probe_dataset_{A}.text  ∪  golden-400 query_text
  * generic cell:    ∪ all probe_dataset.text ∪ golden-400  (+ real corpus if --with-corpus)

Output: benchmark/cache/emb/{embedder}__{mode}[__{AXIS}].parquet
        columns: text_id (sha1), text, embedding (list[float])  + .meta.json
Resumable (skips text_ids already present) and flushed every --checkpoint
batches, with a live progress line (per the incremental-progress rule).

Run (works now for gemini/openai3large; qwen8b/harrier27b need live endpoints):
  python -m benchmark.run_embed --embedder gemini --instr-mode per_axis
  python -m benchmark.run_embed --embedder openai3large --instr-mode generic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from benchmark import embedders as E
from contrastive import p2_io
from contrastive.p2_io import REPO, SAFETY_AXES

EMB_DIR = REPO / "benchmark" / "cache" / "emb"
GOLDEN_400 = REPO / "golden" / "cache" / "golden_set_v1" / "golden_400_filled.parquet"
SUPERCLUSTERS = (REPO / "search-logs-bertopic" / "cache" / "classification"
                 / "superclusters_l2_constrained.jsonl")


def _tid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _golden_texts() -> list[str]:
    g = pd.read_parquet(GOLDEN_400)
    return g["query_text"].astype(str).tolist()


def _axis_texts(axis: str) -> list[str]:
    d = pd.read_parquet(p2_io.probe_dataset_path(axis))
    return d["text"].astype(str).tolist() + _golden_texts()


def _all_labeled_texts() -> list[str]:
    texts: list[str] = []
    for ax in SAFETY_AXES:
        texts += pd.read_parquet(p2_io.probe_dataset_path(ax))["text"].astype(str).tolist()
    return texts + _golden_texts()


def _corpus_texts() -> list[str]:
    """~55k real core+adjacent member queries (deduplicated) for the GPU-hour batch."""
    out: list[str] = []
    with SUPERCLUSTERS.open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            if c.get("clinical_relevance") in ("oncology-core", "oncology-adjacent"):
                for q in c.get("queries", []):
                    out.append(q.get("query", ""))
    return out


def _dedup(texts: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"text": [t for t in texts if isinstance(t, str) and t.strip()]})
    df["text_id"] = df["text"].map(_tid)
    return df.drop_duplicates("text_id").reset_index(drop=True)


def _cell_path(embedder: str, mode: str, axis: str | None):
    tag = f"{embedder}__{mode}" + (f"__{axis}" if (mode in E.PER_AXIS_MODES and axis) else "")
    return EMB_DIR / f"{tag}.parquet"


def _flush(path, rows: list[dict]) -> int:
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    new.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(rows)


def embed_cell(embedder_name: str, mode: str, axis: str | None,
               batch_size: int, concurrency: int, limit: int | None) -> int:
    emb = E.make_embedder(embedder_name)
    instr = E.instruction_for(mode, axis) if emb.supports_instruction else None
    if mode in E.PER_AXIS_MODES:
        texts = _axis_texts(axis)
    else:
        texts = _all_labeled_texts() + (_corpus_texts() if args_with_corpus else [])
    df = _dedup(texts)
    if limit:
        df = df.head(limit)

    out = _cell_path(embedder_name, mode, axis)
    done = set()
    if out.exists():
        done = set(pd.read_parquet(out, columns=["text_id"])["text_id"])
    todo = df[~df["text_id"].isin(done)].reset_index(drop=True)
    print(f"[{embedder_name}/{mode}{'/'+axis if axis and mode=='per_axis' else ''}] "
          f"{len(todo)}/{len(df)} texts to embed (instr={instr!r})", flush=True)
    if todo.empty:
        print("  nothing to do.", flush=True)
        return 0

    # Split into per-request chunks (<= backend max_batch), then process
    # `concurrency` chunks at a time to exploit the endpoint's concurrent-request
    # capacity. Flush + progress after each wave (incremental checkpointing).
    bs = min(batch_size, emb.max_batch)
    chunks = [todo.iloc[i:i + bs] for i in range(0, len(todo), bs)]
    t0 = time.time()
    done_n = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for w in range(0, len(chunks), concurrency):
            wave = chunks[w:w + concurrency]
            futs = [ex.submit(emb.embed_request, c["text"].tolist(), instr) for c in wave]
            buf: list[dict] = []
            for c, fut in zip(wave, futs):
                vecs = fut.result()
                for tid, txt, v in zip(c["text_id"], c["text"], vecs):
                    buf.append({"text_id": tid, "text": txt, "embedding": v.tolist()})
            _flush(out, buf)
            done_n += sum(len(c) for c in wave)
            qps = done_n / max(time.time() - t0, 1e-9)
            print(f"  {done_n}/{len(todo)} | {qps:.0f} q/s | dim={emb.dim} | "
                  f"checkpointed (conc={concurrency})", flush=True)

    meta = {"embedder": embedder_name, "mode": mode, "axis": axis, "dim": emb.dim,
            "instruction": instr, "n_texts": int(len(df)),
            "wall_sec": round(time.time() - t0, 1)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"[{embedder_name}/{mode}] done -> {out} (dim {emb.dim})", flush=True)
    return 0


args_with_corpus = False  # set in main()


def main() -> int:
    global args_with_corpus
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=E.REGISTRY, required=True)
    ap.add_argument("--instr-mode", choices=["none", "generic", "per_axis", "per_axis_pos"],
                    default=None, help="default: run all modes valid for the embedder")
    ap.add_argument("--axis", choices=SAFETY_AXES, default=None,
                    help="per_axis: limit to one axis (default: all three)")
    ap.add_argument("--with-corpus", action="store_true",
                    help="also embed the ~55k real corpus in the generic cell (GPU-hour batch)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=16,
                    help="concurrent requests (exploits the endpoint's max-concurrent)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    args_with_corpus = a.with_corpus

    modes = [a.instr_mode] if a.instr_mode else E.INSTR_MODES[a.embedder]
    for mode in modes:
        if mode in E.PER_AXIS_MODES:
            axes = [a.axis] if a.axis else SAFETY_AXES
            for ax in axes:
                embed_cell(a.embedder, mode, ax, a.batch_size, a.concurrency, a.limit)
        else:
            embed_cell(a.embedder, mode, None, a.batch_size, a.concurrency, a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
