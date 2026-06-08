"""Embed Czech queries with microsoft/harrier-oss-v1-27b via a Hugging Face
Inference Endpoint running the Text Embeddings Inference (TEI) container.

Two modes:
  --smoke   Send 20 queries in a single batch to verify the endpoint accepts
            list input, that vectors come back at dim=5376, are L2-normalized,
            and that the instruction prefix improves topical separation.
  (default) Production embedding run over an input parquet:
            - resumable (skips query_keys already in output parquet)
            - tenacity exp-backoff on RateLimit/Timeout/Conn errors
            - flushes every K batches; a Ctrl+C or crash loses ≤ K*batch_size rows
            - tqdm progress bar with running qps + ETA
            - writes <output>.meta.json with run-level metadata (incl. the
              exact instruction string used)

Harrier is instruction-tuned: each *query* must be prefixed with
  Instruct: {task}\\nQuery: {text}
TEI has NO API-level prompt parameter, so we prepend it client-side here.
*Documents* are embedded raw — pass --no-instruct in that case.

The endpoint exposes both /embed (native TEI) and /v1/embeddings (OpenAI-compat).
This script uses the OpenAI-compat path so the call shape matches embed_qwen3.py.

Env (in .env at repo root or cikm-ds/):
  HF_TOKEN                       HF user token with read access to the endpoint
  HF_HARRIER_ENDPOINT_URL        e.g. https://<id>.endpoints.huggingface.cloud
                                 (script appends /v1 for OpenAI-compat)

Usage:
  python embed_harrier_hfie.py --smoke
  python embed_harrier_hfie.py --input cache/preprocess/embed_input.parquet
  python embed_harrier_hfie.py --instruct-preset sts
  python embed_harrier_hfie.py --instruct "Given a clinical question, retrieve oncology guidelines that answer it"
  python embed_harrier_hfie.py --no-instruct --input docs.parquet --output cache/embeddings/harrier-docs.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")  # repo-root fallback

MODEL = "harrier-oss-v1-27b"
EMBED_DIM = 5376

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "embeddings"
PREPROCESS_DIR = ROOT / "cache" / "preprocess"

DEFAULT_INPUT = PREPROCESS_DIR / "embed_input.parquet"
DEFAULT_OUTPUT = CACHE_DIR / "harrier-oss-v1-27b.parquet"
FAILED_LOG = CACHE_DIR / "failed_queries_harrier.csv"

# ── Instruction presets ──────────────────────────────────────────────────────
# These are the *task* sentences that go after "Instruct:" and before
# "\nQuery: ". Add new presets here; the chosen preset (or --instruct override)
# is recorded verbatim in meta.json so re-runs are reproducible.

INSTRUCT_PRESETS: dict[str, str] = {
    # Microsoft's official Harrier prompt for web retrieval.
    "web_search_query":
        "Given a web search query, retrieve relevant passages that answer the query",
    # Official STS-style prompt — best fit for clustering / similarity work.
    "sts":
        "Retrieve semantically similar text",
    # Notebook default, used in examples/harrier-27b.ipynb. Domain-aware.
    "medical_retrieval":
        "Given a web search query, retrieve relevant medical information that answers the query",
    # Patient-facing variant — emphasises layperson framing for RAG.
    "patient_question":
        "Given a patient's health question, retrieve relevant medical information in clear, patient-friendly language",
    # For BERTopic-style clustering of short queries.
    "cluster_queries":
        "Identify the topic or theme of the given query for clustering",
}
DEFAULT_PRESET = "medical_retrieval"

SMOKE_QUERIES = [
    "co znamená HER2 pozitivní",
    "kolik mi zbývá času",
    "kde je radiologie",
    "mám už druhý den průjem 6× denně po chemo",
    "jsem zoufalá",
    "nemůžu dýchat",
    "vedlejší účinky tamoxifenu",
    "mamograf objednání",
    "co je biologická léčba",
    "bude umírání bolet",
    "krvácím z rány po operaci",
    "měl bych přestat s chemoterapií?",
    "trápí mě úzkost po každé kontrole",
    "rakovina prsu metastázy do jater",
    "ki-67 80%",
    "kdy mi spadnou vlasy",
    "co znamená T3N1M0",
    "jak probíhá radioterapie",
    "doktor řekl ok ale bolí mě záda",
    "porovnání tamoxifen vs anastrozol",
]


# ── Prompting ────────────────────────────────────────────────────────────────


def resolve_instruction(preset: str | None, override: str | None, disable: bool) -> str | None:
    """Pick the task string. None means 'embed raw' (document mode)."""
    if disable:
        return None
    if override:
        return override.strip()
    return INSTRUCT_PRESETS[preset or DEFAULT_PRESET]


def fmt_input(text: str, task: str | None) -> str:
    """Apply Harrier's instruction template, or pass-through for documents."""
    if task is None:
        return text
    return f"Instruct: {task}\nQuery: {text}"


# ── Client ───────────────────────────────────────────────────────────────────


def _client() -> OpenAI:
    key = os.environ.get("HF_TOKEN")
    base = os.environ.get("HF_HARRIER_ENDPOINT_URL")
    if not key or not base:
        sys.exit("HF_TOKEN and HF_HARRIER_ENDPOINT_URL must be set in .env")
    # Append /v1 for OpenAI-compat path if the user didn't include it.
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    return OpenAI(api_key=key, base_url=base)


# ── Embed batch with retries ─────────────────────────────────────────────────


_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=32),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _embed_batch(client: OpenAI, inputs: list[str]) -> np.ndarray:
    # TEI ignores the `model` field but the OpenAI SDK requires it.
    resp = client.embeddings.create(model="text-embeddings-inference", input=inputs)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


# ── Smoke ────────────────────────────────────────────────────────────────────


def smoke(task: str | None) -> int:
    client = _client()
    base = os.environ.get("HF_HARRIER_ENDPOINT_URL")
    print(f"[smoke] {len(SMOKE_QUERIES)} queries → {MODEL} @ {base}")
    print(f"[smoke] instruction: {task!r}")

    formatted = [fmt_input(q, task) for q in SMOKE_QUERIES]
    t0 = time.time()
    vecs = _embed_batch(client, formatted)
    dt = time.time() - t0

    norms = np.linalg.norm(vecs, axis=1)
    print(f"[smoke] shape={vecs.shape}  dim={vecs.shape[1]}  wall={dt:.2f}s")
    print(f"[smoke] L2 norms: min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}")
    print(f"[smoke] first vector head: {vecs[0][:8]}")
    print(f"[smoke] qps: {len(SMOKE_QUERIES) / dt:.1f}")

    if vecs.shape[1] != EMBED_DIM:
        print(f"[smoke] WARNING: expected dim={EMBED_DIM}, got {vecs.shape[1]}")

    # Same sanity check as the qwen3 smoke — emergency cluster should be tighter
    # than emergency↔navigational.
    e_idx = SMOKE_QUERIES.index("nemůžu dýchat")
    n_idx = SMOKE_QUERIES.index("mamograf objednání")
    bleed_idx = SMOKE_QUERIES.index("krvácím z rány po operaci")
    cos_emerg = float(vecs[bleed_idx] @ vecs[e_idx] / (norms[bleed_idx] * norms[e_idx]))
    cos_nav = float(vecs[bleed_idx] @ vecs[n_idx] / (norms[bleed_idx] * norms[n_idx]))
    print(f"[smoke] cos('krvácím...', 'nemůžu dýchat')      = {cos_emerg:.4f}")
    print(f"[smoke] cos('krvácím...', 'mamograf objednání') = {cos_nav:.4f}")
    if cos_emerg <= cos_nav:
        print("[smoke] WARNING: emergency phrase is NOT closer to other emergency than to navigational")
    return 0


# ── Production run: resumable, flushed, retry-wrapped ────────────────────────


def _load_input(path: Path, query_col: str, key_col: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"input not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    for col in (query_col, key_col):
        if col not in df.columns:
            sys.exit(f"input missing required column '{col}'; got {list(df.columns)}")
    return df


def _load_done_keys(out_pq: Path) -> set[str]:
    if not out_pq.exists():
        return set()
    try:
        existing = pd.read_parquet(out_pq, columns=["query_key"])
    except Exception as e:
        print(f"[warn] could not read existing {out_pq.name} for resume: {e}", file=sys.stderr)
        return set()
    return set(existing["query_key"].astype(str))


def _flush(out_pq: Path, keys: list[str], queries: list[str], vec_chunks: list[np.ndarray]) -> int:
    if not vec_chunks:
        return 0
    new_vecs = np.concatenate(vec_chunks, axis=0)
    new_df = pd.DataFrame({
        "query_key": keys,
        "query": queries,
        "embedding": [v.tolist() for v in new_vecs],
    })
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    if out_pq.exists():
        existing = pd.read_parquet(out_pq)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    tmp = out_pq.with_suffix(out_pq.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(out_pq)
    return len(new_df)


def _log_failed(query_key: str, query: str, err: str) -> None:
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not FAILED_LOG.exists()
    with open(FAILED_LOG, "a", encoding="utf-8", newline="") as f:
        if new:
            f.write("query_key,query,error\n")
        f.write(f'"{query_key}","{query.replace(chr(34), chr(39))}","{err.replace(chr(34), chr(39))}"\n')


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def full(
    input_path: Path,
    output_parquet: Path,
    task: str | None,
    instruct_label: str,
    batch_size: int,
    checkpoint_every: int,
    limit: int | None,
    query_col: str,
    key_col: str,
) -> int:
    df = _load_input(input_path, query_col, key_col)
    if limit:
        df = df.head(limit)
    df[query_col] = df[query_col].astype(str)
    df[key_col] = df[key_col].astype(str)

    done_keys = _load_done_keys(output_parquet)
    if done_keys:
        print(f"[resume] {len(done_keys):,} keys already embedded in {output_parquet.name}; skipping")
    todo = df[~df[key_col].isin(done_keys)].reset_index(drop=True)
    n_total = len(df)
    n_todo = len(todo)
    print(f"[run] {n_todo:,} of {n_total:,} rows to embed  "
          f"(model={MODEL}, batch={batch_size}, instruct={instruct_label})")
    if n_todo == 0:
        print("[run] nothing to do; output already complete.")
        return 0

    raw_texts = todo[query_col].tolist()
    keys = todo[key_col].tolist()
    inputs = [fmt_input(t, task) for t in raw_texts]

    client = _client()

    buf_vecs: list[np.ndarray] = []
    buf_keys: list[str] = []
    buf_queries: list[str] = []
    n_failed = 0
    n_retries_total = 0
    flush_threshold_rows = checkpoint_every * batch_size

    start_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    interrupted = {"flag": False}

    def _handle_signal(signum, frame):  # noqa: ARG001
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, OSError):
            pass

    pbar = tqdm(total=n_todo, unit="q", smoothing=0.1, dynamic_ncols=True)
    last_qps = 0.0
    try:
        for i in range(0, n_todo, batch_size):
            if interrupted["flag"]:
                break
            chunk_in = inputs[i:i + batch_size]
            chunk_raw = raw_texts[i:i + batch_size]
            chunk_k = keys[i:i + batch_size]
            try:
                vecs = _embed_batch(client, chunk_in)
            except _RETRYABLE as e:
                pbar.write(f"[abort] retries exhausted at batch i={i}: {type(e).__name__}: {e}")
                break
            except openai.BadRequestError as e:
                pbar.write(f"[badrequest] batch i={i} fell back to per-query: {e}")
                ok_vecs: list[np.ndarray] = []
                ok_keys: list[str] = []
                ok_queries: list[str] = []
                for one_in, one_raw, one_k in zip(chunk_in, chunk_raw, chunk_k):
                    try:
                        v1 = _embed_batch(client, [one_in])
                        ok_vecs.append(v1[0])
                        ok_keys.append(one_k)
                        ok_queries.append(one_raw)
                    except Exception as ee:
                        _log_failed(one_k, one_raw, f"{type(ee).__name__}: {ee}")
                        n_failed += 1
                if ok_vecs:
                    vecs = np.stack(ok_vecs, axis=0)
                    chunk_k = ok_keys
                    chunk_raw = ok_queries
                else:
                    pbar.update(len(chunk_in))
                    continue

            try:
                n_retries_total += _embed_batch.statistics.get("attempt_number", 1) - 1
            except Exception:
                pass

            buf_vecs.append(vecs)
            buf_keys.extend(chunk_k)
            buf_queries.extend(chunk_raw)
            pbar.update(len(chunk_in))

            elapsed = time.time() - t0
            done = pbar.n
            last_qps = done / elapsed if elapsed > 0 else 0.0
            pbar.set_postfix_str(f"{last_qps:.1f} qps  retries={n_retries_total}  failed={n_failed}")

            if len(buf_keys) >= flush_threshold_rows:
                written = _flush(output_parquet, buf_keys, buf_queries, buf_vecs)
                pbar.write(f"[flush] +{written:,} rows -> {output_parquet.name}")
                buf_vecs, buf_keys, buf_queries = [], [], []
    finally:
        pbar.close()
        if buf_vecs:
            written = _flush(output_parquet, buf_keys, buf_queries, buf_vecs)
            print(f"[flush] final +{written:,} rows -> {output_parquet}")

        end_iso = datetime.now(timezone.utc).isoformat()
        wall = time.time() - t0
        total_in_out = 0
        if output_parquet.exists():
            total_in_out = len(pd.read_parquet(output_parquet, columns=["query_key"]))
        meta = {
            "model": MODEL,
            "endpoint": os.environ.get("HF_HARRIER_ENDPOINT_URL"),
            "engine": "tei",
            "dim": EMBED_DIM,
            "instruction_label": instruct_label,
            "instruction_task": task,
            "instruction_template": "Instruct: {task}\nQuery: {text}" if task else None,
            "input_path": str(input_path),
            "output_path": str(output_parquet),
            "query_column": query_col,
            "key_column": key_col,
            "batch_size": batch_size,
            "checkpoint_every_batches": checkpoint_every,
            "input_rows_total": n_total,
            "rows_to_embed_this_run": n_todo,
            "rows_in_output": total_in_out,
            "rows_failed_this_run": n_failed,
            "retries_total_approx": n_retries_total,
            "qps_avg": round(last_qps, 2),
            "wall_clock_sec": round(wall, 2),
            "start_ts": start_iso,
            "end_ts": end_iso,
            "interrupted": interrupted["flag"],
        }
        meta_path = output_parquet.with_suffix(".meta.json")
        _write_meta(meta_path, meta)
        print(json.dumps(meta, indent=2, ensure_ascii=False))

    if interrupted["flag"]:
        print("[run] interrupted; partial output is safe — rerun the same command to resume.")
        return 130
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available instruction presets:\n  " + "\n  ".join(
            f"{k:20s} {v}" for k, v in INSTRUCT_PRESETS.items()
        ),
    )
    ap.add_argument("--smoke", action="store_true",
                    help="20-query probe to verify endpoint, dim, and instruction effect")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help=f"Input parquet/csv (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Output parquet (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--query-col", default="query",
                    help="Column name containing text to embed (default: query)")
    ap.add_argument("--key-col", default="query_key",
                    help="Unique row id column used for resume (default: query_key)")
    # Instruction control — mutually exclusive trio
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--instruct-preset", choices=sorted(INSTRUCT_PRESETS),
                   help=f"Use a named preset (default: {DEFAULT_PRESET})")
    g.add_argument("--instruct", type=str, metavar="TASK",
                   help='Custom task string, e.g. "Given a clinical question, retrieve oncology guidelines"')
    g.add_argument("--no-instruct", action="store_true",
                   help="Embed raw text without an Instruct: prefix (use this for documents)")

    ap.add_argument("--batch-size", type=int, default=64,
                    help="Inputs per request. H200 + TEI 16K-token batch handles 64-256 short Czech queries.")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="Flush to parquet every N batches (default 10)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Embed only first N rows (iteration)")
    args = ap.parse_args()

    task = resolve_instruction(args.instruct_preset, args.instruct, args.no_instruct)
    if args.no_instruct:
        instruct_label = "none"
    elif args.instruct:
        instruct_label = "custom"
    else:
        instruct_label = args.instruct_preset or DEFAULT_PRESET

    if args.smoke:
        return smoke(task)
    return full(
        args.input, args.output, task, instruct_label,
        args.batch_size, args.checkpoint_every, args.limit,
        args.query_col, args.key_col,
    )


if __name__ == "__main__":
    sys.exit(main())
