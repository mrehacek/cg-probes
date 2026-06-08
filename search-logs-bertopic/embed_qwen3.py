"""Embed Czech queries with qwen3-embedding-4b via an OpenAI-compatible endpoint.

Two modes:
  --smoke   Send 20 queries in a single batch to verify the endpoint accepts list
            input and that vectors come back L2-normalized at the expected
            dimensionality. Prints shape, norms, and the first vector's first 8 values.
  (default) Production embedding run over `cache/preprocess/embed_input.parquet`:
            - resumable (skips query_keys already in output parquet)
            - tenacity exp-backoff on RateLimit/Timeout/Conn errors
            - flushes every K batches, so a Ctrl+C or crash loses ≤ K*batch_size rows
            - tqdm progress bar with running qps + ETA
            - writes <output>.meta.json with run-level metadata

Any OpenAI-compatible embedding endpoint serving qwen3-embedding-4b works (batch
list input). Point it at your own deployment.

Env:
  EMBED_API_KEY    bearer token
  EMBED_API_URL    base, e.g. https://<openai-compatible-endpoint>/v1/

Usage:
  python embed_qwen3.py --smoke
  python embed_qwen3.py                       # full production run
  python embed_qwen3.py --limit 1000          # subset for iteration
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

MODEL = "qwen3-embedding-4b"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache" / "embeddings"
PREPROCESS_DIR = ROOT / "cache" / "preprocess"

DEFAULT_INPUT = PREPROCESS_DIR / "embed_input.parquet"
DEFAULT_OUTPUT = CACHE_DIR / "qwen3-embedding-4b.parquet"
DEFAULT_META = CACHE_DIR / "qwen3-embedding-4b.meta.json"
FAILED_LOG = CACHE_DIR / "failed_queries.csv"

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


# ── Client ───────────────────────────────────────────────────────────────────


def _client() -> OpenAI:
    key = os.environ.get("EMBED_API_KEY")
    base = os.environ.get("EMBED_API_URL")
    if not key or not base:
        sys.exit("EMBED_API_KEY and EMBED_API_URL must be set in .env")
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
def _embed_batch(client: OpenAI, queries: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL, input=queries)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def _try_serial_for_smoke(client: OpenAI, queries: list[str]) -> np.ndarray:
    out: list[list[float]] = []
    for q in queries:
        resp = client.embeddings.create(model=MODEL, input=q)
        out.append(resp.data[0].embedding)
    return np.array(out, dtype=np.float32)


# ── Smoke ────────────────────────────────────────────────────────────────────


def smoke() -> int:
    client = _client()
    print(f"[smoke] {len(SMOKE_QUERIES)} queries → {MODEL} @ {os.environ.get('EMBED_API_URL')}")

    t0 = time.time()
    try:
        vecs = _embed_batch(client, SMOKE_QUERIES)
        mode = "BATCH"
    except Exception as e:
        print(f"[smoke] batch failed ({type(e).__name__}: {e}); falling back to serial")
        vecs = _try_serial_for_smoke(client, SMOKE_QUERIES)
        mode = "SERIAL"
    dt = time.time() - t0

    norms = np.linalg.norm(vecs, axis=1)
    print(f"[smoke] mode={mode}  shape={vecs.shape}  dim={vecs.shape[1]}  wall={dt:.2f}s")
    print(f"[smoke] L2 norms: min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}")
    print(f"[smoke] first vector head: {vecs[0][:8]}")
    print(f"[smoke] qps: {len(SMOKE_QUERIES) / dt:.1f}")

    e_idx = SMOKE_QUERIES.index("nemůžu dýchat")
    n_idx = SMOKE_QUERIES.index("mamograf objednání")
    bleed_idx = SMOKE_QUERIES.index("krvácím z rány po operaci")
    cos_emerg = float(vecs[bleed_idx] @ vecs[e_idx] / (norms[bleed_idx] * norms[e_idx]))
    cos_nav = float(vecs[bleed_idx] @ vecs[n_idx] / (norms[bleed_idx] * norms[n_idx]))
    print(f"[smoke] cos('krvácím...', 'nemůžu dýchat') = {cos_emerg:.4f}")
    print(f"[smoke] cos('krvácím...', 'mamograf objednání') = {cos_nav:.4f}")
    if cos_emerg <= cos_nav:
        print("[smoke] WARNING: emergency phrase is NOT closer to other emergency than to navigational")
    return 0


# ── Production run: resumable, flushed, retry-wrapped ────────────────────────


def _load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"input not found: {path}\nRun preprocess.py first to produce embed_input.parquet.")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    for col in ("query", "query_key"):
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
    """Append buffered rows to out_pq. Returns number of rows written."""
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
    tmp.replace(out_pq)  # atomic on Windows + POSIX
    return len(new_df)


def _log_failed(query_key: str, query: str, err: str) -> None:
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not FAILED_LOG.exists()
    with open(FAILED_LOG, "a", encoding="utf-8", newline="") as f:
        if new:
            f.write("query_key,query,error\n")
        # naive CSV escaping is fine — these rows are for human audit, not reload
        f.write(f'"{query_key}","{query.replace(chr(34), chr(39))}","{err.replace(chr(34), chr(39))}"\n')


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def full(
    input_path: Path,
    output_parquet: Path,
    batch_size: int,
    checkpoint_every: int,
    limit: int | None,
) -> int:
    df = _load_input(input_path)
    if limit:
        df = df.head(limit)
    df["query"] = df["query"].astype(str)
    df["query_key"] = df["query_key"].astype(str)

    done_keys = _load_done_keys(output_parquet)
    if done_keys:
        print(f"[resume] {len(done_keys):,} keys already embedded in {output_parquet.name}; skipping")
    todo = df[~df["query_key"].isin(done_keys)].reset_index(drop=True)
    n_total = len(df)
    n_todo = len(todo)
    print(f"[run] {n_todo:,} of {n_total:,} queries to embed  (model={MODEL}, batch={batch_size})")
    if n_todo == 0:
        print("[run] nothing to do; output already complete.")
        return 0

    queries = todo["query"].tolist()
    keys = todo["query_key"].tolist()

    client = _client()

    buf_vecs: list[np.ndarray] = []
    buf_keys: list[str] = []
    buf_queries: list[str] = []
    n_failed = 0
    n_retries_total = 0
    flush_threshold_rows = checkpoint_every * batch_size

    start_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # Graceful shutdown — Ctrl+C / SIGTERM: flush buffer, write meta, exit cleanly.
    interrupted = {"flag": False}

    def _handle_signal(signum, frame):  # noqa: ARG001
        interrupted["flag"] = True
        # Don't print from inside the handler on Windows — set flag and let the
        # main loop bail at the next batch boundary so we never abandon a half-batch.

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
            chunk_q = queries[i:i + batch_size]
            chunk_k = keys[i:i + batch_size]
            try:
                vecs = _embed_batch(client, chunk_q)
            except _RETRYABLE as e:
                # tenacity exhausted retries on a retryable family — abort run; buffer still flushes in finally.
                pbar.write(f"[abort] retries exhausted at batch starting i={i}: {type(e).__name__}: {e}")
                break
            except openai.BadRequestError as e:
                # Bad input in this batch. Try per-query so the rest of the batch survives.
                pbar.write(f"[badrequest] batch i={i} fell back to per-query: {e}")
                ok_vecs: list[np.ndarray] = []
                ok_keys: list[str] = []
                ok_queries: list[str] = []
                for q, k in zip(chunk_q, chunk_k):
                    try:
                        v1 = _embed_batch(client, [q])
                        ok_vecs.append(v1[0])
                        ok_keys.append(k)
                        ok_queries.append(q)
                    except Exception as ee:
                        _log_failed(k, q, f"{type(ee).__name__}: {ee}")
                        n_failed += 1
                if ok_vecs:
                    vecs = np.stack(ok_vecs, axis=0)
                    chunk_k = ok_keys
                    chunk_q = ok_queries
                else:
                    pbar.update(len(chunk_q))
                    continue

            # tenacity.statistics is per-call — best-effort accumulation
            try:
                n_retries_total += _embed_batch.statistics.get("attempt_number", 1) - 1
            except Exception:
                pass

            buf_vecs.append(vecs)
            buf_keys.extend(chunk_k)
            buf_queries.extend(chunk_q)
            pbar.update(len(chunk_q))

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
            "endpoint": os.environ.get("EMBED_API_URL"),
            "dim": 2560,
            "instruction": None,
            "input_path": str(input_path),
            "output_path": str(output_parquet),
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
        _write_meta(DEFAULT_META if output_parquet == DEFAULT_OUTPUT else
                    output_parquet.with_suffix(".meta.json"), meta)
        print(json.dumps(meta, indent=2, ensure_ascii=False))

    if interrupted["flag"]:
        print("[run] interrupted; partial output is safe — rerun the same command to resume.")
        return 130
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="20-query probe to verify batch support and dim")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help=f"Input parquet/csv (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Output parquet (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="Flush to parquet every N batches (default 50 = 1000 rows)")
    ap.add_argument("--limit", type=int, default=None, help="Embed only first N rows (iteration)")
    args = ap.parse_args()

    if args.smoke:
        return smoke()
    return full(args.input, args.output, args.batch_size, args.checkpoint_every, args.limit)


if __name__ == "__main__":
    sys.exit(main())
