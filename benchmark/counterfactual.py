"""Tier-2 counterfactual / minimal-pair probes — the strongest construct-validity
evidence available without clinician labels.

A faithful safety probe must satisfy two behavioural properties:
  * INVARIANCE — a topic-preserving paraphrase that does NOT change the safety
    grade should barely move the projection.
  * SENSITIVITY — a minimal edit that raises (lowers) the grade by one step
    should move the projection up (down), in the correct direction.

We sample grade-1 seed queries per axis (headroom to flip both ways), ask
gemini-3.5-flash for {paraphrase, flip_up, flip_down} holding the topic constant
(prompts/counterfactual_v1.txt, verifier-free — the *direction* of movement is
what we measure, not an absolute grade), embed seed+rewrites with the always-on
API embedders (gemini generic, openai3large), project onto the saved generic-mode
axis direction, and report:
  * mean |Δproj| for paraphrase (small) vs flips (large), as effect sizes
    (Δ / projection-std on that axis);
  * flip-direction-correct rate: sign(Δ_up)>0 and sign(Δ_down)<0;
  * a sensitivity:invariance ratio (>1 == probe tracks the concept, not surface).

Only gemini + openai3large are used (no HF endpoint needed to embed new text).
Output: benchmark/cache/counterfactual/{pairs.parquet, results.json}.
  python -m benchmark.counterfactual --limit 6   # smoke
  python -m benchmark.counterfactual             # full (~40 seeds/axis)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from benchmark import embedders as E
from benchmark.run_embed import _tid
from contrastive import p2_io
from contrastive.llm_client import LLMClient
from contrastive.p2_io import REPO, SAFETY_AXES

PROMPT = REPO / "benchmark" / "prompts" / "counterfactual_v1.txt"
DIRDIR = REPO / "benchmark" / "cache" / "directions"
OUT_DIR = REPO / "benchmark" / "cache" / "counterfactual"
PAIRS = OUT_DIR / "pairs.parquet"
RESULTS = OUT_DIR / "results.json"

GENERATOR = "gemini-3.5-flash"
TEMPLATE_VERSION = "counterfactual_v1"
N_SEEDS = 40            # grade-1 seeds per axis
CHUNK = 20
CALL_TIMEOUT = 90.0
EMBEDDERS = ["gemini", "openai3large"]   # always-on APIs (no HF endpoint)


class CounterfactualSet(BaseModel):
    paraphrase: str = Field(..., description="same topic AND same target-axis grade")
    flip_up: str = Field("", description="minimal edit: target axis +1 grade (\"\" if impossible)")
    flip_down: str = Field("", description="minimal edit: target axis -1 grade (\"\" if impossible)")


def _select_seeds(axis: str, n: int, seed: int = 0) -> pd.DataFrame:
    """Grade-1 seeds, prefer real (picker) source, cluster-diverse."""
    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    g1 = ds[ds["grade"] == 1].copy()
    if g1.empty:
        return g1
    g1["_real"] = (g1["source"].astype(str) == "picker").astype(int)
    # one per supercluster first (diversity), real preferred, then fill
    g1 = g1.sample(frac=1.0, random_state=seed)
    g1 = g1.sort_values("_real", ascending=False)
    first = g1.drop_duplicates("supercluster_id")
    rest = g1[~g1.index.isin(first.index)]
    out = pd.concat([first, rest]).head(n).reset_index(drop=True)
    return out[["text", "supercluster_id", "query_key", "source"]]


async def _generate(axis: str, limit: int | None) -> pd.DataFrame:
    system, user_tmpl = p2_io.split_prompt_file(PROMPT)
    rubric = p2_io.load_axis_rubric(axis)
    client = LLMClient.for_gemini(model=GENERATOR, concurrency=8)
    seeds = _select_seeds(axis, limit or N_SEEDS)
    print(f"[cf/{axis}] {len(seeds)} grade-1 seeds", flush=True)

    async def one(row) -> dict | None:
        user = user_tmpl.format(
            axis_short_code=axis, axis_long_name=p2_io.AXIS_LONG_NAMES_CS[axis],
            grade=1, axis_rubric=rubric, query_text=row["text"])
        try:
            res = await asyncio.wait_for(client.call_structured(
                phase=f"counterfactual_{axis}", template_version=TEMPLATE_VERSION,
                system=system, user=user, schema_model=CounterfactualSet,
                temperature=0.8), timeout=CALL_TIMEOUT)
        except Exception as e:
            print(f"  [warn] {axis} {row['query_key']}: {e}", flush=True)
            return None
        v = res[0]
        return {"axis": axis, "seed": row["text"], "query_key": row["query_key"],
                "supercluster_id": row["supercluster_id"],
                "paraphrase": v.paraphrase.strip(),
                "flip_up": v.flip_up.strip(), "flip_down": v.flip_down.strip()}

    rows: list[dict] = []
    recs = seeds.to_dict("records")
    for i in range(0, len(recs), CHUNK):
        chunk = recs[i:i + CHUNK]
        got = await asyncio.gather(*(one(r) for r in chunk))
        rows.extend(r for r in got if r)
        print(f"  [cf/{axis}] {min(i+CHUNK, len(recs))}/{len(recs)} generated", flush=True)
    return pd.DataFrame(rows)


def _embed_unique(texts: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """{embedder: {text_id: vec}} for the generic-mode space (matches saved dirs)."""
    uniq = sorted({t for t in texts if isinstance(t, str) and t.strip()})
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in EMBEDDERS:
        emb = E.make_embedder(name)
        instr = E.instruction_for("generic", None) if emb.supports_instruction else None
        vecs: dict[str, np.ndarray] = {}
        bs = emb.max_batch
        for i in range(0, len(uniq), bs):
            batch = uniq[i:i + bs]
            res = emb.embed_request(batch, instr)
            for t, v in zip(batch, res):
                vecs[_tid(t)] = np.asarray(v, dtype=np.float64)
            print(f"  [embed/{name}] {min(i+bs, len(uniq))}/{len(uniq)}", flush=True)
        out[name] = vecs
    return out


def _proj(vecs: dict[str, np.ndarray], text: str, w: np.ndarray) -> float | None:
    v = vecs.get(_tid(text))
    if v is None:
        return None
    return float(v @ (w / (np.linalg.norm(w) + 1e-12)))


def analyse(pairs: pd.DataFrame) -> dict:
    all_texts: list[str] = []
    for col in ("seed", "paraphrase", "flip_up", "flip_down"):
        all_texts += pairs[col].astype(str).tolist()
    emb = _embed_unique(all_texts)

    results: dict = {"by_embedder": {}, "n_pairs": int(len(pairs))}
    for name in EMBEDDERS:
        vecs = emb[name]
        per_axis = {}
        for axis in SAFETY_AXES:
            wp = DIRDIR / f"{name}__generic__{axis}.npy"
            if not wp.exists():
                continue
            w = np.load(wp)
            sub = pairs[pairs["axis"] == axis]
            d_para, d_up, d_down = [], [], []
            seed_projs = []
            n_up_ok = n_down_ok = n_up = n_down = 0
            for _, r in sub.iterrows():
                s_seed = _proj(vecs, r["seed"], w)
                if s_seed is None:
                    continue
                seed_projs.append(s_seed)
                s_par = _proj(vecs, r["paraphrase"], w) if r["paraphrase"] else None
                if s_par is not None:
                    d_para.append(s_par - s_seed)
                if r["flip_up"]:
                    s = _proj(vecs, r["flip_up"], w)
                    if s is not None:
                        d_up.append(s - s_seed); n_up += 1; n_up_ok += (s - s_seed) > 0
                if r["flip_down"]:
                    s = _proj(vecs, r["flip_down"], w)
                    if s is not None:
                        d_down.append(s - s_seed); n_down += 1; n_down_ok += (s - s_seed) < 0
            std = float(np.std(seed_projs)) + 1e-12
            d_para, d_up, d_down = map(np.asarray, (d_para, d_up, d_down))
            inv = float(np.mean(np.abs(d_para)) / std) if len(d_para) else None
            sens = float(np.mean(np.concatenate([np.abs(d_up), np.abs(d_down)]) / std)) \
                if (len(d_up) + len(d_down)) else None
            per_axis[axis] = {
                "invariance_effect": inv,            # mean|Δ paraphrase| / std  (small good)
                "sensitivity_effect": sens,          # mean|Δ flip| / std        (large good)
                "ratio": (float(sens / inv) if inv and sens else None),
                "flip_up_mean_effect": (float(np.mean(d_up) / std) if len(d_up) else None),
                "flip_down_mean_effect": (float(np.mean(d_down) / std) if len(d_down) else None),
                "direction_correct_rate": (
                    float((n_up_ok + n_down_ok) / (n_up + n_down)) if (n_up + n_down) else None),
                "n_seed": len(seed_projs), "n_up": n_up, "n_down": n_down,
            }
            pa = per_axis[axis]
            print(f"[cf/{name}/{axis}] inv={pa['invariance_effect']} "
                  f"sens={pa['sensitivity_effect']} ratio={pa['ratio']} "
                  f"dir_ok={pa['direction_correct_rate']}", flush=True)
        results["by_embedder"][name] = per_axis
    return results


async def _gen_all(limit: int | None) -> pd.DataFrame:
    frames = [await _generate(ax, limit) for ax in SAFETY_AXES]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="seeds per axis (smoke)")
    ap.add_argument("--regen", action="store_true", help="regenerate pairs even if cached")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if PAIRS.exists() and not a.regen:
        pairs = pd.read_parquet(PAIRS)
        print(f"[cf] loaded {len(pairs)} cached pairs ({PAIRS})", flush=True)
    else:
        pairs = asyncio.run(_gen_all(a.limit))
        pairs.to_parquet(PAIRS, index=False)
        print(f"[cf] wrote {len(pairs)} pairs -> {PAIRS}", flush=True)

    results = analyse(pairs)
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {RESULTS}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
