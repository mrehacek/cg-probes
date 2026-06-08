"""P2 step 4 — synthesize contrastive grade-2 (and missing grade-1) positives.

The golden constraint-sampler consumed *every* real rare safety pick, so after
the cluster-disjoint golden exclusion the anchor pool has **zero** real MU2, PU1
and PU2 queries (see `build_anchor_pool` sparsity table). The probe still needs
grade-2 positives to compute the difference-in-means direction, so those cells
must be synthesized. ET is dense in real data and is NOT synthesized.

Design decisions (user, 2026-06-06):
  * Generator = **gemini-3.5-flash** — a different model family than the
    golden-set eval positives (gpt-5.4), so separability that survives this
    mismatch can't be a shared synthetic-style artifact.
  * Moderate, topic-diverse scale: expand many anchors across many clusters with
    a per-cluster cap (not maximal per-cluster ladders — DoM saturates early and
    extra synthetic only amplifies the synthetic confound).
  * Every variant is **anchor-paired** (keeps `anchor_query_key`) so the probe
    can compute a within-anchor *paired* difference, cancelling the shared topic
    / ET component (the PU<->ET collinearity fix).
  * Hard realism: the prompt forbids forcing a marker onto an unsuitable topic
    (no "glukan chci umřít") and bans rubric jargon ("náhlý"). A second model,
    **gpt-5.4-mini** (blind, different family from the gemini generator),
    re-grades each variant; `verifier_confirmed = verifier_grade == target`.

Anchor sources (real PU0/MU0/MU1, natural-language only, per-cluster capped):
  * MU: real MU1 (alarm symptoms) + MU0 on ET1 illness topics  -> target MU2
  * PU: real PU0 on ET2 (high-load) + ET1 (mild-load) topics    -> target PU1, PU2

Output, one parquet per axis: `contrastive/cache/synth_variants_{axis}.parquet`
Plus an HTML eyeball report. LLMClient caches every call by content hash, so
re-runs are free/resumable. Run from the repo root:
    python -m contrastive.synth_variants --axes MU PU
"""

from __future__ import annotations

import argparse
import asyncio
import html
import re
import sys

import pandas as pd

from contrastive import p2_io
from contrastive.llm_client import LLMClient
from contrastive.schemas import SynthVariantSet
from golden.llm_annotator.pipeline._schema import AxisJudgeVerdict
from golden.pipeline_v2._io import write_parquet_atomic

GENERATOR = "gemini-3.5-flash"
TEMPLATE_VERSION = "synth_variant_v3"  # bumped: realism hard-rule + jargon ban
TEMPERATURE = 0.8  # >=0.7 for generation (house rule)

# Independent verifier: DIFFERENT family than the gemini generator, blind to the
# requested target grade. Mirrors golden PAPER.md step 4 (generate->verify->keep).
# gpt-5.4-mini rejects reasoning_effort='minimal' (project memory) -> 'low'.
VERIFIER_MODEL = "gpt-5.4-mini"
VERIFIER_EFFORT = "low"
JUDGE_PROMPT = p2_io.REPO / "golden" / "llm_annotator" / "prompts" / "axis_judge.txt"

PER_CLUSTER_CAP = 3

# --- anchor selection --------------------------------------------------------

# Code/dose-like stubs (biomarker codes, Gy doses, gene loci) are not plausible
# hosts for a rewrite. Require a natural-language patient query.
_ALPHA_WORD = re.compile(r"[a-zá-ž]{4,}", re.IGNORECASE)


def _natural_query(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if len(text.split()) < 3:
        return False
    return len(_ALPHA_WORD.findall(text)) >= 2


def _cap_per_cluster(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    return df.groupby("supercluster_id", group_keys=False).head(cap)


def _select_mu(wide: pd.DataFrame, *, max_anchors: int, cap: int) -> pd.DataFrame:
    """Real MU1 alarm-symptom queries (all kept — rare/valuable), topped up with
    per-cluster-capped MU0 queries on ET1 illness topics (escalatable). The
    prompt's realism rule + verifier drop anchors that can't host MU2."""
    nat = wide[wide["query_text"].map(_natural_query)]
    mu1 = nat[nat["mu"] == 1].sort_values("query_key")
    mu0 = _cap_per_cluster(
        nat[(nat["mu"] == 0) & (nat["et"] == 1)].sort_values("query_key"), cap
    )
    need = max(0, max_anchors - len(mu1))
    return pd.concat([mu1, mu0.head(need)], ignore_index=True)


def _select_pu(wide: pd.DataFrame, *, max_anchors: int, cap: int,
               et2_frac: float = 0.6) -> pd.DataFrame:
    """Real PU0 anchors across MIXED topic load so the PU direction is not
    confounded with ET: a `et2_frac` share from high-load ET2 topics, the rest
    from mild-load ET1 topics, each per-cluster capped."""
    nat = wide[(wide["pu"] == 0) & wide["query_text"].map(_natural_query)]
    n2 = int(max_anchors * et2_frac)
    n1 = max_anchors - n2
    et2 = _cap_per_cluster(nat[nat["et"] == 2].sort_values("query_key"), cap).head(n2)
    et1 = _cap_per_cluster(nat[nat["et"] == 1].sort_values("query_key"), cap).head(n1)
    return pd.concat([et2, et1], ignore_index=True)


SYNTH_JOBS: dict[str, dict] = {
    "MU": {"select": _select_mu, "target_grades": [2], "default_max": 800},
    "PU": {"select": _select_pu, "target_grades": [1, 2], "default_max": 1000},
}


def _per_query_frame() -> pd.DataFrame:
    """Pivot anchors_safety into one row per query with mu/pu/et ordinal cols."""
    a = pd.read_parquet(p2_io.ANCHORS_SAFETY)
    wide = a.pivot_table(
        index=["query_key", "query_text", "supercluster_id", "clinical_relevance"],
        columns="axis", values="grade", aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    for ax in ("MU", "PU", "ET"):
        wide[ax.lower()] = wide.get(ax)
    return wide


# --- generation --------------------------------------------------------------

def _render_user(axis: str, anchor_text: str, anchor_grade_token: str, targets: list[int]) -> str:
    _, user_tmpl = p2_io.load_synth_prompt()
    target_block = "\n".join(f"- {p2_io.int_to_token(axis, g)}" for g in targets)
    return user_tmpl.format(
        axis_short_code=axis,
        axis_long_name=p2_io.AXIS_LONG_NAMES_CS[axis],
        axis_grades_csv=p2_io.axis_grades_csv(axis),
        axis_rubric=p2_io.load_axis_rubric(axis),
        anchor_query=anchor_text,
        anchor_grade=anchor_grade_token,
        target_grades_block=target_block,
    )


CHUNK = 40             # anchors per checkpoint (incremental save + progress line)
CALL_TIMEOUT = 90.0    # hard per-call ceiling: one hung flex call can't stall the batch


async def _bounded(coro):
    """Await with a hard timeout; any failure/timeout -> None (caller drops it)."""
    try:
        return await asyncio.wait_for(coro, timeout=CALL_TIMEOUT)
    except Exception:
        return None


async def _gen_anchor(axis, row, targets, target_tokens, system, client) -> list[dict]:
    anchor_token = p2_io.int_to_token(axis, int(row[axis.lower()]))
    user = _render_user(axis, row["query_text"], anchor_token, targets)
    res = await _bounded(client.call_structured(
        phase=f"p2_synth_{axis}", template_version=TEMPLATE_VERSION,
        system=system, user=user, schema_model=SynthVariantSet, temperature=TEMPERATURE))
    if res is None:
        return []
    parsed, _ = res
    out = []
    for v in parsed.variants:
        if v.target_grade not in target_tokens:
            continue
        out.append({
            "axis": axis, "anchor_query_key": row["query_key"],
            "anchor_text": row["query_text"], "anchor_grade": int(row[axis.lower()]),
            "supercluster_id": row["supercluster_id"],
            "clinical_relevance": row["clinical_relevance"],
            "target_grade": p2_io.token_to_int(v.target_grade),
            "target_grade_token": v.target_grade, "text": v.text,
            "justification_cs": v.justification_cs, "generator": GENERATOR, "source": "synth",
        })
    return out


async def _verify_text(axis, text, system, user_tmpl, rubric, client):
    user = user_tmpl.format(
        axis_short_code=axis, axis_long_name=p2_io.AXIS_LONG_NAMES_CS[axis],
        axis_grades_csv=p2_io.axis_grades_csv(axis), axis_rubric=rubric, query_text=text)
    res = await _bounded(client.call_structured(
        phase=f"p2_verify_{axis}", template_version="axis_judge_v1",
        system=system, user=user, schema_model=AxisJudgeVerdict, reasoning_effort=VERIFIER_EFFORT))
    if res is None:
        return (None, None)
    parsed, _ = res
    return (parsed.grade, parsed.confidence)


async def _run_axis(axis: str, pool: pd.DataFrame, targets: list[int],
                    gen: LLMClient, ver: LLMClient) -> pd.DataFrame:
    """Chunked, resumable, checkpointing driver for one axis. Writes the output
    parquet after every CHUNK anchors and prints a flushed progress line, so a
    crash loses <=1 chunk and progress is visible live."""
    system_g, _ = p2_io.load_synth_prompt()
    system_v, user_v = p2_io.split_prompt_file(JUDGE_PROMPT)
    rubric = p2_io.load_axis_rubric(axis)
    target_tokens = {p2_io.int_to_token(axis, g) for g in targets}
    out_path = p2_io.synth_variants_path(axis)

    acc: list[dict] = []
    done: set = set()
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        if {"anchor_query_key", "verifier_confirmed"} <= set(prev.columns):
            acc = prev.to_dict("records")
            done = set(prev["anchor_query_key"])
    todo = pool[~pool["query_key"].isin(done)].reset_index(drop=True)
    total = len(todo)
    print(f"[{axis}] {total} anchors to do ({len(done)} already done, {len(acc)} cached variants)",
          flush=True)

    for i in range(0, total, CHUNK):
        chunk = todo.iloc[i:i + CHUNK]
        gen_rows = await asyncio.gather(
            *(_gen_anchor(axis, r, targets, target_tokens, system_g, gen) for _, r in chunk.iterrows()))
        rows = [x for sub in gen_rows for x in sub]
        verdicts = await asyncio.gather(
            *(_verify_text(axis, r["text"], system_v, user_v, rubric, ver) for r in rows))
        for r, (g, c) in zip(rows, verdicts):
            r["verifier_grade"], r["verifier_confidence"] = g, c
            r["verifier_confirmed"] = (g == r["target_grade_token"])
        acc.extend(rows)
        df = pd.DataFrame(acc)
        write_parquet_atomic(df, out_path)
        conf = df["verifier_confirmed"].mean() if len(df) else 0.0
        print(f"[{axis}] {min(i + CHUNK, total)}/{total} anchors | {len(df)} variants | "
              f"{conf:.0%} confirmed | checkpointed", flush=True)
    return pd.DataFrame(acc)


# --- report ------------------------------------------------------------------

def _write_report(axis: str, df: pd.DataFrame, path) -> None:
    conf = df["verifier_confirmed"].mean() if len(df) else 0.0
    kept = df[df["verifier_confirmed"]] if "verifier_confirmed" in df else df
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:1150px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 9px;vertical-align:top}"
        "td.t{text-align:left;max-width:430px}th{background:#f0f0f0}"
        ".no{background:#fff0f0}</style></head><body>",
        f"<h1>P2 synth variants — {axis} ({p2_io.AXIS_LONG_NAMES_EN[axis]})</h1>",
        f"<p>generator <b>{GENERATOR}</b> · verifier <b>{VERIFIER_MODEL}</b> (blind) · "
        f"{len(df)} generated, <b>{len(kept)} verifier-confirmed</b> "
        f"({conf:.0%}) · confirmed by target: {kept['target_grade_token'].value_counts().to_dict()}</p>",
        "<p>Rows with pink background were rejected by the verifier (kept in parquet, "
        "flagged <code>verifier_confirmed=False</code>; assemble_pairs drops them).</p>",
        "<table><tr><th>target</th><th>verif</th><th class='t'>anchor (grade)</th>"
        "<th class='t'>synthetic variant</th></tr>",
    ]
    show = df.sample(min(60, len(df)), random_state=0) if len(df) else df
    for _, r in show.iterrows():
        cls = "" if r["verifier_confirmed"] else " class='no'"
        parts.append(
            f"<tr{cls}><td>{r['target_grade_token']}</td><td>{r['verifier_grade']}</td>"
            f"<td class='t'>{html.escape(str(r['anchor_text']))} "
            f"<i>({p2_io.int_to_token(axis, int(r['anchor_grade']))})</i></td>"
            f"<td class='t'>{html.escape(str(r['text']))}</td></tr>"
        )
    parts.append("</table></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


# --- driver ------------------------------------------------------------------

async def main_async(args) -> int:
    p2_io.ensure_cache()
    wide = _per_query_frame()
    gen = LLMClient.for_gemini(model=GENERATOR, concurrency=args.concurrency)
    # request_timeout caps the underlying call; _bounded(CALL_TIMEOUT) is the outer guard.
    ver = LLMClient(model=VERIFIER_MODEL, concurrency=args.concurrency,
                    service_tier="flex", request_timeout=CALL_TIMEOUT)

    for axis in (args.axes or list(SYNTH_JOBS.keys())):
        job = SYNTH_JOBS[axis]
        max_anchors = args.max_mu if axis == "MU" else args.max_pu
        pool = job["select"](wide, max_anchors=max_anchors, cap=args.cap)
        print(f"[{axis}] pool={len(pool)} anchors, {pool['supercluster_id'].nunique()} clusters, "
              f"targets={job['target_grades']}", flush=True)
        df = await _run_axis(axis, pool, job["target_grades"], gen, ver)

        report = p2_io.synth_variants_path(axis).with_name(f"synth_variants_{axis}_report.html")
        _write_report(axis, df, report)
        kept = df[df["verifier_confirmed"]]
        print(f"[{axis}] DONE: {len(df)} variants, {len(kept)} confirmed "
              f"({df['verifier_confirmed'].mean():.0%}); by grade "
              f"{kept['target_grade_token'].value_counts().to_dict()}; report {report}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="*", choices=list(SYNTH_JOBS.keys()), default=None)
    ap.add_argument("--max-mu", type=int, default=SYNTH_JOBS["MU"]["default_max"])
    ap.add_argument("--max-pu", type=int, default=SYNTH_JOBS["PU"]["default_max"])
    ap.add_argument("--cap", type=int, default=PER_CLUSTER_CAP,
                    help="max anchors per supercluster (topic diversity)")
    ap.add_argument("--concurrency", type=int, default=16)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
