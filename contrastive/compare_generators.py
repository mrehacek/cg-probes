"""P2 helper — head-to-head generator comparison for the synth-variant task.

Runs the SAME anchors through both candidate training-side generators
(gemini-3.5-flash and gpt-5.4) on the identical `synth_variant_v2` prompt, and
emits a side-by-side HTML report so a human can judge which model produces
better topic-preserving, register-preserving, correctly-graded synthetic
variants — before committing one as the training-label source.

Small by design (a handful of anchors per axis). Run from the repo root:
    python -m contrastive.compare_generators --n-mu 8 --n-pu 6
Output: contrastive/cache/generator_compare_report.html
"""

from __future__ import annotations

import argparse
import asyncio
import html
import sys

import pandas as pd

from contrastive import p2_io
from contrastive.llm_client import LLMClient
from contrastive.schemas import SynthVariantSet
from contrastive.synth_variants import SYNTH_JOBS, _per_query_frame, _render_user

MODELS = {
    "gemini-3.5-flash": dict(kind="gemini", temperature=0.8),
    "gpt-5.4": dict(kind="openai", reasoning_effort="low"),
}


def _client(kind: str) -> LLMClient:
    if kind == "gemini":
        return LLMClient.for_gemini(model="gemini-3.5-flash", concurrency=6)
    return LLMClient(model="gpt-5.4", concurrency=6, service_tier="flex")


async def _gen(client: LLMClient, cfg: dict, axis: str, row, targets: list[int]) -> list[dict]:
    system, _ = p2_io.load_synth_prompt()
    anchor_token = p2_io.int_to_token(axis, int(row[axis.lower()]))
    user = _render_user(axis, row["query_text"], anchor_token, targets)
    parsed, _ = await client.call_structured(
        phase=f"p2_compare_{axis}",
        template_version="synth_variant_v2",
        system=system,
        user=user,
        schema_model=SynthVariantSet,
        reasoning_effort=cfg.get("reasoning_effort", "minimal"),
        temperature=cfg.get("temperature"),
    )
    return [{"target": v.target_grade, "text": v.text, "just": v.justification_cs}
            for v in parsed.variants]


async def main_async(args) -> int:
    p2_io.ensure_cache()
    wide = _per_query_frame()
    clients = {name: _client(cfg["kind"]) for name, cfg in MODELS.items()}

    sample: dict[str, pd.DataFrame] = {}
    for axis, n in (("MU", args.n_mu), ("PU", args.n_pu)):
        job = SYNTH_JOBS[axis]
        pool = job["select"](wide, max_anchors=n, cap=99)
        sample[axis] = pool.head(n)

    # rows[axis] = list of dict(anchor, target, {model: text/just})
    report_rows: dict[str, list[dict]] = {}
    for axis in ("MU", "PU"):
        job = SYNTH_JOBS[axis]
        targets = job["target_grades"]
        rows: list[dict] = []
        for _, anchor in sample[axis].iterrows():
            per_model = {}
            for name, cfg in MODELS.items():
                per_model[name] = await _gen(clients[name], cfg, axis, anchor, targets)
            # align by target grade token
            for tgt in [p2_io.int_to_token(axis, g) for g in targets]:
                rows.append({
                    "anchor": anchor["query_text"],
                    "anchor_grade": p2_io.int_to_token(axis, int(anchor[axis.lower()])),
                    "target": tgt,
                    "models": {
                        name: next((v for v in per_model[name] if v["target"] == tgt), None)
                        for name in MODELS
                    },
                })
        report_rows[axis] = rows

    _write_html(report_rows)
    return 0


def _cell(v: dict | None) -> str:
    if not v:
        return "<i style='color:#999'>(no variant)</i>"
    return (f"<div>{html.escape(v['text'])}</div>"
            f"<div style='color:#777;font-size:0.85em;margin-top:3px'>{html.escape(v['just'])}</div>")


def _write_html(report_rows: dict[str, list[dict]]) -> None:
    model_names = list(MODELS.keys())
    parts = [
        "<html><head><meta charset='utf-8'><title>Generator comparison</title><style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:1300px}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "td,th{border:1px solid #ccc;padding:7px 10px;vertical-align:top;text-align:left}"
        "th{background:#f0f0f0}td.a{background:#fafafa;max-width:240px}"
        "td.m{max-width:430px}.tg{font-weight:bold;color:#b30}</style></head><body>",
        "<h1>Synth-variant generator comparison</h1>",
        "<p>Same anchors + same <code>synth_variant_v2</code> prompt through both "
        "candidate training-side generators. Judge: topic preserved? register "
        "(short lowercase patient query) preserved? grade correct per rubric?</p>",
    ]
    for axis in ("MU", "PU"):
        parts.append(f"<h2>{axis} — {p2_io.AXIS_LONG_NAMES_EN[axis]}</h2>")
        parts.append("<table><tr><th>anchor</th><th>target</th>"
                     + "".join(f"<th>{html.escape(m)}</th>" for m in model_names) + "</tr>")
        for r in report_rows[axis]:
            parts.append(
                f"<tr><td class='a'>{html.escape(r['anchor'])} "
                f"<i>({r['anchor_grade']})</i></td>"
                f"<td class='tg'>{r['target']}</td>"
                + "".join(f"<td class='m'>{_cell(r['models'][m])}</td>" for m in model_names)
                + "</tr>"
            )
        parts.append("</table>")
    parts.append("</body></html>")
    out = p2_io.P2_CACHE / "generator_compare_report.html"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"[report] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mu", type=int, default=8)
    ap.add_argument("--n-pu", type=int, default=6)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
