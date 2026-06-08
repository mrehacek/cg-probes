"""Raw-HTML report helpers (pure HTML/CSS, no new deps).

Each pipeline step renders a self-contained .html the user reviews before the
next step runs. These helpers build the common scaffolding — page shell, tables,
the 3x3 axis/grade grid, CSS bars, and a provenance footer — so step scripts only
assemble sections. Mirrors the raw-f-string approach in
golden/llm_annotator/pipeline/01_judge.py.
"""

from __future__ import annotations

import html as _html
from typing import Iterable, Mapping, Sequence

STYLE = """
:root { --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --ok:#1a7f37; --bad:#cf222e; --bg2:#f6f8fa; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; color: var(--fg);
       max-width: 1100px; margin: 2rem auto; padding: 0 1.2rem; line-height: 1.45; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem;
     border-bottom: 2px solid var(--line); padding-bottom: .3rem; }
table { border-collapse: collapse; width: 100%; margin: .6rem 0; font-size: .9rem; }
th, td { border: 1px solid var(--line); padding: .35rem .55rem; text-align: left;
         vertical-align: top; }
th { background: var(--bg2); }
code { background: var(--bg2); padding: .1rem .3rem; border-radius: 4px; font-size: .85em; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: .25rem .8rem; }
dt { color: var(--muted); } dd { margin: 0; }
.ok { color: var(--ok); font-weight: 600; } .bad { color: var(--bad); font-weight: 600; }
.muted { color: var(--muted); }
.bar { background: #2f81f7; height: 1rem; border-radius: 3px; display: inline-block; }
.barwrap { background: var(--bg2); border-radius: 3px; width: 240px; display: inline-block;
           vertical-align: middle; margin-right: .5rem; }
.grid3 td { text-align: center; } .grid3 .rowhdr { text-align: left; font-weight: 600; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line);
         color: var(--muted); font-size: .8rem; }
"""


def esc(s: object) -> str:
    return _html.escape("" if s is None else str(s), quote=False)


def page(title: str, *sections: str) -> str:
    """Wrap section HTML fragments into a full document."""
    body = "\n".join(sections)
    return (
        f'<!doctype html><html lang="cs"><head><meta charset="utf-8">'
        f"<title>{esc(title)}</title><style>{STYLE}</style></head><body>"
        f"<h1>{esc(title)}</h1>\n{body}</body></html>"
    )


def section(heading: str, *html_fragments: str) -> str:
    return f"<h2>{esc(heading)}</h2>\n" + "\n".join(html_fragments)


def kv(meta: Mapping[str, object]) -> str:
    """A <dl> of key/value metadata."""
    items = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in meta.items())
    return f"<dl>{items}</dl>"


def table(rows: Sequence[Mapping[str, object]], columns: Sequence[str] | None = None) -> str:
    """An HTML table from a list of record dicts."""
    rows = list(rows)
    if not rows:
        return '<p class="muted">— none —</p>'
    cols = list(columns) if columns else list(rows[0].keys())
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def bar_row(label: str, count: int, total: int) -> str:
    """One labeled CSS bar (count / total) — for confidence/coverage histograms."""
    pct = 0 if total <= 0 else round(100 * count / total)
    return (
        f'<div><span class="barwrap"><span class="bar" style="width:{pct}%"></span></span>'
        f"{esc(label)} — {count} ({pct}%)</div>"
    )


def grade_grid(per_axis_counts: Mapping[str, Mapping[str, int]]) -> str:
    """3x3 axis x grade count table.

    `per_axis_counts` maps axis -> {grade_token -> count}, e.g.
    {"MU": {"MU0": 120, "MU1": 95, "MU2": 41}, ...}. Renders one row per axis
    with the three grade columns (0/1/2).
    """
    axes = list(per_axis_counts.keys())
    head = "<tr><th>Axis</th><th>grade 0</th><th>grade 1</th><th>grade 2</th><th>total</th></tr>"
    body = ""
    for ax in axes:
        counts = per_axis_counts[ax]
        vals = [counts.get(f"{ax}{i}", 0) for i in range(3)]
        body += (
            f'<tr><td class="rowhdr">{esc(ax)}</td>'
            + "".join(f"<td>{v}</td>" for v in vals)
            + f"<td>{sum(vals)}</td></tr>"
        )
    return f'<table class="grid3"><thead>{head}</thead><tbody>{body}</tbody></table>'


def status_badge(ok: bool, label: str) -> str:
    cls = "ok" if ok else "bad"
    mark = "✓" if ok else "✗"
    return f'<span class="{cls}">{mark} {esc(label)}</span>'


def provenance(meta: Mapping[str, object]) -> str:
    items = "".join(f"<dt>{esc(k)}</dt><dd><code>{esc(v)}</code></dd>" for k, v in meta.items())
    return f"<footer><strong>Provenance</strong><dl>{items}</dl></footer>"
