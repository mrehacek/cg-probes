"""Paper-ready figure primitives for the CG-Probes benchmark.

Every figure is saved to `benchmark/cache/figures/{name}.{svg,pdf,png}` (vector
for LaTeX, PNG for the HTML report) AND returned as an inline base64 PNG so the
report stays a single self-contained file. Style: serif, colorblind-safe
(Okabe-Ito) palette, restrained — built to drop straight into a CIKM submission.

Pure matplotlib (Agg). `emit(fig, name)` is the single save/encode entry point.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from contrastive.p2_io import REPO  # noqa: E402

FIG_DIR = REPO / "benchmark" / "cache" / "figures"

# Okabe-Ito colorblind-safe palette
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7f7f7f"}
GRADE_COLORS = {0: CB["sky"], 1: CB["yellow"], 2: CB["red"]}
GRADE_LABELS = {0: "grade 0", 1: "grade 1", 2: "grade 2"}
AXIS_COLORS = {"MU": CB["blue"], "PU": CB["orange"], "ET": CB["green"]}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})


def emit(fig, name: str, crop: bool = True) -> str:
    """Save {name}.svg/.pdf/.png to FIG_DIR and return an inline base64 PNG URI.
    crop=True uses bbox='tight' (rcParam); crop=False keeps the exact figsize so
    LaTeX \\includegraphics[width=\\columnwidth] renders 1:1 (no font magnification)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    extra = {} if crop else {"bbox_inches": None}
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", **extra)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, **extra)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --- F1: projection separability (KDE-ish histogram by grade) ----------------

def projection_hist(scores, y, t1, t2, title, name) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    lo, hi = float(np.min(scores)), float(np.max(scores))
    bins = np.linspace(lo - 1e-3, hi + 1e-3, 32)
    for g in (0, 1, 2):
        sel = scores[y == g]
        if len(sel):
            ax.hist(sel, bins=bins, density=True, alpha=0.55, color=GRADE_COLORS[g],
                    label=f"{GRADE_LABELS[g]} (n={len(sel)})")
    for t in (t1, t2):
        ax.axvline(t, color="#222", ls="--", lw=1)
    ax.set_xlabel("projection onto probe direction"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend()
    return emit(fig, name)


def grade_box(scores, y, title, name) -> str:
    """Projection by true grade — the ordinality (monotone ladder) view."""
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    data = [scores[y == g] for g in (0, 1, 2)]
    bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.6, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], (0, 1, 2)):
        patch.set_facecolor(GRADE_COLORS[g]); patch.set_alpha(0.6)
    rng = np.random.RandomState(0)
    for g in (0, 1, 2):
        s = scores[y == g]
        if len(s):
            ax.scatter(np.full(len(s), g) + rng.uniform(-0.11, 0.11, len(s)), s,
                       s=5, color=GRADE_COLORS[g], alpha=0.3, zorder=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["g0", "g1", "g2"])
    ax.set_ylabel("projection"); ax.set_title(title); ax.grid(axis="x", alpha=0)
    return emit(fig, name)


def confusion(y_true, y_pred, title, name) -> str:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(3.1, 3.0))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(["p0", "p1", "p2"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["t0", "t1", "t2"])
    ax.set_title(title); ax.grid(False)
    return emit(fig, name)


# --- F2: probe-vs-LLM grouped dot/forest plot --------------------------------

def probe_vs_llm(data: dict, metric_label: str, name: str,
                 ceiling: dict | None = None, title: str | None = None,
                 ceiling_raw: dict | None = None) -> str:
    """data = {axis: {method: (value, lo, hi or None)}}. Methods share colors.
    Optional `ceiling` = {axis: value} draws the human-agreement ceiling per row;
    `ceiling_raw` = {axis: value} draws a faint pre-correction tick (for disclosure)."""
    axes_order = list(data)
    methods = list(next(iter(data.values())))
    colors = [CB["green"], CB["blue"], CB["orange"], CB["grey"]]
    cmap = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
    # Short legend labels — long names (gpt-oss-safeguard-20b) make the legend wider
    # than the axes, and bbox="tight" then expands the canvas, leaving the plot narrow.
    short = {"gpt-oss-safeguard-20b": "safeguard-20b", "gpt-oss-120b": "oss-120b",
             "gpt-5.4": "gpt-5.4", "probe": "probe"}
    # Fixed canvas at ACM column width (3.33in); saved un-cropped (emit crop=False)
    # so width=\columnwidth is exactly 1:1 and fonts render at their true pt (= body).
    fig, ax = plt.subplots(figsize=(3.33, 2.05))
    n_m = len(methods)
    span = 0.88  # rows fill most of their band -> small gaps between MU/PU/TS
    for ai, axis in enumerate(axes_order):
        for mi, m in enumerate(methods):
            v = data[axis].get(m)
            if v is None:
                continue
            val, lo, hi = (v if isinstance(v, (list, tuple)) else (v, None, None))
            yoff = ai + (mi - (n_m - 1) / 2) * span / n_m
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [yoff, yoff], color=cmap[m], lw=1.3, alpha=0.85,
                        solid_capstyle="round", zorder=2)
            ax.scatter(val, yoff, color=cmap[m], s=30, zorder=3, edgecolors="white",
                       linewidths=0.5, label=short.get(m, m) if ai == 0 else None)
        if ceiling_raw and ceiling_raw.get(axis) is not None:
            rx = ceiling_raw[axis]
            ax.plot([rx, rx], [ai - 0.49, ai + 0.49], color=CB["grey"], ls=":", lw=1.2,
                    alpha=0.9, zorder=3, label="raw" if ai == 0 else None)
        if ceiling and ceiling.get(axis) is not None:
            cx = ceiling[axis]
            ax.plot([cx, cx], [ai - 0.49, ai + 0.49], color="#222", ls="--", lw=1.3,
                    zorder=4, label="ceiling" if ai == 0 else None)
    ax.set_yticks(range(len(axes_order)))
    # paper-facing axis names: code symbol ET -> "TS" (Topic Sensitivity)
    ax.set_yticklabels([{"ET": "TS"}.get(a, a) for a in axes_order], fontsize=7.5)
    ax.set_ylim(len(axes_order) - 0.42, -0.58)  # inverted + tight crop -> less whitespace
    ax.set_xlabel(metric_label, fontsize=7, labelpad=2)
    ax.tick_params(axis="x", labelsize=6.5, width=0.8, length=3, pad=2)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=8)
    ncol = min(3, len(methods) + (1 if ceiling else 0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=ncol, fontsize=6,
              handlelength=1.1, handletextpad=0.3, columnspacing=0.7, borderaxespad=0.1)
    ax.grid(axis="y", alpha=0)
    ax.grid(axis="x", alpha=0.3, linewidth=0.7)
    ax.margins(x=0.02)
    # fixed-size canvas (no tight crop) -> 1:1 at \columnwidth, fonts at true pt
    fig.subplots_adjust(left=0.13, right=0.975, top=0.96, bottom=0.29)
    return emit(fig, name, crop=False)


# --- F3: cost-performance Pareto ---------------------------------------------

def pareto(points: list[dict], metric_label: str, name: str) -> str:
    """points: [{label, cost_ms, score, kind, color?}]. Marker by kind (probe=circle,
    llm/frontier=square so the LLMs share a shape); per-point `color` overrides."""
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    marker = {"probe": "o", "llm": "s", "frontier": "s"}
    dflt = {"probe": CB["green"], "llm": CB["blue"], "frontier": CB["purple"]}
    for p in points:
        mk = marker.get(p["kind"], "o")
        c = p.get("color") or dflt.get(p["kind"], CB["grey"])
        ax.scatter(p["cost_ms"], p["score"], color=c, marker=mk, s=72, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.annotate(p["label"], (p["cost_ms"], p["score"]), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("inference cost — ms / query (log)"); ax.set_ylabel(metric_label)
    ax.set_title("Cost vs performance (top-left is best)")
    return emit(fig, name)


# --- F4 / F6: heatmaps --------------------------------------------------------

def heatmap(M, row_labels, col_labels, title, name, cmap="RdBu_r",
            vmin=-1.0, vmax=1.0, fmt="{:+.2f}") -> str:
    M = np.asarray(M, dtype=float)
    fig, ax = plt.subplots(figsize=(0.62 * len(col_labels) + 2.0,
                                    0.55 * len(row_labels) + 1.7))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8.5,
                        color="white" if abs((M[i, j] - (vmin + vmax) / 2)) > (vmax - vmin) * 0.32
                        else "black")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=40, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title(title); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return emit(fig, name)


# --- F5: topic-orthogonality strip -------------------------------------------

def orthogonality_strip(by_axis: dict, name: str) -> str:
    """by_axis = {axis: [cosines]}. Bands at 0.4 / 0.6."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    rng = np.random.RandomState(0)
    for ai, (axis, cosines) in enumerate(by_axis.items()):
        ac = np.abs(np.asarray(cosines))
        ax.scatter(ac, np.full(len(ac), ai) + rng.uniform(-0.16, 0.16, len(ac)),
                   s=10, alpha=0.4, color=AXIS_COLORS.get(axis, CB["grey"]))
        ax.scatter(ac.mean(), ai, marker="|", s=600, color="black", zorder=4)
    top = len(by_axis) - 0.4
    ax.axvline(0.4, color=CB["green"], ls="--", lw=1)
    ax.axvline(0.6, color=CB["red"], ls="--", lw=1)
    ax.text(0.4, top, "0.4 near-orth", color=CB["green"], fontsize=7.5, ha="center", va="bottom")
    ax.text(0.6, top, "0.6 entangled", color=CB["red"], fontsize=7.5, ha="center", va="bottom")
    ax.set_yticks(range(len(by_axis))); ax.set_yticklabels(list(by_axis))
    ax.set_ylim(-0.6, len(by_axis) - 0.1)
    ax.set_xlabel("|cos(axis direction, topic centroid)|  (black bar = mean)")
    ax.set_xlim(0, 1); ax.set_title("Topic-orthogonality (grade-0 centroids)")
    ax.grid(axis="y", alpha=0)
    return emit(fig, name)


# --- F7 / F8: grouped bars ----------------------------------------------------

def grouped_bar(groups, series, ylabel, title, name, colors=None,
                hline=None, hline_label=None, markers=None, marker_label="human ceiling") -> str:
    """groups = x categories; series = {name: [vals aligned to groups]}.
    `markers` = [val per group] draws a black ceiling cap spanning each group."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    keys = list(series)
    palette = colors or [CB["green"], CB["blue"], CB["orange"], CB["grey"]]
    n = len(keys); width = 0.8 / n
    x = np.arange(len(groups))
    for i, k in enumerate(keys):
        ax.bar(x + (i - (n - 1) / 2) * width, series[k], width, label=k,
               color=palette[i % len(palette)], alpha=0.9)
    if hline is not None:
        ax.axhline(hline, color="#444", ls=":", lw=1.2,
                   label=hline_label or f"chance ({hline:.2f})")
    if markers is not None:
        for xi, mv in zip(x, markers):
            if mv is not None:
                ax.plot([xi - 0.4, xi + 0.4], [mv, mv], color="#222", ls="--", lw=1.6,
                        zorder=5, label=marker_label if xi == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); ax.grid(axis="x", alpha=0)
    return emit(fig, name)


# --- report-only: 2-D entanglement scatter -----------------------------------

def entanglement_scatter(sx, sy, y, ax_x, ax_y, title, name) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for g in (0, 1, 2):
        sel = y == g
        if sel.any():
            ax.scatter(sx[sel], sy[sel], s=10, alpha=0.45, color=GRADE_COLORS[g],
                       label=GRADE_LABELS[g])
    ax.set_xlabel(f"projection on {ax_x}"); ax.set_ylabel(f"projection on {ax_y}")
    ax.set_title(title); ax.legend()
    return emit(fig, name)
