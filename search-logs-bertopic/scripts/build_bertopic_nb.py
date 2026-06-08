"""Generate the rewired bertopic_query_clusters.ipynb (qwen3-4b, 2560-d).

Run this when the notebook schema needs regenerating. Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "bertopic_query_clusters.ipynb"


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


CELLS = [
    md(
        "# BERTopic — query clusters over qwen3-embedding-4b (2560-d)\n\n"
        "Stage P1-04 of the CIKM pipeline. Reads the filtered queries from P1-02 and the qwen3 "
        "embeddings from P1-03, clusters them with UMAP + HDBSCAN (leaf, `min_cluster_size=10`), "
        "reassigns outliers, and writes the cluster artefacts that P1-05 and P2 consume.\n\n"
        "**Inputs**\n"
        "- `cache/preprocess/embed_input.parquet` (`query_key`, `query`, `clicks`, …)\n"
        "- `cache/embeddings/qwen3-embedding-4b.parquet` (`query_key`, `query`, `embedding`)\n\n"
        "**Outputs** (under `cache/clusters/`)\n"
        "- `bertopic_assignments_qwen3_minc10_leaf.csv`\n"
        "- `cluster_overview_qwen3_minc10_leaf.jsonl`\n"
        "- `centroids_qwen3.parquet`\n"
        "- `bertopic_umap_qwen3.npy` (2D for the paper figure)"
    ),
    md("## 1. Setup"),
    code(
        "import json, time, warnings\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "warnings.filterwarnings('ignore')\n\n"
        "ROOT = Path('.').resolve()\n"
        "# Make repo root the working dir if the notebook was launched elsewhere.\n"
        "if ROOT.name != 'search-logs-bertopic':\n"
        "    cand = ROOT / 'search-logs-bertopic'\n"
        "    if cand.exists():\n"
        "        ROOT = cand\n"
        "print('ROOT =', ROOT)\n\n"
        "PREPROCESS_PARQUET = ROOT / 'cache' / 'preprocess' / 'embed_input.parquet'\n"
        "EMBED_PARQUET      = ROOT / 'cache' / 'embeddings' / 'qwen3-embedding-4b.parquet'\n"
        "OUT_CLU            = ROOT / 'cache' / 'clusters'\n"
        "OUT_CLU.mkdir(parents=True, exist_ok=True)\n\n"
        "MIN_CLUSTER_SIZE   = 10\n"
        "CLUSTER_SELECTION  = 'leaf'\n"
        "STEM = f'qwen3_minc{MIN_CLUSTER_SIZE}_{CLUSTER_SELECTION}'\n"
        "ASSIGNMENTS_CSV    = OUT_CLU / f'bertopic_assignments_{STEM}.csv'\n"
        "OVERVIEW_JSONL     = OUT_CLU / f'cluster_overview_{STEM}.jsonl'\n"
        "CENTROIDS_PARQUET  = OUT_CLU / 'centroids_qwen3.parquet'\n"
        "UMAP_2D_NPY        = OUT_CLU / 'bertopic_umap_qwen3.npy'"
    ),
    md(
        "## 2. Load filtered queries + qwen3 embeddings\n\n"
        "Inner-join on `query_key`. Embedding cache should be a strict superset "
        "of the preprocess output; if anything is missing, fail loudly."
    ),
    code(
        "preprocess = pd.read_parquet(PREPROCESS_PARQUET)\n"
        "embs       = pd.read_parquet(EMBED_PARQUET)\n"
        "print(f'preprocess rows: {len(preprocess):,}    embedding rows: {len(embs):,}')\n\n"
        "df = preprocess.merge(embs[['query_key', 'embedding']], on='query_key', how='left')\n"
        "missing = df['embedding'].isna().sum()\n"
        "assert missing == 0, f'{missing} queries are missing embeddings — rerun embed_qwen3.py'\n\n"
        "vectors = np.array(df['embedding'].tolist(), dtype=np.float32)\n"
        "queries = df['query'].astype(str).tolist()\n"
        "print(f'Loaded {len(df):,} queries × {vectors.shape[1]} dims')\n"
        "norms = np.linalg.norm(vectors, axis=1)\n"
        "print(f'L2 norms: min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f}')"
    ),
    md(
        "## 3. BERTopic — UMAP → HDBSCAN(leaf) → c-TF-IDF\n\n"
        "Mirrors the tuning from the old OpenAI-3072 notebook: leaf HDBSCAN with "
        "`min_cluster_size=10`, `min_samples=5`. UMAP cosine to 5-D, then HDBSCAN "
        "euclidean on the UMAP output."
    ),
    code(
        "from bertopic import BERTopic\n"
        "from hdbscan import HDBSCAN\n"
        "from umap import UMAP\n"
        "from sklearn.feature_extraction.text import CountVectorizer\n\n"
        "from czech_stopwords import czech_stopwords\n\n"
        "stopwords = czech_stopwords()\n"
        "print(f'Czech stopwords: {len(stopwords)}')\n\n"
        "vectorizer = CountVectorizer(\n"
        "    stop_words=stopwords,\n"
        "    ngram_range=(1, 2),\n"
        "    min_df=2,\n"
        ")\n\n"
        "umap_model = UMAP(\n"
        "    n_neighbors=15, n_components=5, min_dist=0.0,\n"
        "    metric='cosine', random_state=42,\n"
        ")\n"
        "hdbscan_model = HDBSCAN(\n"
        "    min_cluster_size=MIN_CLUSTER_SIZE,\n"
        "    min_samples=5,\n"
        "    cluster_selection_method=CLUSTER_SELECTION,\n"
        "    metric='euclidean',\n"
        "    prediction_data=True,\n"
        ")\n\n"
        "topic_model = BERTopic(\n"
        "    embedding_model=None,   # pass embeddings explicitly\n"
        "    umap_model=umap_model,\n"
        "    hdbscan_model=hdbscan_model,\n"
        "    vectorizer_model=vectorizer,\n"
        "    calculate_probabilities=False,\n"
        "    verbose=True,\n"
        ")\n\n"
        "t0 = time.time()\n"
        "topics, probs = topic_model.fit_transform(queries, embeddings=vectors)\n"
        "print(f'BERTopic.fit_transform: {time.time() - t0:.1f}s')\n"
        "print(f'Initial clusters (incl. -1): {len(set(topics))}')\n"
        "print(f'Initial outliers (-1): {(np.array(topics) == -1).sum():,}')"
    ),
    md(
        "## 4. Reduce outliers — embedding-space nearest-centroid\n\n"
        "Strategy `embeddings`: assign each -1 to the cluster whose centroid in the "
        "embedding space is closest in cosine. Works well here because qwen3 vectors "
        "are L2-normalized."
    ),
    code(
        "new_topics = topic_model.reduce_outliers(\n"
        "    queries, topics, strategy='embeddings', embeddings=vectors,\n"
        ")\n"
        "topic_model.update_topics(queries, topics=new_topics, vectorizer_model=vectorizer)\n\n"
        "df['cluster_id'] = new_topics\n"
        "df['topic_prob'] = None  # not computed (calculate_probabilities=False for speed)\n\n"
        "n_remaining_outliers = (df['cluster_id'] == -1).sum()\n"
        "print(f'Remaining outliers after reassignment: {n_remaining_outliers:,}')\n"
        "print(f'Final cluster count: {df.cluster_id.nunique() - (1 if n_remaining_outliers > 0 else 0)}')"
    ),
    md(
        "## 5. Centroids + outputs\n\n"
        "Centroids are L2-renormalized means of member vectors (mean of unit vectors "
        "isn't unit-norm). Required by P2 (E_Topic axis) and P4 (cluster-centroid feature)."
    ),
    code(
        "centroids: dict[int, np.ndarray] = {}\n"
        "for cid, sub in df.groupby('cluster_id'):\n"
        "    if cid == -1:\n"
        "        continue\n"
        "    mean = np.array(sub['embedding'].tolist(), dtype=np.float32).mean(axis=0)\n"
        "    centroids[int(cid)] = (mean / np.linalg.norm(mean)).astype(np.float32)\n"
        "print(f'Centroids written: {len(centroids)}')\n\n"
        "# Top words + exemplars per cluster\n"
        "top_words = {cid: [w for w, _ in topic_model.get_topic(cid)[:20]] for cid in centroids}\n\n"
        "exemplars_per_cluster: dict[int, list[str]] = {}\n"
        "for cid, sub in df.groupby('cluster_id'):\n"
        "    if cid == -1:\n"
        "        continue\n"
        "    sub_vecs = np.array(sub['embedding'].tolist(), dtype=np.float32)\n"
        "    cos = sub_vecs @ centroids[int(cid)]\n"
        "    order = np.argsort(-cos)[:10]\n"
        "    exemplars_per_cluster[int(cid)] = sub.iloc[order]['query_key'].tolist()"
    ),
    code(
        "# Assignments CSV\n"
        "df[['query_key', 'query', 'clicks', 'cluster_id', 'topic_prob']].to_csv(\n"
        "    ASSIGNMENTS_CSV, index=False,\n"
        ")\n"
        "print('wrote', ASSIGNMENTS_CSV)\n\n"
        "# Centroids parquet\n"
        "cent_df = pd.DataFrame({\n"
        "    'cluster_id': list(centroids.keys()),\n"
        "    'centroid':   [v.tolist() for v in centroids.values()],\n"
        "})\n"
        "cent_df.to_parquet(CENTROIDS_PARQUET, index=False)\n"
        "print('wrote', CENTROIDS_PARQUET)\n\n"
        "# Cluster overview JSONL\n"
        "with open(OVERVIEW_JSONL, 'w', encoding='utf-8') as f:\n"
        "    for cid, sub in df.groupby('cluster_id'):\n"
        "        cid_i = int(cid)\n"
        "        if cid_i == -1:\n"
        "            continue\n"
        "        row = {\n"
        "            'cluster_id':          cid_i,\n"
        "            'size':                int(len(sub)),\n"
        "            'total_clicks':        int(sub['clicks'].fillna(0).sum()),\n"
        "            'top_words':           top_words.get(cid_i, []),\n"
        "            'member_query_keys':   sub['query_key'].tolist(),\n"
        "            'exemplar_query_keys': exemplars_per_cluster.get(cid_i, []),\n"
        "        }\n"
        "        f.write(json.dumps(row, ensure_ascii=False) + '\\n')\n"
        "print('wrote', OVERVIEW_JSONL)"
    ),
    md(
        "## 6. 2D UMAP for the paper figure\n\n"
        "Cascade BERTopic's already-fitted 5D UMAP into 2D rather than re-running UMAP "
        "on the raw 2560-d vectors. Much cheaper (~5 s vs ~3 min) and preserves the "
        "cluster topology that HDBSCAN actually saw."
    ),
    code(
        "from umap import UMAP as UMAP2D\n\n"
        "_5d = topic_model.umap_model.embedding_\n"
        "_2d = UMAP2D(n_neighbors=15, n_components=2, min_dist=0.1,\n"
        "             metric='euclidean', random_state=42).fit_transform(_5d)\n"
        "np.save(UMAP_2D_NPY, _2d.astype(np.float32))\n"
        "print('wrote', UMAP_2D_NPY, _2d.shape)"
    ),
    md("## 7. Diagnostics"),
    code(
        "sizes = df['cluster_id'].value_counts().sort_values()\n"
        "print(f'Clusters: {len(sizes)} (incl. -1 if present)')\n"
        "print(f'Size distribution: min={sizes.min()} med={int(sizes.median())} max={sizes.max()}')\n"
        "print(f'Outliers (-1): {(df.cluster_id == -1).sum():,}')\n\n"
        "# Top-30 clusters by size, with 3 highest-clicks queries each\n"
        "rows = []\n"
        "for cid, g in df.groupby('cluster_id'):\n"
        "    cid_i = int(cid)\n"
        "    if cid_i == -1:\n"
        "        continue\n"
        "    sample = ' / '.join(g.nlargest(3, 'clicks')['query'].tolist())\n"
        "    rows.append((cid_i, len(g), int(g['clicks'].fillna(0).sum()), sample))\n"
        "diag = pd.DataFrame(rows, columns=['cluster_id', 'size', 'clicks', 'top3']) \\\n"
        "        .sort_values('size', ascending=False)\n"
        "print(diag.head(30).to_string(index=False))"
    ),
    md(
        "## 8. axes.md anchor audit\n\n"
        "For each canonical anchor phrase from `axes.md` (U/E grades), embed it via the same "
        "endpoint and locate the nearest cluster centroid. The anchor should land in a cluster "
        "whose top words / exemplars match what the anchor is about. If anchors drift to "
        "off-topic clusters, the qwen3 topology has diverged enough that P2 needs re-eliciting.\n\n"
        "Spot-check only — full re-elicitation is a P2 task."
    ),
    code(
        "import os, sys\n"
        "from dotenv import load_dotenv\n"
        "from openai import OpenAI\n\n"
        "load_dotenv(ROOT.parent / '.env')\n"
        "client = OpenAI(api_key=os.environ['EMBED_API_KEY'], base_url=os.environ['EMBED_API_URL'])\n\n"
        "ANCHORS = [\n"
        "    # (axis_grade, phrase)\n"
        "    ('U0', 'kontakt mamograf žlutý kopec'),\n"
        "    ('U0', 'kde je rentgen'),\n"
        "    ('U1', 'jak dlouho trvá biologická léčba'),\n"
        "    ('U1', 'vedlejší účinky chemoterapie u prsu'),\n"
        "    ('U1', 'kdy mi spadnou vlasy'),\n"
        "    ('U2', 'mám už druhý den průjem 6× denně po chemo'),\n"
        "    ('U2', 'od včera mě hodně bolí záda a brní mi noha'),\n"
        "    ('U2', 'mám krev ve stolici'),\n"
        "    ('U3', 'mám 39° a třesu se'),\n"
        "    ('U3', 'nemůžu dýchat'),\n"
        "    ('U3', 'krvácím z rány po operaci'),\n"
        "    ('E0', 'mamograf objednání'),\n"
        "    ('E0', 'co znamená HER2 pozitivní'),\n"
        "    ('E1', 'co když se to vrátí'),\n"
        "    ('E1', 'kolik mám času, doktoři neříkají'),\n"
        "    ('E2', 'jsem zoufalá'),\n"
        "    ('E2', 'mám hrozný strach z recidivy'),\n"
        "    ('E2', 'trápí mě úzkost po každé kontrole'),\n"
        "    ('E3', 'nechci dál žít'),\n"
        "    ('E3', 'bolesti jsou neúnosné, chci to ukončit'),\n"
        "]\n\n"
        "resp = client.embeddings.create(model='qwen3-embedding-4b',\n"
        "                                 input=[p for _, p in ANCHORS])\n"
        "anchor_vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)\n\n"
        "cids = np.array(list(centroids.keys()))\n"
        "C    = np.stack([centroids[int(c)] for c in cids], axis=0)\n"
        "sims = anchor_vecs @ C.T  # both are L2-norm; this is cosine\n"
        "best = sims.argmax(axis=1)\n"
        "best_cos = sims.max(axis=1)\n\n"
        "rows = []\n"
        "for (grade, phrase), bi, cos in zip(ANCHORS, best, best_cos):\n"
        "    cid = int(cids[bi])\n"
        "    rows.append({\n"
        "        'grade': grade,\n"
        "        'anchor': phrase,\n"
        "        'cluster_id': cid,\n"
        "        'cos': round(float(cos), 3),\n"
        "        'top_words': ' '.join(top_words[cid][:8]),\n"
        "        'exemplars': ' / '.join(\n"
        "            df.loc[df.cluster_id == cid].nlargest(3, 'clicks')['query'].tolist()\n"
        "        ),\n"
        "    })\n"
        "audit = pd.DataFrame(rows)\n"
        "audit.to_csv(OUT_CLU / f'anchor_audit_{STEM}.csv', index=False)\n"
        "pd.set_option('display.max_colwidth', 90)\n"
        "print(audit.to_string(index=False))"
    ),
    md(
        "## 9. Run summary\n\n"
        "Counts + paths for downstream stages."
    ),
    code(
        "summary = {\n"
        "    'rows':            int(len(df)),\n"
        "    'n_clusters':      int((df.cluster_id != -1).nunique()),\n"
        "    'outliers_left':   int((df.cluster_id == -1).sum()),\n"
        "    'median_size':     int(df.cluster_id.value_counts().median()),\n"
        "    'min_size':        int(df.cluster_id.value_counts().min()),\n"
        "    'max_size':        int(df.cluster_id.value_counts().max()),\n"
        "    'assignments_csv': str(ASSIGNMENTS_CSV),\n"
        "    'overview_jsonl':  str(OVERVIEW_JSONL),\n"
        "    'centroids_parquet': str(CENTROIDS_PARQUET),\n"
        "    'umap_2d_npy':     str(UMAP_2D_NPY),\n"
        "    'stem':            STEM,\n"
        "}\n"
        "(OUT_CLU / f'run_summary_{STEM}.json').write_text(\n"
        "    json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8',\n"
        ")\n"
        "print(json.dumps(summary, indent=2, ensure_ascii=False))"
    ),
]


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, OUT_NB)
    print(f"wrote {OUT_NB}  ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
