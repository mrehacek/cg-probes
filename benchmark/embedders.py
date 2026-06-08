"""P4 — frozen-embedder registry with a uniform `embed(texts, instruction)`.

Four benchmark embedders, all frozen, all returning L2-normalized vectors:

| name          | dim  | backend                              | instruction |
|---------------|------|--------------------------------------|-------------|
| qwen8b        | 4096 | HF Inference Endpoint (TEI, OpenAI)  | yes         |
| harrier27b    | 5376 | HF Inference Endpoint (TEI, OpenAI)  | yes         |
| openai3large  | 3072 | OpenAI API                           | no (raw)    |
| gemini        | 3072 | Gemini OpenAI-compat endpoint        | yes         |

Instruction-following embedders take the per-query template
  `Instruct: {task}\nQuery: {text}`  (TEI has no prompt param; prepend client-side).
`openai3large` is not instruction-tuned, so it embeds raw text (the generic cell).

Env vars (in cikm-ds/.env):
  HF_TOKEN, HF_QWEN_ENDPOINT_URL, HF_HARRIER_ENDPOINT_URL,
  OPENAI_API_KEY, GEMINI_API_KEY
Gemini model is configurable (GEMINI_EMBED_MODEL, default gemini-embedding-001;
set to gemini-embedding-2 for the newer multimodal model — note its vector space
is incompatible with 001).

Smoke any embedder once its endpoint is live:
  python -m benchmark.embedders --smoke qwen8b --instr-mode per_axis --axis MU
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import openai
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from contrastive.p2_io import REPO

load_dotenv(REPO / ".env")

# --- instruction strings -----------------------------------------------------

# Classification framing (per user's docs/research/Prompt for harrier-oss.md):
# "Classify ... the given oncology patient question ...". Instruction in English,
# query in Czech — the standard convention for these decoder embedders. Applied
# as `Instruct: {task}\nQuery: {text}`; TEI handles EOS + last-token pooling.
AXIS_INSTRUCTION = {
    "MU": "Classify the medical urgency of the given Czech oncology patient query: "
          "the somatic emergency level, from no urgency to requiring immediate "
          "emergency medical services, ignoring psychological aspects.",
    "PU": "Classify the psychological urgency of the given Czech oncology patient query: "
          "the level of emotional distress, existential concern, or suicidal intent "
          "expressed, ignoring the somatic medical content.",
    "ET": "Classify the intrinsic emotional load of the given Czech oncology patient "
          "query's topic, independent of the tone in which it is phrased.",
}
GENERIC_INSTRUCTION = (
    "Represent the given Czech oncology patient query for classification along "
    "clinical-safety dimensions: medical urgency, psychological urgency, and emotional load."
)

# Instruction-ablation variant: the SAME per-axis instructions with the negation
# clauses removed ("ignoring …", "independent of tone"). Embedders handle negation
# poorly; this isolates whether the per-axis hurt comes from the negation.
AXIS_INSTRUCTION_POS = {
    "MU": "Classify the medical urgency of the given Czech oncology patient query: "
          "the somatic emergency level, from no urgency to requiring immediate "
          "emergency medical services.",
    "PU": "Classify the psychological urgency of the given Czech oncology patient query: "
          "the level of emotional distress, existential concern, or suicidal intent expressed.",
    "ET": "Classify the emotional load of the topic of the given Czech oncology patient query.",
}

# axis-specific modes (one embedding per axis) vs shared modes (one per cell).
PER_AXIS_MODES = {"per_axis", "per_axis_pos"}


def instruction_for(mode: str, axis: str | None) -> str | None:
    """mode in {'none','generic','per_axis','per_axis_pos'}."""
    if mode == "none":
        return None
    if mode == "generic":
        return GENERIC_INSTRUCTION
    if mode == "per_axis":
        return AXIS_INSTRUCTION[_need_axis(axis)]
    if mode == "per_axis_pos":
        return AXIS_INSTRUCTION_POS[_need_axis(axis)]
    raise ValueError(f"unknown instr mode {mode!r}")


def _need_axis(axis: str | None) -> str:
    if axis not in AXIS_INSTRUCTION:
        raise ValueError(f"per-axis mode needs a valid axis, got {axis!r}")
    return axis


_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError,
              openai.APITimeoutError, openai.InternalServerError)


def _fmt(text: str, task: str | None) -> str:
    return text if task is None else f"Instruct: {task}\nQuery: {text}"


def _l2norm(vecs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(n, 1e-12, None)


# --- embedder base + backends ------------------------------------------------

@dataclass
class Embedder:
    name: str
    dim: int | None  # None => detect at runtime (e.g. gemini-embedding-2)
    supports_instruction: bool
    _client: OpenAI
    _model: str  # the `model` field the backend expects
    max_batch: int = 256  # backend cap on inputs/request (gemini compat = 100)

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=1, min=2, max=90),
           retry=retry_if_exception_type(_RETRYABLE), reraise=True)
    def _call(self, inputs: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=inputs)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    def embed_request(self, texts: list[str], instruction: str | None) -> np.ndarray:
        """One backend request for a single chunk (len <= max_batch). Formats,
        calls, detects dim, L2-normalizes. Thread-safe (httpx client) so callers
        may run several of these concurrently to exploit the endpoint's
        max-concurrent-requests."""
        task = instruction if self.supports_instruction else None
        vecs = self._call([_fmt(t, task) for t in texts])
        if self.dim is None:
            self.dim = int(vecs.shape[1])
        elif vecs.shape[1] != self.dim:
            raise ValueError(f"{self.name}: expected dim {self.dim}, got {vecs.shape[1]}")
        return _l2norm(vecs)

    def embed(self, texts: list[str], instruction: str | None,
              batch_size: int = 64) -> np.ndarray:
        """Embed texts (applies the Instruct: template iff supports_instruction).
        Returns an (N, dim) L2-normalized float32 array."""
        task = instruction if self.supports_instruction else None
        inputs = [_fmt(t, task) for t in texts]
        bs = min(batch_size, self.max_batch)
        out: list[np.ndarray] = []
        for i in range(0, len(inputs), bs):
            out.append(self._call(inputs[i:i + bs]))
        if not out:
            return np.zeros((0, self.dim or 0), np.float32)
        vecs = np.concatenate(out, axis=0)
        if self.dim is None:
            self.dim = int(vecs.shape[1])  # detect (gemini-embedding-2)
        elif vecs.shape[1] != self.dim:
            raise ValueError(f"{self.name}: expected dim {self.dim}, got {vecs.shape[1]}")
        return _l2norm(vecs)


def _hfie_client(url_var: str) -> OpenAI:
    key, base = os.environ.get("HF_TOKEN"), os.environ.get(url_var)
    if not key or not base:
        raise RuntimeError(f"HF_TOKEN and {url_var} must be set in .env")
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    return OpenAI(api_key=key, base_url=base)


def _served_model(client: OpenAI, fallback: str) -> str:
    """Auto-detect the served model id from /v1/models. TEI ignores the request
    `model` field, but vLLM requires it to match the served name — so we read it
    from the endpoint and stay correct for both engines."""
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return fallback


def make_embedder(name: str) -> Embedder:
    if name == "qwen8b":
        # TEI default --max-client-batch-size is 32.
        c = _hfie_client("HF_QWEN_ENDPOINT_URL")
        return Embedder("qwen8b", 4096, True, c,
                        _served_model(c, "text-embeddings-inference"), max_batch=32)
    if name == "harrier27b":
        # Served via vLLM (last-token pooling; TEI's Gemma3 backend is mean-only).
        # dim detected at runtime; vLLM needs the served model id (auto-detected).
        c = _hfie_client("HF_HARRIER_ENDPOINT_URL")
        return Embedder("harrier27b", None, True, c,
                        _served_model(c, "harrier"), max_batch=32)
    if name == "openai3large":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY must be set")
        return Embedder("openai3large", 3072, False, OpenAI(api_key=key),
                        "text-embedding-3-large")
    if name == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY must be set")
        # Newer multimodal model per user (2026-06-06). dim detected at runtime.
        model = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2")
        client = OpenAI(api_key=key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        # Gemini OpenAI-compat embeddings caps at 100 inputs per request.
        return Embedder("gemini", None, True, client, model, max_batch=100)
    raise ValueError(f"unknown embedder {name!r}")


REGISTRY = ["qwen8b", "harrier27b", "openai3large", "gemini"]

# Which instruction modes each embedder runs. openai3large is raw-only (its
# "generic" cell is already instruction-free). Instruction-tuned embedders run the
# full ablation: none (raw) / generic / per_axis (negated) / per_axis_pos (positive).
INSTR_MODES = {
    "qwen8b": ["none", "generic", "per_axis", "per_axis_pos"],
    "harrier27b": ["none", "generic", "per_axis", "per_axis_pos"],
    "gemini": ["none", "generic", "per_axis", "per_axis_pos"],
    "openai3large": ["generic"],  # not instruction-tuned -> single raw embedding
}


# --- smoke -------------------------------------------------------------------

_SMOKE = [
    "nemůžu dýchat", "krvácím z rány po operaci", "mamograf objednání",
    "už nechci žít", "vedlejší účinky tamoxifenu", "kolik mi zbývá času",
]


def _smoke(name: str, mode: str, axis: str | None) -> int:
    emb = make_embedder(name)
    instr = instruction_for(mode, axis) if emb.supports_instruction else None
    print(f"[smoke] {name} dim={emb.dim} instr_supported={emb.supports_instruction} "
          f"mode={mode} instr={instr!r}")
    t0 = time.time()
    v = emb.embed(_SMOKE, instr)
    print(f"[smoke] shape={v.shape} wall={time.time()-t0:.2f}s norms~{np.linalg.norm(v,axis=1)[:3]}")
    # emergency phrase should be closer to another emergency than to navigational
    ce = float(v[1] @ v[0]); cn = float(v[1] @ v[2])
    print(f"[smoke] cos(bleeding, can't-breathe)={ce:.3f}  cos(bleeding, mammogram-booking)={cn:.3f}")
    print("[smoke] OK" if v.shape == (len(_SMOKE), emb.dim) else "[smoke] DIM MISMATCH")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=REGISTRY, required=True)
    ap.add_argument("--instr-mode", choices=["per_axis", "generic", "none"], default="generic")
    ap.add_argument("--axis", choices=["MU", "PU", "ET"], default="MU")
    a = ap.parse_args()
    return _smoke(a.smoke, a.instr_mode, a.axis)


if __name__ == "__main__":
    sys.exit(main())
