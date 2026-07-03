import json
import time
from pathlib import Path

import httpx
import numpy as np

from .config import Settings
from .models import Poem

EMBED_DIR = "data"
DOC_FILE = "embeddings_doc.npz"
# The doc index is stored float16 to halve resident RAM (376k×1536 → 1.16GB vs
# 2.31GB). A float16 @ float32 matmul would transiently upcast the whole matrix
# back to float32, so search casts one row-block at a time to bound the spike.
SEARCH_CHUNK = 50000


def retrieval_key(p: Poem) -> str:
    """Document-mode key: English gist + themes + concrete images + title.

    The images field (added by the enrichment swarm) preserves striking but
    incidental imagery — e.g. 将进酒's Yellow-River opening — that a thematic
    gist alone drops. All English, so matching the English query intent stays
    cross-lingual without a language penalty.
    """
    enr = p.enrichment or {}
    gist = enr.get("gist") or p.full_text[:200]
    parts = [gist]
    if enr.get("themes"):
        parts.append("themes: " + ", ".join(enr["themes"]))
    if enr.get("images"):
        parts.append("images: " + ", ".join(enr["images"]))
    if p.title or p.author:
        parts.append(f"{p.title or ''} {p.author or ''}".strip())
    return " | ".join(parts)


def embed_texts(settings: Settings, texts: list[str], batch_size: int = 96) -> np.ndarray:
    """Embed via OpenAI-compatible endpoint. Cohere embed v4 needs array input
    and rejects input_type through this gateway. Returns L2-normalized float32."""
    url = settings.embed_base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {settings.embed_api_key}"}
    vectors: list[list[float]] = []
    with httpx.Client(timeout=180) as client:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for attempt in range(4):
                resp = client.post(
                    url, headers=headers, json={"model": settings.embed_model, "input": chunk}
                )
                if resp.status_code == 429 and attempt < 3:
                    time.sleep(2 * (attempt + 1))  # backoff on rate limit
                    continue
                resp.raise_for_status()
                break
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-8, None)


def embed_query(settings: Settings, text: str) -> np.ndarray:
    return embed_texts(settings, [text])[0]


class VectorIndex:
    """Cosine index over normalized embeddings. ids may repeat (line index maps
    each row to its parent poem); search_unique dedupes to best score per id."""

    def __init__(self, ids: list[str], matrix: np.ndarray):
        self.ids = ids
        self.matrix = matrix

    @classmethod
    def load(cls, filename: str) -> "VectorIndex | None":
        p = Path(EMBED_DIR) / filename
        if not p.exists():
            return None
        npz = np.load(p, allow_pickle=False)
        return cls(json.loads(str(npz["ids"].item())), npz["matrix"])

    def search_unique(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        scores = self._scores(query_vec)
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for i in order:
            pid = self.ids[i]
            if pid in seen:
                continue
            seen.add(pid)
            out.append((pid, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def _scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine scores over the whole index. If the matrix is float16, compute
        block-by-block (casting each block to float32) so the matmul never upcasts
        the entire matrix at once."""
        q = query_vec.astype(np.float32)
        if self.matrix.dtype != np.float16:
            return self.matrix @ q
        n = self.matrix.shape[0]
        scores = np.empty(n, dtype=np.float32)
        for i in range(0, n, SEARCH_CHUNK):
            block = self.matrix[i : i + SEARCH_CHUNK]
            scores[i : i + SEARCH_CHUNK] = block.astype(np.float32) @ q
        return scores


def save_index(ids: list[str], matrix: np.ndarray, filename: str) -> None:
    # float16 halves the on-disk and resident size; search casts per-block.
    np.savez(Path(EMBED_DIR) / filename,
             ids=np.asarray(json.dumps(ids)), matrix=matrix.astype(np.float16))
