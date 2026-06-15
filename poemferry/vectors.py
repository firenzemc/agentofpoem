import json
from pathlib import Path

import httpx
import numpy as np

from .config import Settings
from .models import Poem

# Composite retrieval key: the English gist carries cross-lingual meaning, themes
# add keywords, title/author add work identity. Embedding both corpus and the
# English query intent in this space keeps matching cross-lingual.
EMBED_DIR = "data"
EMBED_FILE = "embeddings.npz"


def retrieval_key(p: Poem) -> str:
    enr = p.enrichment or {}
    gist = enr.get("gist") or p.full_text[:200]
    themes = ", ".join(enr.get("themes", []))
    parts = [gist]
    if themes:
        parts.append(f"themes: {themes}")
    if p.title or p.author:
        parts.append(f"{p.title or ''} {p.author or ''}".strip())
    return " | ".join(parts)


def embed_texts(settings: Settings, texts: list[str], batch_size: int = 96) -> np.ndarray:
    """Embed texts via the OpenAI-compatible endpoint. Cohere embed v4 requires
    array input and rejects input_type through this gateway, so we send plain
    arrays. Returns an L2-normalized float32 matrix."""
    url = settings.embed_base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {settings.embed_api_key}"}
    vectors: list[list[float]] = []
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            resp = client.post(
                url, headers=headers, json={"model": settings.embed_model, "input": chunk}
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-8, None)


def embed_query(settings: Settings, text: str) -> np.ndarray:
    return embed_texts(settings, [text])[0]


class VectorIndex:
    """In-memory cosine index over normalized poem embeddings (~13k × 1536)."""

    def __init__(self, ids: list[str], matrix: np.ndarray):
        self.ids = ids
        self.matrix = matrix  # (N, dim), L2-normalized

    @classmethod
    def load(cls, path: str = f"{EMBED_DIR}/{EMBED_FILE}") -> "VectorIndex | None":
        p = Path(path)
        if not p.exists():
            return None
        npz = np.load(p, allow_pickle=False)
        ids = json.loads(str(npz["ids"].item()))
        return cls(ids, npz["matrix"])

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        scores = self.matrix @ query_vec  # cosine (both normalized)
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


def save_index(ids: list[str], matrix: np.ndarray, path: str = f"{EMBED_DIR}/{EMBED_FILE}") -> None:
    np.savez(path, ids=np.asarray(json.dumps(ids)), matrix=matrix.astype(np.float32))
