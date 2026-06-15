import json
import time
from pathlib import Path

import httpx
import numpy as np

from .config import Settings
from .models import Poem

EMBED_DIR = "data"
DOC_FILE = "embeddings_doc.npz"
LINE_FILE = "embeddings_line.npz"
MAX_LINES_PER_POEM = 12


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


def poem_lines(p: Poem) -> list[str]:
    """Up to MAX_LINES_PER_POEM representative lines (evenly sampled for long
    poems) for the line-level index — minimal-compression retrieval."""
    lines = [ln.strip() for ln in p.full_text.splitlines() if len(ln.strip()) >= 4]
    if len(lines) <= MAX_LINES_PER_POEM:
        return lines
    step = len(lines) / MAX_LINES_PER_POEM
    return [lines[int(i * step)] for i in range(MAX_LINES_PER_POEM)]


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
        scores = self.matrix @ query_vec
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


def save_index(ids: list[str], matrix: np.ndarray, filename: str) -> None:
    np.savez(Path(EMBED_DIR) / filename,
             ids=np.asarray(json.dumps(ids)), matrix=matrix.astype(np.float32))
