"""Spike: jina-colbert-v2 late-interaction (MaxSim) over the full corpus, evaluated
against the same eval set as scripts/eval_retrieval.py.

ISOLATED, NON-PRODUCTION, eval-only. jina-colbert-v2 weights are CC-BY-NC — used
here strictly for a research-quality read on whether token-level MaxSim beats the
current channels, especially on the cross-lingual cases (q6/q7) that baseline
scores 0%. In-memory MaxSim (no Qdrant needed at 13k). Token embeddings are cached
so re-runs skip encoding.

Usage: uv run --with fastembed python scripts/spike_colbert.py [--k 30]
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poemferry.corpus import load_poems  # noqa: E402

MODEL = "jinaai/jina-colbert-v2"
CACHE = Path("data/colbert_jina.pkl")  # gitignored (large)
MAXLEN = 1200  # truncate very long poems (Dante cantos / 离骚) for encoding


def encode_corpus(poems):
    from fastembed import LateInteractionTextEmbedding

    model = LateInteractionTextEmbedding(MODEL)
    texts = [p.full_text[:MAXLEN] for p in poems]
    t0 = time.time()
    vecs = []
    for i, emb in enumerate(model.embed(texts, batch_size=32)):
        vecs.append(np.asarray(emb, dtype=np.float32))  # (n_tokens, 128), already normalized
        if (i + 1) % 1000 == 0:
            print(f"  encoded {i + 1}/{len(texts)}  ({time.time() - t0:.0f}s)")
    return vecs


def load_or_encode(poems):
    if CACHE.exists():
        with CACHE.open("rb") as f:
            ids, vecs = pickle.load(f)
        if ids == [p.id for p in poems]:
            print(f"loaded cached colbert vectors ({len(vecs)} docs)")
            return vecs
    print(f"encoding {len(poems)} poems with {MODEL} (CPU, one-time)...")
    vecs = encode_corpus(poems)
    with CACHE.open("wb") as f:
        pickle.dump(([p.id for p in poems], vecs), f)
    print(f"cached → {CACHE}")
    return vecs


def maxsim_scores(qmat: np.ndarray, doc_vecs: list[np.ndarray]) -> np.ndarray:
    # score = sum over query tokens of max over doc tokens of dot product
    out = np.empty(len(doc_vecs), dtype=np.float32)
    for i, d in enumerate(doc_vecs):
        out[i] = (qmat @ d.T).max(axis=1).sum()
    return out


def main() -> None:
    k = 30
    if "--k" in sys.argv:
        k = int(sys.argv[sys.argv.index("--k") + 1])
    poems = load_poems("data")
    ids = [p.id for p in poems]
    doc_vecs = load_or_encode(poems)

    from fastembed import LateInteractionTextEmbedding

    model = LateInteractionTextEmbedding(MODEL)
    queries = [json.loads(x) for x in Path("eval/queries.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

    print(f"\n{'query':<26} {'colbert@'+str(k):>12}  top-1")
    total = 0.0
    for qq in queries:
        qmat = np.asarray(next(model.query_embed([qq["q"]])), dtype=np.float32)
        scores = maxsim_scores(qmat, doc_vecs)
        order = np.argsort(-scores)[: max(k, 1)]
        topids = [ids[i] for i in order]
        expect = set(qq["expect"])
        recall = len(expect & set(topids[:k])) / len(expect)
        total += recall
        p0 = poems[order[0]]
        print(f"{qq['id']:<26} {recall*100:>11.0f}%  {(p0.title or '')[:18]} ({p0.language})  [{qq['type']}]")
    print(f"\n{'MEAN recall@'+str(k):<26} {total/len(queries)*100:>11.0f}%")


if __name__ == "__main__":
    main()
