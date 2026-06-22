"""Evaluate retrieval channels on the hand-labeled eval set (eval/queries.jsonl).

Pure retrieval (no expert/LLM stage): for each query and each mode, fetch the
top-K candidate poem ids and measure whether the expected poems are recalled.
This isolates retrieval quality — exactly where the gist-compression blind spot
and the ColBERT comparison live. Image mode embeds the English intent (parity
with production); line/keyword/colbert use the raw query.

Usage: EMBED_API_KEY=... python scripts/eval_retrieval.py [modes...] [--k 30]
       modes default: image line keyword hybrid   (colbert added by spike_colbert.py)
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poemferry.config import load_settings  # noqa: E402
from poemferry.corpus import load_poems  # noqa: E402
from poemferry.lexical import LexicalIndex  # noqa: E402
from poemferry.llm import make_client  # noqa: E402
from poemferry.swarm import (  # noqa: E402
    RRF_WEIGHTS,
    _rrf_fuse,
    _round_robin,
    understand_query,
)
from poemferry.vectors import DOC_FILE, VectorIndex, embed_query  # noqa: E402

K = 30


def load_queries() -> list[dict]:
    path = Path("eval/queries.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def retrieve(mode, q, reps, settings, doc, lex, k) -> list[str]:
    intent_text = reps["intent"]  # English LLM intent
    expanded = reps.get("expanded") or {}  # {lang: terms w/ near-synonyms} from understand
    concept = [lex.search(str(t), k) for t in expanded.values() if t]  # per-lang round-robin
    if mode == "image":  # EN intent vs EN gist (semantic, cross-lingual)
        return [pid for pid, _ in doc.search_unique(embed_query(settings, intent_text), k)]
    if mode == "keyword":  # lexical term-overlap on raw text (same-language detail)
        return [pid for pid, _ in lex.search(q, k)]
    if mode == "concept":  # cross-lingual near-synonym expansion
        return _round_robin(*concept)[:k] if concept else []
    if mode == "hybrid":  # production: weighted RRF over image + keyword + concept
        d = doc.search_unique(embed_query(settings, intent_text), k)
        kw = lex.search(q, k)
        channels = [
            (RRF_WEIGHTS["image"], [pid for pid, _ in d]),
            (RRF_WEIGHTS["keyword"], [pid for pid, _ in kw]),
        ]
        if concept:
            channels.append((RRF_WEIGHTS["concept"], _round_robin(*concept)))
        return _rrf_fuse(channels)[:k]
    raise ValueError(mode)


async def main() -> None:
    raw = sys.argv[1:]
    k = K
    if "--k" in raw:
        i = raw.index("--k")
        k = int(raw[i + 1])
        raw = raw[:i] + raw[i + 2 :]
    modes = raw or ["image", "keyword", "concept", "hybrid"]

    settings = load_settings()
    poems = load_poems(settings.poems_path)
    doc = VectorIndex.load(DOC_FILE)
    lex = LexicalIndex(poems)
    client = make_client(settings)
    queries = load_queries()

    # per-query reps: English LLM intent + cross-lingual concept expansion (from understand)
    reps = {}
    for qq in queries:
        try:
            u = await understand_query(client, settings, qq["q"])
            intent = u.get("intent_summary") or qq["q"]
            expanded = u.get("expanded") or {}
        except Exception:
            intent, expanded = qq["q"], {}
        reps[qq["id"]] = {"intent": intent, "expanded": expanded}

    totals = {m: 0.0 for m in modes}
    print(f"\n{'query':<26} " + " ".join(f"{m:>11}" for m in modes))
    for qq in queries:
        expect = set(qq["expect"])
        row = {}
        for m in modes:
            got = await retrieve(m, qq["q"], reps[qq["id"]], settings, doc, lex, k)
            top = set(got[:k])
            recall = len(expect & top) / len(expect)
            totals[m] += recall
            row[m] = recall
        cells = " ".join(f"{row[m]*100:>10.0f}%" for m in modes)
        print(f"{qq['id']:<26} {cells}   ({qq['type']})")
    n = len(queries)
    print(f"\n{'MEAN recall@'+str(k):<26} " + " ".join(f"{totals[m]/n*100:>10.0f}%" for m in modes))


if __name__ == "__main__":
    asyncio.run(main())
