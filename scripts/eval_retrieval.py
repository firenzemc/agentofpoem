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
from poemferry.swarm import _merge_scores, understand_query  # noqa: E402
from poemferry.vectors import DOC_FILE, LINE_FILE, VectorIndex, embed_query  # noqa: E402

K = 30


def load_queries() -> list[dict]:
    path = Path("eval/queries.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def retrieve(mode, q, intent_text, settings, doc, line, lex, k) -> list[str]:
    if mode == "image":
        return [pid for pid, _ in doc.search_unique(embed_query(settings, intent_text), k)]
    if mode == "line":
        return [pid for pid, _ in line.search_unique(embed_query(settings, q), k)]
    if mode == "keyword":
        return [pid for pid, _ in lex.search(q, k)]
    if mode == "hybrid":
        d = doc.search_unique(embed_query(settings, intent_text), k)
        line_h = line.search_unique(embed_query(settings, q), k)
        kw = lex.search(q, k)
        return _merge_scores(d, line_h, kw)[:k]
    raise ValueError(mode)


async def main() -> None:
    raw = sys.argv[1:]
    k = K
    if "--k" in raw:
        i = raw.index("--k")
        k = int(raw[i + 1])
        raw = raw[:i] + raw[i + 2 :]
    modes = raw or ["image", "line", "keyword", "hybrid"]

    settings = load_settings()
    poems = load_poems(settings.poems_path)
    doc = VectorIndex.load(DOC_FILE)
    line = VectorIndex.load(LINE_FILE)
    lex = LexicalIndex(poems)
    client = make_client(settings)
    queries = load_queries()

    # cache English intent per query (image mode parity)
    intents = {}
    for qq in queries:
        try:
            intents[qq["id"]] = (await understand_query(client, settings, qq["q"])).get("intent_summary") or qq["q"]
        except Exception:
            intents[qq["id"]] = qq["q"]

    totals = {m: 0.0 for m in modes}
    print(f"\n{'query':<26} " + " ".join(f"{m:>9}" for m in modes))
    for qq in queries:
        expect = set(qq["expect"])
        row = {}
        for m in modes:
            got = await retrieve(m, qq["q"], intents[qq["id"]], settings, doc, line, lex, k)
            top = set(got[:k])
            recall = len(expect & top) / len(expect)
            totals[m] += recall
            row[m] = recall
        cells = " ".join(f"{row[m]*100:>8.0f}%" for m in modes)
        print(f"{qq['id']:<26} {cells}   ({qq['type']})")
    n = len(queries)
    print(f"\n{'MEAN recall@'+str(k):<26} " + " ".join(f"{totals[m]/n*100:>8.0f}%" for m in modes))


if __name__ == "__main__":
    asyncio.run(main())
