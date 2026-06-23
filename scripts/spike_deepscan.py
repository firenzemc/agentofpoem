"""Spike ②: brute-force deep scan — a cheap model reads the FULL text of every poem
and scores it against the query. No embedding pre-filter, no compression. Tests
whether "read everything" cracks the cross-lingual incidental-detail case (q7:
English query → 将进酒 by its Yellow-River line) that all retrieval misses, and
how reproducible it is (temp=0 + fixed rubric).

Usage: EMBED unneeded. DEEPSEEK_API_KEY=... python scripts/spike_deepscan.py "<query>" [--runs 2]
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI  # noqa: E402

from poemferry.config import load_settings  # noqa: E402
from poemferry.corpus import load_poems  # noqa: E402
from poemferry.llm import make_client, swarm_model  # noqa: E402

BATCH = 16
MAXCHARS = 420
CONCURRENCY = 24
FAILS = {"n": 0}

RUBRIC = """You are one agent in a swarm reading poems. Judge how well each poem matches \
the user's description. Read the FULL poem in its ORIGINAL language — NEVER translate it. \
Judge across languages: an English description can match a Chinese poem via shared imagery.
Score 0-5: 0 unrelated · 1 faint echo · 2 one minor shared image · 3 a key shared image/theme \
· 4 strong overlap · 5 unmistakable match.
Use the EXACT poem ids given (copy them verbatim; never invent ids).
Respond ONLY with compact JSON listing ONLY poems scoring 2 or higher:
{"s":[{"id":"<exact poem_id>","v":<2-5>}]}  (empty list if none)."""


async def score_batch(client: AsyncOpenAI, model, query, batch, sem, scores):
    import json
    ids = {p.id for p in batch}
    async with sem:
        blob = "\n---\n".join(f"[{p.id}] ({p.language}) {p.full_text[:MAXCHARS]}" for p in batch)
        for attempt in range(2):
            try:
                resp = await client.chat.completions.create(
                    model=model, temperature=0, max_tokens=900,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": RUBRIC},
                              {"role": "user", "content": f"Description: {query}\n\nPoems:\n{blob}"}])
                data = json.loads(resp.choices[0].message.content or "{}")
                for r in data.get("s", []):
                    if r.get("id") in ids:  # ignore hallucinated ids
                        try:
                            scores[r["id"]] = float(r["v"])
                        except (TypeError, ValueError):
                            pass
                return
            except Exception:
                if attempt == 1:
                    FAILS["n"] += 1


async def deepscan(client, settings, poems, query) -> dict:
    model = swarm_model(settings)
    sem = asyncio.Semaphore(CONCURRENCY)
    scores: dict = {}
    batches = [poems[i:i + BATCH] for i in range(0, len(poems), BATCH)]
    await asyncio.gather(*(score_batch(client, model, query, b, sem, scores) for b in batches))
    return scores


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = args[0] if args else "a great river rushing to the sea, never to return; aging and the passing of time"
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 2

    settings = load_settings()
    poems = load_poems("data")
    by_id = {p.id: p for p in poems}
    client = make_client(settings)
    print(f"deep-scanning {len(poems)} poems (full text), query: {query!r}")

    import time
    top_sets = []
    for n in range(runs):
        t0 = time.time()
        scores = await deepscan(client, settings, poems, query)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top = [pid for pid, v in ranked[:10]]
        top_sets.append(set(top))
        jjj = [(i, v) for i, (pid, v) in enumerate(ranked) if "tang300-0004" in pid or "tang300-0124" in pid]
        print(f"\nrun {n+1} ({time.time()-t0:.0f}s, {len(scores)} hits≥2, {FAILS['n']} batch fails):"
              f" 将进酒 rank/score={jjj[:2]}")
        FAILS["n"] = 0
        for pid, v in ranked[:8]:
            p = by_id.get(pid)
            if p:
                print(f"  {v:.0f} ({p.language}) {p.title} — {p.author}")
    if runs > 1:
        overlap = len(top_sets[0] & top_sets[1]) / 10
        print(f"\nreproducibility: top-10 overlap across runs = {overlap*100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
