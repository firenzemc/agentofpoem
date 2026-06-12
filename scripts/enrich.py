"""Offline enrichment swarm: one cheap agent per poem writes a whole-poem gist.

For every poem in data/*.jsonl lacking an `enrichment` field, asks DeepSeek for
a compact English summary covering the ENTIRE poem (including later sections),
so the query-time scout stage can recall poems whose relevant content is not
in the opening lines. Writes results back into the same JSONL files
(atomic .tmp + rename). Idempotent: re-running skips enriched poems.

Usage: DEEPSEEK_API_KEY=... python scripts/enrich.py [data_dir]
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poemferry.config import load_settings  # noqa: E402
from poemferry.llm import chat_json, make_client  # noqa: E402

CONCURRENCY = 48
MAX_CHARS = 6000

ENRICH_SYS = """You are an enrichment agent in a poetry indexing swarm. You receive \
one poem in its original language. Produce compact English metadata covering the \
WHOLE poem — themes and scenes from its middle and ending matter as much as its \
opening (a searcher may only remember a later passage).

Respond ONLY with JSON:
{"themes": ["..."], "gist": "..."}
- themes: 3-6 short English keywords/phrases
- gist: at most 25 English words summarizing the full arc of the poem, including
  distinctive later images, scenes, or famous lines (paraphrased in English).
Never include the original text; never translate lines verbatim; summarize."""


async def enrich_one(client, settings, sem, record: dict, usage: dict) -> None:
    async with sem:
        text = record["full_text"][:MAX_CHARS]
        user = (
            f"Title: {record.get('title') or '?'}\n"
            f"Author: {record.get('author') or '?'}\n"
            f"Language: {record['language']}\n\nPoem:\n{text}"
        )
        try:
            # Generous cap: v4-flash spends reasoning tokens from the same
            # budget before emitting the JSON content.
            data = await chat_json(
                client, settings.deepseek_model, ENRICH_SYS, user, max_tokens=1200, usage=usage
            )
        except Exception as e:
            print(f"  ! {record['id']}: {e}")
            return
        gist = str(data.get("gist", "")).strip()
        if gist:
            record["enrichment"] = {
                "themes": [str(t) for t in data.get("themes", [])][:6],
                "gist": gist[:300],
            }


async def enrich_file(client, settings, path: Path) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    todo = [r for r in records if not r.get("enrichment")]
    if not todo:
        print(f"{path.name}: all {len(records)} already enriched")
        return
    print(f"{path.name}: enriching {len(todo)}/{len(records)}")
    sem = asyncio.Semaphore(CONCURRENCY)
    usage: dict = {}
    await asyncio.gather(*(enrich_one(client, settings, sem, r, usage) for r in todo))

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)
    done = sum(1 for r in records if r.get("enrichment"))
    print(
        f"{path.name}: {done}/{len(records)} enriched | "
        f"tokens prompt={usage.get('prompt', 0):,} completion={usage.get('completion', 0):,}"
    )


async def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    settings = load_settings()
    if not settings.deepseek_api_key:
        sys.exit("DEEPSEEK_API_KEY is not set")
    client = make_client(settings)
    for path in sorted(data_dir.glob("*.jsonl")):
        await enrich_file(client, settings, path)


if __name__ == "__main__":
    asyncio.run(main())
