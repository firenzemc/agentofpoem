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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poemferry.config import load_settings  # noqa: E402
from poemferry.llm import chat_json, make_client  # noqa: E402

# DeepSeek tolerates 48; a local LM Studio server caps at ~4 concurrent
# predictions, so override via ENRICH_CONCURRENCY when pointing there.
CONCURRENCY = int(os.environ.get("ENRICH_CONCURRENCY", "48"))
MAX_CHARS = 6000
FLUSH_EVERY = 2000  # checkpoint interval: a 28h/363k run must survive interruption
# Consecutive failures ⇒ the endpoint/model is gone (e.g. LM Studio model ejected).
# Stop cleanly at the last checkpoint instead of churning through the rest; re-run resumes.
FAIL_LIMIT = 20

# LM Studio rejects json_object; it wants a json_schema. Build one matching the
# enrichment shape and use it whenever the endpoint isn't DeepSeek.
_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "enrichment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "themes": {"type": "array", "items": {"type": "string"}},
                "gist": {"type": "string"},
                "images": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["themes", "gist", "images"],
            "additionalProperties": False,
        },
    },
}

ENRICH_SYS = """You are an enrichment agent in a poetry indexing swarm. You receive \
one poem in its original language. Produce compact English metadata covering the \
WHOLE poem — themes and scenes from its middle and ending matter as much as its \
opening (a searcher may only remember a later passage).

Respond ONLY with JSON:
{"themes": ["..."], "gist": "...", "images": ["..."]}
- themes: 3-6 short English keywords/phrases for the poem's subject and mood
- gist: at most 25 English words summarizing the full arc of the poem
- images: 3-8 English phrases naming the CONCRETE images, scenes, objects, and
  actions that actually appear ANYWHERE in the poem — including striking but
  incidental ones (e.g. a famous opening line whose image is not the poem's main
  theme). This is for search: a reader may recall a vivid image, not the theme.
Never include the original text; never translate lines verbatim; describe in English."""


async def enrich_one(client, settings, sem, record: dict, usage: dict, state: dict) -> None:
    if state["abort"].is_set():
        return
    async with sem:
        if state["abort"].is_set():
            return
        text = record["full_text"][:MAX_CHARS]
        user = (
            f"Title: {record.get('title') or '?'}\n"
            f"Author: {record.get('author') or '?'}\n"
            f"Language: {record['language']}\n\nPoem:\n{text}"
        )
        is_deepseek = "api.deepseek.com" in settings.deepseek_base_url
        # DeepSeek: json_object + thinking OFF (this task needs no reasoning;
        # thinking ~4x's the output tokens/cost for no quality gain).
        # LM Studio: json_schema, no thinking toggle.
        fmt = None if is_deepseek else _SCHEMA_FORMAT
        extra = {"thinking": {"type": "disabled"}} if is_deepseek else None
        try:
            data = await chat_json(
                client, settings.deepseek_model, ENRICH_SYS, user,
                max_tokens=1600, usage=usage, response_format=fmt, extra_body=extra,
            )
        except Exception as e:
            state["fails"] += 1
            if state["fails"] >= FAIL_LIMIT:
                state["abort"].set()
            print(f"  ! {record['id']}: {e}")
            return
        state["fails"] = 0
        # v4-flash occasionally wraps the object in a JSON array; tolerate it
        # instead of crashing the whole run.
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        if not isinstance(data, dict):
            return
        gist = str(data.get("gist", "")).strip()
        if gist:
            record["enrichment"] = {
                "themes": [str(t) for t in data.get("themes", [])][:6],
                "gist": gist[:300],
                "images": [str(x) for x in data.get("images", [])][:8],
            }


def _checkpoint(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)


async def enrich_file(client, settings, path: Path, state: dict) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    # Re-enrich anything missing the images field (added after the first pass).
    todo = [r for r in records if not (r.get("enrichment") or {}).get("images")]
    if not todo:
        print(f"{path.name}: all {len(records)} already enriched")
        return
    print(f"{path.name}: enriching {len(todo)}/{len(records)}", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    usage: dict = {}
    # Flush in chunks so an interrupted long run resumes (enrich_one is
    # idempotent — a restart re-scans todo and skips what's already enriched).
    for i in range(0, len(todo), FLUSH_EVERY):
        chunk = todo[i : i + FLUSH_EVERY]
        await asyncio.gather(*(enrich_one(client, settings, sem, r, usage, state) for r in chunk))
        _checkpoint(path, records)
        done = sum(1 for r in records if r.get("enrichment"))
        print(
            f"{path.name}: checkpoint {done}/{len(records)} | "
            f"tokens prompt={usage.get('prompt', 0):,} completion={usage.get('completion', 0):,}",
            flush=True,
        )
        if state["abort"].is_set():
            print(
                f"{path.name}: stopped after {FAIL_LIMIT} consecutive failures "
                "(endpoint/model gone); progress saved, re-run to resume",
                flush=True,
            )
            return


async def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    settings = load_settings()
    if not settings.deepseek_api_key:
        sys.exit("DEEPSEEK_API_KEY is not set")
    client = make_client(settings)
    state = {"fails": 0, "abort": asyncio.Event()}
    for path in sorted(data_dir.glob("*.jsonl")):
        await enrich_file(client, settings, path, state)
        if state["abort"].is_set():
            break


if __name__ == "__main__":
    asyncio.run(main())
