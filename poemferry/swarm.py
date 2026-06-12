import asyncio
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .config import Settings
from .llm import chat_json
from .models import Poem, Verdict
from .retriever import Retriever

UNDERSTAND_SYS = """You are the query-understanding module of a multilingual poetry \
search engine. The user describes, in any language, a poem they are trying to find \
— an image, a mood, a scene, or a half-remembered line. Extract a structured search \
intent.

Respond ONLY with a JSON object with exactly these keys:
- intent_summary: a concise normalized semantic summary, in English, of what they seek
- query_lang: ISO code of the language the user wrote in (e.g. "zh", "en", "ja", "fr")
- imagery: array of short strings (key images)
- emotion: array of short strings (emotions/mood)
- setting: array of short strings (time, place, situation)
- has_fragment: boolean — true if they quoted or remembered an actual line
- fragments: array of the verbatim lines/phrases the user quoted or half-remembers
  from the poem itself (copy them exactly as written; empty array if none)
- lang_hint: array of language codes to restrict the search to, or null for any language

Do not add extra keys. Do not translate the user's words into the poem; just analyze."""

SCOUT_SYS = """You are a scout agent in a poetry search swarm. You receive compact \
digests of poems (id | language | author | title | opening lines) followed by a \
user's description. Pick the poems that plausibly match the description by meaning, \
imagery, or mood — across languages. Be inclusive: a later stage reads the full \
texts and judges strictly; your job is recall, not precision.

Respond ONLY with JSON: {"candidates": [{"poem_id": "...", "score": 0.0}]}
score is 0.0–1.0 plausibility. Include only plausible poems (empty array if none)."""

MATCH_SYS = """You are one agent in a swarm that judges whether candidate poems match \
a user's description.

Hard rules:
1. NEVER translate any poem. Poems stay in their original language, untouched.
2. Judge by meaning, imagery, and emotion — across languages. A description in one \
language can match a poem in another.
3. For each candidate poem, decide:
   - match (boolean): true only when there is a genuine semantic correspondence
   - confidence (0.0–1.0)
   - matched_description_aspects: which parts of the user's description it matches
   - evidence_lines: lines copied VERBATIM from the poem's original text (never translated)
   - explanation: HOW it matches
4. The explanation MUST be written in the user's query language (given as query_lang).
5. Be strict. Most candidates will not match; say so.

Respond ONLY with JSON of this shape:
{"verdicts": [{"poem_id": "...", "match": true, "confidence": 0.0,
  "matched_description_aspects": ["..."], "evidence_lines": ["..."],
  "explanation": "...", "explanation_lang": "<query_lang>"}]}
Include ONLY the poems that match (an empty array when none do); never list
non-matching poems."""


def _format_poem(p: Poem) -> str:
    head = f"[id={p.id}] {p.title or '(untitled)'} — {p.author or '(unknown)'} (lang={p.language})"
    return head + "\n" + p.full_text


def _digest(p: Poem) -> str:
    # With a whole-poem gist available, one opening line suffices; the gist
    # is what lets scouts see content from a long poem's later sections.
    gist = (p.enrichment or {}).get("gist")
    if gist:
        opening = (p.full_text.splitlines() or [""])[0][:40]
        return (
            f"{p.id} | {p.language} | {p.author or '?'} | {p.title or '?'} | "
            f"{opening} | gist: {gist}"
        )
    opening = " / ".join(p.full_text.splitlines()[:2])[:80]
    return f"{p.id} | {p.language} | {p.author or '?'} | {p.title or '?'} | {opening}"


async def understand_query(
    client: AsyncOpenAI, settings: Settings, description: str, usage: dict | None = None
) -> dict:
    user = f"User description:\n{description}"
    return await chat_json(
        client, settings.deepseek_model, UNDERSTAND_SYS, user, max_tokens=600, usage=usage
    )


async def scout_batch(
    client: AsyncOpenAI,
    settings: Settings,
    description: str,
    batch: list[Poem],
    usage: dict | None = None,
) -> list[tuple[str, float]]:
    """Stage-1 scan over compact digests; returns (poem_id, score) candidates.

    The static digest block comes FIRST and the query LAST: chunks are stable
    across queries, so DeepSeek's automatic prefix caching makes repeat scans
    of the same chunk ~10x cheaper on input tokens.
    """
    digests = "\n".join(_digest(p) for p in batch)
    user = f"Poem digests:\n{digests}\n\nUser description:\n{description}"
    data = await chat_json(
        client, settings.deepseek_model, SCOUT_SYS, user, max_tokens=1500, usage=usage
    )
    ids = {p.id for p in batch}
    out: list[tuple[str, float]] = []
    for c in data.get("candidates", []):
        pid = c.get("poem_id")
        if pid in ids:
            try:
                out.append((pid, float(c.get("score", 0.5))))
            except (TypeError, ValueError):
                out.append((pid, 0.5))
    return out


async def match_batch(
    client: AsyncOpenAI,
    settings: Settings,
    description: str,
    query_lang: str,
    batch: list[Poem],
    usage: dict | None = None,
) -> list[Verdict]:
    poems_blob = "\n\n---\n\n".join(_format_poem(p) for p in batch)
    user = (
        f"Candidate poems (judge each; keep original language):\n\n{poems_blob}\n\n"
        f"query_lang: {query_lang}\n"
        f"User description:\n{description}"
    )
    data = await chat_json(
        client, settings.deepseek_model, MATCH_SYS, user, max_tokens=2500, usage=usage
    )
    verdicts: list[Verdict] = []
    for raw in data.get("verdicts", []):
        try:
            verdicts.append(Verdict(**raw))
        except Exception:
            continue
    return verdicts


def _chunk(items: list, size: int, max_chunks: int) -> list[list]:
    size = max(size, -(-len(items) // max_chunks))
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _run_stage(
    stage: int,
    batches: list[list[Poem]],
    worker,
    concurrency: int,
) -> AsyncIterator[dict]:
    """Run one swarm wave; yields agent_start/agent_done (+ worker events)."""
    sem = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def run_agent(idx: int, batch: list[Poem]) -> None:
        async with sem:
            await queue.put(
                {"type": "agent_start", "stage": stage, "agent": idx, "size": len(batch)}
            )
            hits = await worker(idx, batch, queue)
            await queue.put({"type": "agent_done", "stage": stage, "agent": idx, "hits": hits})

    tasks = [asyncio.create_task(run_agent(i, b)) for i, b in enumerate(batches)]
    finished = 0
    while finished < len(batches):
        event = await queue.get()
        if event["type"] == "agent_done":
            finished += 1
        yield event
    await asyncio.gather(*tasks)


async def search_stream(
    client: AsyncOpenAI,
    settings: Settings,
    retriever: Retriever,
    poems: list[Poem],
    description: str,
    fragment_index=None,
) -> AsyncIterator[dict]:
    """Run the two-stage query pipeline, yielding progress events for SSE.

    Stage 1 (scouts): bounded swarm scans compact digests of every candidate
    and shortlists plausible poems — cheap tokens, prefix-cache friendly.
    Stage 2 (judges): swarm reads the full text of the shortlist and produces
    verdicts with verbatim evidence. Small corpora skip straight to stage 2.

    Event types: start, understanding, candidates, swarm_dispatched(stage),
    agent_start(stage), agent_done(stage), shortlist, verdict, done, error.
    """
    yield {"type": "start", "description": description}
    usage: dict = {}

    try:
        intent = await understand_query(client, settings, description, usage)
    except Exception as e:
        yield {"type": "error", "stage": "understanding", "message": str(e)}
        intent = {"intent_summary": description, "query_lang": "", "lang_hint": None}
    query_lang = intent.get("query_lang") or ""
    yield {"type": "understanding", "intent": intent}

    candidates = retriever.retrieve(intent, poems)
    yield {"type": "candidates", "count": len(candidates)}
    if not candidates:
        yield {"type": "done", "matched": 0}
        return

    by_id = {p.id: p for p in candidates}

    # Exact-fragment channel: user-quoted lines are matched verbatim against
    # the whole corpus (normalized, trad→simp folded) and go straight to the
    # judges — immune to scout misses on long poems' later sections.
    forced: list[Poem] = []
    fragments = intent.get("fragments") or []
    if fragment_index is not None and fragments:
        hit_ids = set(fragment_index.find(fragments))
        forced = [p for p in candidates if p.id in hit_ids]
        if forced:
            yield {"type": "fragment_hits", "count": len(forced)}
    forced_ids = {p.id for p in forced}

    if len(candidates) > settings.shortlist_size:
        scout_batches = _chunk(candidates, settings.scout_batch_size, settings.swarm_max_agents)
        yield {
            "type": "swarm_dispatched",
            "stage": 1,
            "agents": len(scout_batches),
            "batch_size": len(scout_batches[0]),
        }

        scored: list[tuple[str, float]] = []

        async def scout_worker(idx: int, batch: list[Poem], queue: asyncio.Queue) -> int:
            try:
                found = await scout_batch(client, settings, description, batch, usage)
            except Exception:
                found = []
            scored.extend(found)
            return len(found)

        async for event in _run_stage(1, scout_batches, scout_worker, settings.swarm_concurrency):
            yield event

        scored.sort(key=lambda t: -t[1])
        scouted = [
            by_id[pid]
            for pid, _ in scored[: settings.shortlist_size]
            if pid not in forced_ids
        ]
        shortlist = forced + scouted
        yield {"type": "shortlist", "count": len(shortlist)}
        if not shortlist:
            yield {"type": "done", "matched": 0}
            return
    else:
        shortlist = candidates

    judge_batches = _chunk(shortlist, settings.swarm_batch_size, settings.swarm_max_agents)
    yield {
        "type": "swarm_dispatched",
        "stage": 2,
        "agents": len(judge_batches),
        "batch_size": len(judge_batches[0]),
    }

    matched = 0

    async def judge_worker(idx: int, batch: list[Poem], queue: asyncio.Queue) -> int:
        nonlocal matched
        try:
            verdicts = await match_batch(client, settings, description, query_lang, batch, usage)
        except Exception:
            verdicts = []
        hits = 0
        for v in verdicts:
            if v.match and v.poem_id in by_id:
                hits += 1
                matched += 1
                await queue.put(
                    {
                        "type": "verdict",
                        "agent": idx,
                        "verdict": v.model_dump(),
                        "poem": by_id[v.poem_id].model_dump(),
                    }
                )
        return hits

    async for event in _run_stage(2, judge_batches, judge_worker, settings.swarm_concurrency):
        yield event

    yield {"type": "done", "matched": matched, "usage": usage}
