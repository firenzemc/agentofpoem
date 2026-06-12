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
- lang_hint: array of language codes to restrict the search to, or null for any language

Do not add extra keys. Do not translate the user's words into the poem; just analyze."""

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


async def understand_query(client: AsyncOpenAI, settings: Settings, description: str) -> dict:
    user = f"User description:\n{description}"
    return await chat_json(client, settings.deepseek_model, UNDERSTAND_SYS, user, max_tokens=600)


async def match_batch(
    client: AsyncOpenAI,
    settings: Settings,
    description: str,
    query_lang: str,
    batch: list[Poem],
) -> list[Verdict]:
    poems_blob = "\n\n---\n\n".join(_format_poem(p) for p in batch)
    user = (
        f"query_lang: {query_lang}\n"
        f"User description:\n{description}\n\n"
        f"Candidate poems (judge each; keep original language):\n\n{poems_blob}"
    )
    data = await chat_json(client, settings.deepseek_model, MATCH_SYS, user, max_tokens=2500)
    verdicts: list[Verdict] = []
    for raw in data.get("verdicts", []):
        try:
            verdicts.append(Verdict(**raw))
        except Exception:
            continue
    return verdicts


async def search_stream(
    client: AsyncOpenAI,
    settings: Settings,
    retriever: Retriever,
    poems: list[Poem],
    description: str,
) -> AsyncIterator[dict]:
    """Run the full query pipeline, yielding progress events for SSE.

    Event types: start, understanding, candidates, swarm_dispatched, agent_start,
    verdict, agent_done, done, error.

    Each agent reads a small batch of candidates. Events are streamed through a
    queue as agents begin and finish, so the UI can render the live swarm —
    dozens to hundreds of agents lighting up and result cards arriving one by one.
    """
    yield {"type": "start", "description": description}

    try:
        intent = await understand_query(client, settings, description)
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

    # Grow the per-agent batch when needed so the agent count stays bounded
    # (≤ swarm_max_agents) however large the candidate set gets.
    bs = max(settings.swarm_batch_size, -(-len(candidates) // settings.swarm_max_agents))
    batches = [candidates[i : i + bs] for i in range(0, len(candidates), bs)]
    by_id = {p.id: p for p in candidates}
    sem = asyncio.Semaphore(settings.swarm_concurrency)
    queue: asyncio.Queue[dict] = asyncio.Queue()

    yield {"type": "swarm_dispatched", "agents": len(batches), "batch_size": bs}

    async def run_agent(idx: int, batch: list[Poem]) -> None:
        async with sem:
            await queue.put({"type": "agent_start", "agent": idx, "size": len(batch)})
            try:
                verdicts = await match_batch(client, settings, description, query_lang, batch)
            except Exception:
                verdicts = []
            hits = 0
            for v in verdicts:
                if v.match and v.poem_id in by_id:
                    hits += 1
                    await queue.put(
                        {
                            "type": "verdict",
                            "agent": idx,
                            "verdict": v.model_dump(),
                            "poem": by_id[v.poem_id].model_dump(),
                        }
                    )
            await queue.put({"type": "agent_done", "agent": idx, "hits": hits})

    tasks = [asyncio.create_task(run_agent(i, b)) for i, b in enumerate(batches)]

    matched = 0
    finished = 0
    total = len(batches)
    while finished < total:
        event = await queue.get()
        if event["type"] == "verdict":
            matched += 1
        elif event["type"] == "agent_done":
            finished += 1
        yield event

    await asyncio.gather(*tasks)
    yield {"type": "done", "matched": matched}
