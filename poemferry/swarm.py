import asyncio
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .config import Settings
from .llm import chat_json
from .models import Poem, Verdict
from .retriever import Retriever

UNDERSTAND_SYS = """You are the query-understanding module of a multilingual poetry \
search engine. The user describes, in any language, a poem they are trying to find \
— an image, a mood, a scene, a sequence of events, or a half-remembered line. \
Decompose the request into independently checkable criteria.

Respond ONLY with a JSON object with exactly these keys:
- intent_summary: a concise normalized English summary of what they seek (used for retrieval)
- query_lang: ISO code of the language the user wrote in (e.g. "zh", "en", "ja", "fr")
- criteria: array of 1-5 objects, each an independently verifiable condition:
    {"aspect": "<short English description>",
     "kind": "imagery|emotion|action|sequence|setting",
     "weight": 0.0-1.0}
  Decompose compound requests: give EACH distinct emotion, action, image, or scene
  its own criterion. If the order of events matters, ADD a separate "sequence"
  criterion on top of the component ones. Example: "crying then drinking" →
  [weeping(emotion), drinking wine(action), crying precedes drinking(sequence)].
- fragments: array of verbatim lines/phrases the user quoted from the poem (empty if none)
- lang_hint: array of language codes — ONLY when the user EXPLICITLY asks for a
  language (e.g. "a French poem"). Otherwise null. Never infer it from the language
  the user happens to be writing in; matching is cross-lingual by design.

Do not add extra keys. Do not translate; just analyze."""

EXPERT_SYS = """You are a specialist agent in a poetry-judging swarm. You verify ONE \
criterion against candidate poems.

Hard rules:
1. NEVER translate any poem; poems stay in their original language.
2. Judge across languages — the criterion is in English, poems may be any language.
3. A poem satisfies the criterion only on genuine evidence in its text.
4. For "sequence" criteria, judge whether the events occur in the stated order; be \
lenient if the poem strongly implies it.
5. evidence_lines must be copied VERBATIM from the poem (never translated). The note \
explaining the match must be written in the given query_lang.

Respond ONLY with JSON: {"hits": [{"poem_id": "...", "evidence_lines": ["..."], \
"note": "..."}]}
Include ONLY poems that satisfy THIS criterion (empty array if none)."""

SCOUT_SYS = """You are a scout agent in a poetry search swarm. You receive compact \
digests of poems (id | language | author | title | gist) followed by a user's \
description. Pick the poems that plausibly match by meaning, imagery, or mood — \
across languages. Be inclusive; a later stage judges strictly.

Respond ONLY with JSON: {"candidates": [{"poem_id": "...", "score": 0.0}]}
score is 0.0-1.0 plausibility. Include only plausible poems (empty array if none)."""


def _format_poem(p: Poem) -> str:
    head = f"[id={p.id}] {p.title or '(untitled)'} — {p.author or '(unknown)'} (lang={p.language})"
    return head + "\n" + p.full_text


def _digest(p: Poem) -> str:
    enr = p.enrichment or {}
    gist = enr.get("gist")
    if gist:
        return f"{p.id} | {p.language} | {p.author or '?'} | {p.title or '?'} | gist: {gist}"
    opening = " / ".join(p.full_text.splitlines()[:2])[:80]
    return f"{p.id} | {p.language} | {p.author or '?'} | {p.title or '?'} | {opening}"


async def understand_query(
    client: AsyncOpenAI, settings: Settings, description: str, usage: dict | None = None
) -> dict:
    user = f"User description:\n{description}"
    return await chat_json(
        client, settings.deepseek_model, UNDERSTAND_SYS, user, max_tokens=900, usage=usage
    )


async def scout_batch(
    client: AsyncOpenAI,
    settings: Settings,
    description: str,
    batch: list[Poem],
    usage: dict | None = None,
) -> list[tuple[str, float]]:
    digests = "\n".join(_digest(p) for p in batch)
    user = f"Poem digests:\n{digests}\n\nUser description:\n{description}"
    data = await chat_json(
        client, settings.deepseek_model, SCOUT_SYS, user, max_tokens=1500, usage=usage
    )
    ids = {p.id for p in batch}
    out: list[tuple[str, float]] = []
    for c in data.get("candidates", []):
        if c.get("poem_id") in ids:
            try:
                out.append((c["poem_id"], float(c.get("score", 0.5))))
            except (TypeError, ValueError):
                out.append((c["poem_id"], 0.5))
    return out


async def expert_batch(
    client: AsyncOpenAI,
    settings: Settings,
    criterion: dict,
    query_lang: str,
    batch: list[Poem],
    usage: dict | None = None,
) -> dict[str, dict]:
    """Verify one criterion against a batch; returns {poem_id: {evidence_lines, note}}."""
    poems_blob = "\n\n---\n\n".join(_format_poem(p) for p in batch)
    user = (
        f"Criterion ({criterion.get('kind', 'imagery')}): {criterion.get('aspect', '')}\n"
        f"query_lang: {query_lang}\n\n"
        f"Candidate poems (keep original language):\n\n{poems_blob}"
    )
    data = await chat_json(
        client, settings.deepseek_model, EXPERT_SYS, user, max_tokens=2000, usage=usage
    )
    ids = {p.id for p in batch}
    hits: dict[str, dict] = {}
    for h in data.get("hits", []):
        pid = h.get("poem_id")
        if pid in ids:
            hits[pid] = {
                "evidence_lines": [str(x) for x in h.get("evidence_lines", [])][:4],
                "note": str(h.get("note", "")),
            }
    return hits


def aggregate(
    shortlist: list[Poem],
    criteria: list[dict],
    hits_by_criterion: list[dict[str, dict]],
    query_lang: str,
    limit: int = 8,
) -> list[Verdict]:
    """Weighted vote across criteria. Sequence criteria contribute when satisfied
    but never hard-reject (missing one only lowers the score)."""
    total_weight = sum(c.get("weight", 1.0) for c in criteria) or 1.0
    by_id = {p.id: p for p in shortlist}
    verdicts: list[Verdict] = []
    for pid, poem in by_id.items():
        score = 0.0
        aspects: list[str] = []
        evidence: list[str] = []
        notes: list[str] = []
        for crit, hits in zip(criteria, hits_by_criterion):
            hit = hits.get(pid)
            if hit:
                score += crit.get("weight", 1.0)
                aspects.append(crit.get("aspect", ""))
                evidence.extend(hit["evidence_lines"])
                if hit["note"]:
                    notes.append(hit["note"])
        if not aspects:
            continue
        confidence = round(score / total_weight, 3)
        verdicts.append(
            Verdict(
                poem_id=pid,
                match=True,
                confidence=confidence,
                matched_description_aspects=aspects,
                evidence_lines=list(dict.fromkeys(evidence))[:6],
                explanation=" ".join(notes),
                explanation_lang=query_lang,
            )
        )
    verdicts.sort(key=lambda v: -v.confidence)
    # Keep clear matches; if everything is weak, still return the best few.
    strong = [v for v in verdicts if v.confidence >= 0.5]
    return (strong or verdicts)[:limit]


def _chunk(items: list, size: int, max_chunks: int) -> list[list]:
    size = max(size, -(-len(items) // max_chunks))
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _retrieve(
    client, settings, intent, candidates, by_id, vector_index, usage, emit
) -> list[Poem]:
    """Layer 1: narrow candidates to ~vec_topk. Vector mode embeds the English
    intent and ranks by cosine; scout mode falls back to a digest-reading swarm."""
    query_text = intent.get("intent_summary") or ""
    use_vector = settings.retrieval_mode == "vector" and vector_index is not None and query_text
    if use_vector:
        from .vectors import embed_query

        try:
            qvec = await asyncio.to_thread(embed_query, settings, query_text)
            ranked = vector_index.search(qvec, settings.vec_topk)
            return [by_id[pid] for pid, _ in ranked if pid in by_id]
        except Exception as e:
            await emit({"type": "error", "stage": "retrieve", "message": str(e)})
    # scout fallback
    scout_batches = _chunk(candidates, settings.scout_batch_size, settings.swarm_max_agents)
    await emit({"type": "swarm_dispatched", "stage": 1, "agents": len(scout_batches),
                "batch_size": len(scout_batches[0])})
    scored: list[tuple[str, float]] = []
    sem = asyncio.Semaphore(settings.swarm_concurrency)

    async def run(idx, batch):
        async with sem:
            await emit({"type": "agent_start", "stage": 1, "agent": idx, "size": len(batch)})
            try:
                found = await scout_batch(client, settings, query_text, batch, usage)
            except Exception:
                found = []
            scored.extend(found)
            await emit({"type": "agent_done", "stage": 1, "agent": idx, "hits": len(found)})

    await asyncio.gather(*(run(i, b) for i, b in enumerate(scout_batches)))
    scored.sort(key=lambda t: -t[1])
    return [by_id[pid] for pid, _ in scored[: settings.vec_topk] if pid in by_id]


async def search_stream(
    client: AsyncOpenAI,
    settings: Settings,
    retriever: Retriever,
    poems: list[Poem],
    description: str,
    fragment_index=None,
    vector_index=None,
) -> AsyncIterator[dict]:
    """Funnel pipeline: retrieve (vector/scout) → trim → dynamic expert swarm →
    aggregate. Streams events for the live UI.

    Events: start, understanding, criteria, candidates, retrieved, fragment_hits,
    shortlist, swarm_dispatched(stage=2), agent_start, agent_done, verdict, done, error.
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def emit(ev: dict) -> None:
        await queue.put(ev)

    async def pipeline() -> None:
        await emit({"type": "start", "description": description})
        usage: dict = {}
        try:
            intent = await understand_query(client, settings, description, usage)
        except Exception as e:
            await emit({"type": "error", "stage": "understanding", "message": str(e)})
            intent = {"intent_summary": description, "query_lang": "", "criteria": [],
                      "fragments": [], "lang_hint": None}
        query_lang = intent.get("query_lang") or ""
        criteria = intent.get("criteria") or [
            {"aspect": intent.get("intent_summary", description), "kind": "imagery", "weight": 1.0}
        ]
        await emit({"type": "understanding", "intent": intent})
        await emit({"type": "criteria", "criteria": criteria})

        candidates = retriever.retrieve(intent, poems)
        by_id = {p.id: p for p in candidates}
        await emit({"type": "candidates", "count": len(candidates)})
        if not candidates:
            await emit({"type": "done", "matched": 0, "usage": usage})
            return

        retrieved = await _retrieve(
            client, settings, intent, candidates, by_id, vector_index, usage, emit
        )
        await emit({"type": "retrieved", "count": len(retrieved)})

        # Exact-fragment channel: quoted lines go straight in, ahead of vector hits.
        forced: list[Poem] = []
        if fragment_index is not None and intent.get("fragments"):
            hit_ids = set(fragment_index.find(intent["fragments"]))
            forced = [by_id[i] for i in hit_ids if i in by_id]
            if forced:
                await emit({"type": "fragment_hits", "count": len(forced)})
        forced_ids = {p.id for p in forced}

        shortlist = forced + [p for p in retrieved if p.id not in forced_ids]
        shortlist = shortlist[: settings.trim_topn]
        cards = [
            {"id": p.id, "title": p.title, "author": p.author, "language": p.language,
             "preview": "\n".join(p.full_text.splitlines()[:6])}
            for p in shortlist
        ]
        await emit({"type": "shortlist", "count": len(shortlist), "cards": cards})
        if not shortlist:
            await emit({"type": "done", "matched": 0, "usage": usage})
            return

        # Layer 3: one expert per (criterion × poem-batch), all concurrent.
        batches = _chunk(shortlist, settings.expert_batch, settings.swarm_max_agents)
        jobs = [(c, b) for c in criteria for b in batches]
        await emit({"type": "swarm_dispatched", "stage": 2, "agents": len(jobs),
                    "batch_size": settings.expert_batch})
        sem = asyncio.Semaphore(settings.swarm_concurrency)
        per_criterion: list[dict[str, dict]] = [{} for _ in criteria]
        crit_index = {id(c): i for i, c in enumerate(criteria)}

        async def run_expert(idx, criterion, batch):
            async with sem:
                await emit({"type": "agent_start", "stage": 2, "agent": idx,
                            "criterion": criterion.get("aspect", ""),
                            "kind": criterion.get("kind", ""), "size": len(batch)})
                try:
                    hits = await expert_batch(client, settings, criterion, query_lang, batch, usage)
                except Exception:
                    hits = {}
                per_criterion[crit_index[id(criterion)]].update(hits)
                for pid, h in hits.items():
                    await emit({"type": "expert_hit", "agent": idx, "poem_id": pid,
                                "kind": criterion.get("kind", ""),
                                "aspect": criterion.get("aspect", ""),
                                "evidence": h["evidence_lines"]})
                await emit({"type": "agent_done", "stage": 2, "agent": idx, "hits": len(hits)})

        await asyncio.gather(*(run_expert(i, c, b) for i, (c, b) in enumerate(jobs)))

        verdicts = aggregate(shortlist, criteria, per_criterion, query_lang)
        for v in verdicts:
            await emit({"type": "verdict", "verdict": v.model_dump(),
                        "poem": by_id[v.poem_id].model_dump()})
        await emit({"type": "done", "matched": len(verdicts), "usage": usage})

    task = asyncio.create_task(pipeline())
    while True:
        event = await queue.get()
        yield event
        if event["type"] == "done":
            break
    await task
