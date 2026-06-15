import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse

from .config import load_settings
from .corpus import load_poems
from .fragments import FragmentIndex
from .llm import make_client
from .retriever import NaiveRetriever
from .swarm import search_stream
from .vectors import VectorIndex

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.client = make_client(settings)
    app.state.poems = load_poems(settings.poems_path)
    app.state.retriever = NaiveRetriever()
    app.state.fragment_index = FragmentIndex(app.state.poems)
    app.state.vector_index = VectorIndex.load() if settings.retrieval_mode == "vector" else None
    yield


app = FastAPI(title="PoemFerry", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
async def info() -> dict:
    s = app.state.settings
    return {
        "poem_count": len(app.state.poems),
        "model": s.deepseek_model,
        "has_key": bool(s.deepseek_api_key),
        "languages": sorted({p.language for p in app.state.poems}),
        "retrieval": s.retrieval_mode if app.state.vector_index else "scout",
        "indexed": len(app.state.vector_index.ids) if app.state.vector_index else 0,
    }


@app.get("/api/browse")
async def browse(
    language: str | None = None,
    author: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict:
    poems = app.state.poems
    if language:
        poems = [p for p in poems if p.language == language]
    authors: dict[str, int] = {}
    for p in poems:
        if p.author:
            authors[p.author] = authors.get(p.author, 0) + 1
    if author:
        poems = [p for p in poems if p.author and author.lower() in p.author.lower()]
    start = (page - 1) * page_size
    return {
        "total": len(poems),
        "page": page,
        "page_size": page_size,
        "poems": [p.model_dump() for p in poems[start : start + page_size]],
        "authors": sorted(authors.items(), key=lambda kv: -kv[1])[:30],
    }


@app.get("/api/sample")
async def sample(n: int = Query(240, ge=1, le=800)) -> dict:
    """Poem lines for the digital-rain animation. Deterministic stride sample
    across the corpus (no RNG needed) for language variety."""
    poems = app.state.poems
    step = max(1, len(poems) // n)
    lines: list[str] = []
    for p in poems[::step]:
        for ln in p.full_text.splitlines():
            ln = ln.strip()
            if 2 <= len(ln) <= 32:
                lines.append(ln)
                break
    return {"lines": lines[:n]}


@app.get("/api/search")
async def search(q: str = Query(..., min_length=1)) -> StreamingResponse:
    async def event_gen():
        stream = search_stream(
            app.state.client,
            app.state.settings,
            app.state.retriever,
            app.state.poems,
            q,
            fragment_index=app.state.fragment_index,
            vector_index=app.state.vector_index,
        )
        async for event in stream:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
