import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from .config import load_settings
from .corpus import load_poems
from .fragments import FragmentIndex
from .lexical import LexicalIndex
from .llm import make_client
from .rate_limit import RateLimiter
from .retriever import NaiveRetriever
from .swarm import search_stream
from .vectors import DOC_FILE, VectorIndex

STATIC_DIR = Path(__file__).parent / "static"
RETRIEVAL_MODES = {"image", "keyword", "concept", "hybrid", "scan", "exact"}
# scan reads hundreds of full texts per query (~10x the funnel cost) — the public
# profile drops it so an anonymous request can't trigger that spend.
PUBLIC_MODES = RETRIEVAL_MODES - {"scan"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.client = make_client(settings)
    app.state.poems = load_poems(settings.poems_path)
    app.state.languages = {p.language for p in app.state.poems}
    app.state.retriever = NaiveRetriever()
    app.state.fragment_index = FragmentIndex(app.state.poems)
    app.state.doc_index = VectorIndex.load(DOC_FILE)
    app.state.lexical_index = LexicalIndex(app.state.poems)
    # public → light theme, internal → dark; injected once so there's no flash.
    is_public = settings.profile == "public"
    theme = "light" if is_public else "dark"
    app.state.allowed_modes = PUBLIC_MODES if is_public else RETRIEVAL_MODES
    app.state.limiter = (
        RateLimiter(
            [(300, settings.rl_5min), (3600, settings.rl_hour), (86400, settings.rl_day)],
            settings.rl_global_day,
        )
        if is_public
        else None
    )
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app.state.index_html = html.replace('data-theme="dark"', f'data-theme="{theme}"', 1)
    yield


app = FastAPI(title="PoemFerry", lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(app.state.index_html)


@app.get("/health")
async def health() -> dict:
    # FastAPI's lifespan blocks request serving until startup (index build) is
    # done, so a plain 200 here already means the app is ready — the signal
    # vinyard's deploy gate waits on.
    return {"status": "ok"}


@app.get("/api/info")
async def info() -> dict:
    s = app.state.settings
    return {
        "poem_count": len(app.state.poems),
        "model": s.deepseek_model,
        "has_key": bool(s.deepseek_api_key),
        "languages": sorted({p.language for p in app.state.poems}),
        "retrieval": s.retrieval_mode,
        "modes": sorted(app.state.allowed_modes),
        "profile": s.profile,
        "doc_indexed": len(app.state.doc_index.ids) if app.state.doc_index else 0,
    }


@app.get("/api/sources")
async def sources() -> dict:
    """Acknowledgments: every corpus source with its poem count, link and license,
    auto-generated from per-poem metadata (license-compliant attribution)."""
    agg: dict[str, dict] = {}
    for p in app.state.poems:
        s = agg.setdefault(p.source_name, {"count": 0, "url": p.source_url, "license": p.license,
                                            "languages": set()})
        s["count"] += 1
        s["languages"].add(p.language)
    out = [{"source": k, "count": v["count"], "url": v["url"], "license": v["license"],
            "languages": sorted(v["languages"])}
           for k, v in agg.items()]
    out.sort(key=lambda d: -d["count"])
    return {"sources": out, "total": len(app.state.poems)}


@app.get("/api/corpus")
async def corpus() -> dict:
    """Ordered poem ids + language, for the funnel dot-grid visualization."""
    return {"ids": [p.id for p in app.state.poems],
            "langs": [p.language for p in app.state.poems]}


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


def _client_ip(request: Request) -> str:
    # behind Caddy/Cloudflare the real client is the first X-Forwarded-For hop
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _sse(events: list[dict]) -> StreamingResponse:
    async def gen():
        for ev in events:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/search")
async def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    retrieval: str | None = Query(None),
    lang: str | None = Query(None),
) -> StreamingResponse:
    if app.state.limiter is not None:
        retry = app.state.limiter.check(_client_ip(request), time.time())
        if retry is not None:
            mins = max(1, round(retry / 60))
            return _sse([
                {"type": "error", "stage": "ratelimit",
                 "message": f"检索太频繁了，约 {mins} 分钟后再试。"},
                {"type": "done", "matched": 0},
            ])
    mode = retrieval if retrieval in app.state.allowed_modes else None
    lang_filter = lang if lang in app.state.languages else None

    async def event_gen():
        stream = search_stream(
            app.state.client,
            app.state.settings,
            app.state.retriever,
            app.state.poems,
            q,
            fragment_index=app.state.fragment_index,
            doc_index=app.state.doc_index,
            lexical_index=app.state.lexical_index,
            retrieval_mode=mode,
            lang_filter=lang_filter,
        )
        async for event in stream:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
