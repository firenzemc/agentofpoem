# PoemFerry (诗渡)

Multilingual public-domain poetry semantic search. Describe a poem in any
language — an image, a mood, a half-remembered line — and a swarm of cheap LLM
agents finds matching poems across a multilingual corpus, returns each one in
its **original language (never translated)**, and explains how it matched.

## How it works

A funnel pipeline (`poemferry/swarm.py`):

1. **Understand** — one LLM call turns the query into an English intent,
   independently-checkable criteria, any quoted fragments, and a cross-lingual
   concept expansion (河 → río / Fluss / 江 / 川).
2. **Retrieve** — weighted Reciprocal Rank Fusion over complementary channels:
   - `image` — semantic vectors over English gists (cross-lingual)
   - `keyword` — idf lexical overlap on raw text (same-language surface detail)
   - `concept` — lexical over the concept expansion (cross-lingual bridge)
   - plus an exact-fragment channel for remembered lines
3. **Judge** — a dynamic swarm of cheap agents (DeepSeek v4-flash) verifies each
   criterion against the candidates and writes the explanation in the query
   language. Results stream over SSE.

A `scan` mode trades the funnel's top-N cut for deep full-text reading in
retrieval order, streaming matches and stopping when satisfied.

Poems are public-domain only; every poem carries its source name, URL, and
license.

## Stack

- Python 3.11+ (uv), FastAPI + uvicorn, vanilla JS / SSE frontend
- DeepSeek (agent swarm) and Cohere embeddings, via OpenAI-compatible APIs

## Run

```sh
uv sync
cp .env.example .env          # add your API keys
uv run uvicorn poemferry.app:app --reload
```

Build the corpus with the `scripts/ingest_*.py` scripts. Corpus files and
embedding indices are gitignored — the repo ships the scripts, not the database.

## Develop

```sh
uv run pytest
uv run ruff check poemferry scripts tests
```
