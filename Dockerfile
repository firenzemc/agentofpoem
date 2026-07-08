FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Bake the full venv at build time so the container starts with zero package
# work and zero network. [tool.uv] package=false → deps only; --frozen pins to
# uv.lock. (Dependencies first for layer caching.)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY poemferry ./poemferry

# The corpus + embedding index (data/, ~1.5GB, gitignored) is NOT baked into the
# image: vinyard builds from git (where data/ is absent) and never copies images
# host-to-host. Mount it as a named volume at /app/data (see vinyard.toml).

# Run the baked venv directly — no `uv run` (which re-syncs and pulls dev deps at
# startup). `python -m uvicorn` from /app puts poemferry on sys.path.
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
# Shell form so $PORT expands (default 8000); `exec` hands SIGTERM straight to
# uvicorn so `docker stop` shuts down gracefully instead of killing the shell.
CMD exec python -m uvicorn poemferry.app:app --host 0.0.0.0 --port ${PORT:-8000}
