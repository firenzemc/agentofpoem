FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
COPY uv.lock* ./
RUN uv sync --no-dev

COPY poemferry ./poemferry

# The corpus + embedding index (data/, ~1.5GB, gitignored) is NOT baked into the
# image: vinyard builds from git (where data/ is absent) and never copies images
# host-to-host. Mount it as a named volume at /app/data (see vinyard.toml).

EXPOSE 8000

# Shell form so $PORT expands; defaults to 8000 for the legacy Colima deploy.
# Bind 0.0.0.0 so the app is reachable from other hosts (e.g. over Tailscale by IP).
CMD uv run uvicorn poemferry.app:app --host 0.0.0.0 --port ${PORT:-8000}
