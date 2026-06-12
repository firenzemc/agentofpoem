FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
COPY uv.lock* ./
RUN uv sync --no-dev

COPY poemferry ./poemferry
COPY data ./data

EXPOSE 8000

# Bind to 0.0.0.0 so the app is reachable from other hosts (e.g. over Tailscale by IP).
CMD ["uv", "run", "uvicorn", "poemferry.app:app", "--host", "0.0.0.0", "--port", "8000"]
