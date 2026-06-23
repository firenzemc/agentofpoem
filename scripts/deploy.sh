#!/usr/bin/env bash
# Deploy PoemFerry on Apple `container` (replaces docker-compose).
# OrbStack kept stopping / nagging login mid-session, so we run on Apple's
# native `container` CLI instead. It supports -p host-port publishing, so the
# app is reachable on the tailnet at http://<tailscale-ip>:8077.
set -euo pipefail
cd "$(dirname "$0")/.."

container system start 2>/dev/null || true

echo "building poemferry:latest ..."
container build -t poemferry:latest .

container rm -f poemferry 2>/dev/null || true
# -m 8g: the in-memory lexical+fragment indices over ~376k poems need ~5GB RSS.
container run -d --name poemferry --env-file .env -m 8g -c 4 \
  -p 0.0.0.0:8077:8000 poemferry:latest

echo "waiting for startup (builds opencc indices over the full corpus, ~70s+) ..."
until curl -s --max-time 3 http://127.0.0.1:8077/api/info >/dev/null 2>&1; do sleep 3; done
echo "deployed:"
curl -s http://127.0.0.1:8077/api/info
echo
