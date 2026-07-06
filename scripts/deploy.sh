#!/usr/bin/env bash
# Deploy PoemFerry on Docker via Colima.
# Port 8077 is published on 0.0.0.0, so Colima forwards it to the host on all
# interfaces and the app is reachable on the tailnet at http://<tailscale-ip>:8077.
set -euo pipefail
cd "$(dirname "$0")/.."

# 12GiB VM: the full float16 doc index (1.18GB) + lexical/fragment indices over
# ~385k poems put steady RSS near 6GB, and a query spikes it higher — 6g/7g caps
# OOM-killed it (exit 137). Colima allocates lazily, so the higher ceiling only
# costs host RAM when actually used.
colima start --cpu 4 --memory 12 2>/dev/null || colima start 2>/dev/null || true

# Docker Desktop left `credsStore: osxkeychain` in ~/.docker/config.json, but that
# helper is gone under Colima and breaks even anonymous public-image pulls. Use a
# throwaway config with no cred helper, and talk to the Colima daemon socket
# directly so we don't depend on the docker context metadata either.
export DOCKER_HOST="${DOCKER_HOST:-unix://$HOME/.colima/default/docker.sock}"
DOCKER_CONFIG="$(mktemp -d)"; export DOCKER_CONFIG
printf '{"auths":{}}' > "$DOCKER_CONFIG/config.json"

echo "building poemferry:latest ..."
docker build -t poemferry:latest .

docker rm -f poemferry 2>/dev/null || true
# -m 9g: startup RSS ~6GB, query-time spike (chunked search + swarm batches) tops 7g,
# so give headroom under the 12GiB VM.
docker run -d --name poemferry --env-file .env -m 9g --cpus 4 \
  -p 0.0.0.0:8077:8000 poemferry:latest

echo "waiting for startup (builds opencc indices over the full corpus, ~70s+) ..."
until curl -s --max-time 3 http://127.0.0.1:8077/api/info >/dev/null 2>&1; do sleep 3; done
echo "deployed:"
curl -s http://127.0.0.1:8077/api/info
echo
