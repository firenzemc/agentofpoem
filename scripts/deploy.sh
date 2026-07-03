#!/usr/bin/env bash
# Deploy PoemFerry on Docker via Colima.
# Port 8077 is published on 0.0.0.0, so Colima forwards it to the host on all
# interfaces and the app is reachable on the tailnet at http://<tailscale-ip>:8077.
set -euo pipefail
cd "$(dirname "$0")/.."

colima start 2>/dev/null || true

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
# -m 6g: the in-memory lexical+fragment indices over ~385k poems need ~5GB RSS;
# capped below the 8GiB Colima VM so it can't starve the VM itself.
docker run -d --name poemferry --env-file .env -m 6g --cpus 4 \
  -p 0.0.0.0:8077:8000 poemferry:latest

echo "waiting for startup (builds opencc indices over the full corpus, ~70s+) ..."
until curl -s --max-time 3 http://127.0.0.1:8077/api/info >/dev/null 2>&1; do sleep 3; done
echo "deployed:"
curl -s http://127.0.0.1:8077/api/info
echo
