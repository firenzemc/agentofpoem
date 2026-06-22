"""Ingest open-chinese/poetry-collection — 370k Chinese poems (MIT, 370 volumes).

The most comprehensive structured Chinese poetry dataset; supersedes our small
zh-classics subset. Schema per poem: {uuid, title, author, content, dynasty,
collection_info}. Captures text only (enrichment/embedding is a later phase).
Writes data/zh-collection.jsonl.

Usage: python scripts/ingest_collection.py [num_volumes]
"""

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://raw.githubusercontent.com/open-chinese/poetry-collection/main/volumes"
N_VOLUMES = 370
SOURCE_NAME = "open-chinese · 诗歌总集"
REPO = "https://github.com/open-chinese/poetry-collection"
OUT = Path("data/zh-collection.jsonl")


def fetch_volume(n: int):
    url = f"{BASE}/volume_{n:03d}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "PoemFerry/0.1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return n, json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                return n, []
    return n, []


def main() -> None:
    nvol = int(sys.argv[1]) if len(sys.argv) > 1 else N_VOLUMES
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=16) as ex:
        for n, poems in ex.map(fetch_volume, range(1, nvol + 1)):
            for item in poems:
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                ci = item.get("collection_info") or {}
                poem = {
                    "id": f"zh-coll-{item.get('uuid')}",
                    "title": item.get("title"),
                    "author": item.get("author"),
                    "language": "zh",
                    "era": item.get("dynasty"),
                    "full_text": content,
                    "source_name": f"{SOURCE_NAME} · {ci.get('collection', '')}".strip(" ·"),
                    "source_url": REPO,
                    "license": "MIT",
                }
                f.write(json.dumps(poem, ensure_ascii=False) + "\n")
                written += 1
            if n % 50 == 0:
                print(f"  volume {n}/{nvol}  ({written} poems)")
    print(f"wrote {written} poems to {OUT}")


if __name__ == "__main__":
    main()
