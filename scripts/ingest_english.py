"""Ingest English public-domain poetry from PoetryDB.

Source: https://poetrydb.org (PoetryDB, public domain). Poems are already
segmented with title/author/lines. Writes data/en-poetrydb.jsonl in the
PoemFerry schema.

Usage: python scripts/ingest_english.py [max_authors]
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://poetrydb.org"
SOURCE_NAME = "PoetryDB"
OUT = Path("data/en-poetrydb.jsonl")


def fetch(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "PoemFerry/0.1 (ingest)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main() -> None:
    max_authors = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    authors = fetch(f"{BASE}/author")["authors"]
    if max_authors:
        authors = authors[:max_authors]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8") as f:
        for author in authors:
            poems = fetch(f"{BASE}/author/{urllib.parse.quote(author)}")
            if not isinstance(poems, list):
                continue
            for n, item in enumerate(poems, 1):
                lines = item.get("lines") or []
                if not lines:
                    continue
                slug = f"{author}-{item.get('title', '')}".lower()
                pid = "en-pdb-" + urllib.parse.quote(slug, safe="")[:48] + f"-{n}"
                poem = {
                    "id": pid,
                    "title": item.get("title"),
                    "author": author,
                    "language": "en",
                    "era": None,
                    "full_text": "\n".join(lines),
                    "source_name": SOURCE_NAME,
                    "source_url": f"{BASE}/author/{urllib.parse.quote(author)}",
                    "license": "Public Domain",
                }
                f.write(json.dumps(poem, ensure_ascii=False) + "\n")
                written += 1
            print(f"  {author}: {len(poems)}  (total {written})")

    print(f"wrote {written} poems to {OUT}")


if __name__ == "__main__":
    main()
