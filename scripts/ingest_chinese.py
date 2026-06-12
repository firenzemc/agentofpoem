"""Ingest Chinese classical poetry from chinese-poetry (MIT).

Collections: 唐诗三百首 (Tang, 366) and 宋词三百首 (Song ci, 280).
Public-domain classical poems in traditional/simplified Chinese, kept in
their original language. Writes data/zh-classics.jsonl.

Usage: python scripts/ingest_chinese.py
"""

import json
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master"
COLLECTIONS = [
    {
        "url": f"{REPO}/%E5%85%A8%E5%94%90%E8%AF%97/%E5%94%90%E8%AF%97%E4%B8%89%E7%99%BE%E9%A6%96.json",
        "prefix": "zh-tang300",
        "era": "唐",
        "source_name": "chinese-poetry · 唐诗三百首",
    },
    {
        "url": f"{REPO}/%E5%AE%8B%E8%AF%8D/%E5%AE%8B%E8%AF%8D%E4%B8%89%E7%99%BE%E9%A6%96.json",
        "prefix": "zh-songci300",
        "era": "宋",
        "source_name": "chinese-poetry · 宋词三百首",
    },
]
OUT = Path("data/zh-classics.jsonl")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8") as f:
        for col in COLLECTIONS:
            with urllib.request.urlopen(col["url"], timeout=60) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            for n, item in enumerate(raw, 1):
                paragraphs = item.get("paragraphs") or []
                if not paragraphs:
                    continue
                # Song ci have a tune name (rhythmic) instead of a title.
                title = item.get("title") or item.get("rhythmic")
                poem = {
                    "id": f"{col['prefix']}-{n:04d}",
                    "title": title,
                    "author": item.get("author"),
                    "language": "zh",
                    "era": col["era"],
                    "full_text": "\n".join(paragraphs),
                    "source_name": col["source_name"],
                    "source_url": col["url"],
                    "license": "MIT",
                }
                f.write(json.dumps(poem, ensure_ascii=False) + "\n")
                written += 1
            print(f"  {col['source_name']}: total {written}")

    print(f"wrote {written} poems to {OUT}")


if __name__ == "__main__":
    main()
