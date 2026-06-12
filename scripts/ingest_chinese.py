"""Ingest Chinese classical poetry from chinese-poetry (MIT).

Collections: 唐诗三百首, 宋词三百首, 诗经, 楚辞, 纳兰性德词.
Public-domain classical poems kept in their original language.
Writes data/zh-classics.jsonl.

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
    {
        "url": f"{REPO}/%E8%AF%97%E7%BB%8F/shijing.json",
        "prefix": "zh-shijing",
        "era": "先秦",
        "source_name": "chinese-poetry · 诗经",
        "author": "佚名",
    },
    {
        "url": f"{REPO}/%E6%A5%9A%E8%BE%9E/chuci.json",
        "prefix": "zh-chuci",
        "era": "先秦",
        "source_name": "chinese-poetry · 楚辞",
    },
    {
        "url": (
            f"{REPO}/%E7%BA%B3%E5%85%B0%E6%80%A7%E5%BE%B7/"
            "%E7%BA%B3%E5%85%B0%E6%80%A7%E5%BE%B7%E8%AF%97%E9%9B%86.json"
        ),
        "prefix": "zh-nalan",
        "era": "清",
        "source_name": "chinese-poetry · 纳兰性德诗集",
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
            count = 0
            for n, item in enumerate(raw, 1):
                # Field name varies by collection: paragraphs / content / para.
                lines = item.get("paragraphs") or item.get("content") or item.get("para") or []
                if not lines:
                    continue
                # Song ci have a tune name (rhythmic) instead of a title.
                title = item.get("title") or item.get("rhythmic")
                if item.get("chapter"):  # 诗经: 国风·周南·关雎
                    title = f"{item['chapter']}·{item['section']}·{item['title']}"
                poem = {
                    "id": f"{col['prefix']}-{n:04d}",
                    "title": title,
                    "author": item.get("author") or col.get("author"),
                    "language": "zh",
                    "era": col["era"],
                    "full_text": "\n".join(lines),
                    "source_name": col["source_name"],
                    "source_url": col["url"],
                    "license": "MIT",
                }
                f.write(json.dumps(poem, ensure_ascii=False) + "\n")
                count += 1
                written += 1
            print(f"  {col['source_name']}: {count}  (total {written})")

    print(f"wrote {written} poems to {OUT}")


if __name__ == "__main__":
    main()
