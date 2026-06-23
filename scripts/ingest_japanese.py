"""Ingest Japanese classical waka: 小倉百人一首 (100 poems, 13th century).

Source: tomari/ogura-hyakunin-isshu (JSON transcription of the public-domain
classic anthology; faithful transcription of PD text carries no new copyright).
Writes data/ja-hyakunin.jsonl.

Usage: python scripts/ingest_japanese.py
"""

import json
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/tomari/ogura-hyakunin-isshu/master/hyakunin.json"
SOURCE_NAME = "小倉百人一首 (tomari/ogura-hyakunin-isshu)"
OUT = Path("data/ja-hyakunin.jsonl")


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8") as f:
        for item in raw:
            text = item.get("text") or []
            if not text:
                continue
            author = (item.get("author") or {}).get("name")
            poem = {
                "id": f"ja-hyakunin-{item['n']:03d}",
                "title": f"百人一首 第{item['n']}番",
                "author": author,
                "language": "ja",
                "era": "平安〜鎌倉",
                "full_text": "\n".join(text),
                "source_name": SOURCE_NAME,
                "source_url": URL,
                "license": "Public Domain (13th-century anthology)",
            }
            f.write(json.dumps(poem, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} poems to {OUT}")


if __name__ == "__main__":
    main()
