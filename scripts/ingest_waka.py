"""Ingest classical Japanese waka from the Hachidaishu (八代集) — the 8 imperial
waka anthologies (Kokinshu … Shinkokinshu, 10th–13th c.), ~9,500 poems.

Source: yamagen/hachidaishu — a per-token morphological dataset. We reconstruct
each poem by grouping rows on (Anthology, Poem) and concatenating the Surface
field in order. Only the public-domain poem text is used (not the annotations).
Writes data/ja-hachidaishu.jsonl.

Usage: python scripts/ingest_waka.py
"""

import json
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/yamagen/hachidaishu/main/hachidaishu.jsonl"
REPO = "https://github.com/yamagen/hachidaishu"
OUT = Path("data/ja-hachidaishu.jsonl")
ANTHOLOGY_JP = {
    "Kokinshu": "古今集", "Gosenshu": "後撰集", "Shuishu": "拾遺集",
    "Goshuishu": "後拾遺集", "Kinyoshu": "金葉集", "Shikashu": "詞花集",
    "Senzaishu": "千載集", "Shinkokinshu": "新古今集",
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    poems: dict[tuple[str, str], list[str]] = {}
    order: list[tuple[str, str]] = []
    req = urllib.request.Request(URL, headers={"User-Agent": "PoemFerry/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            d = json.loads(line)
            key = (d.get("Anthology", "?"), str(d.get("Poem", "?")))
            if key not in poems:
                poems[key] = []
                order.append(key)
            poems[key].append(d.get("Surface", ""))

    written = 0
    with OUT.open("w", encoding="utf-8") as f:
        for anth, num in order:
            text = "".join(poems[(anth, num)]).strip()
            if len(text) < 4:
                continue
            anth_jp = ANTHOLOGY_JP.get(anth, anth)
            poem = {
                "id": f"ja-hachidai-{anth}-{num}",
                "title": f"{anth_jp} 第{num}首",
                "author": None,
                "language": "ja",
                "era": "平安〜鎌倉",
                "full_text": text,
                "source_name": f"八代集 · {anth_jp}",
                "source_url": REPO,
                "license": "Public Domain (10th–13th c. waka)",
            }
            f.write(json.dumps(poem, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} waka to {OUT}")


if __name__ == "__main__":
    main()
