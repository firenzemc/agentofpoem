"""Ingest Werneror/Poetry — the broadest public-domain classical Chinese corpus
(~850k poems, 先秦→清), to fill our Ming/Qing gap (our existing 363k is唐宋元-only:
明=0, 清=12). CSV per period; columns 题目/朝代/作者/内容.

Public-domain only: files whose period label contains 当代/近现代/民国 are skipped
(authors may still be in copyright — life+70 → died after 1956). Rows are deduped
by normalized full text (trad/simp folded via opencc) against our existing zh
corpus AND within this run, so overlapping唐宋 poems don't duplicate.

Writes data/zh-werneror.jsonl (text only; enrichment/embedding is a later phase).

Usage: python scripts/ingest_werneror.py
"""

import csv
import hashlib
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poemferry.fragments import normalize  # noqa: E402

BASE = "https://raw.githubusercontent.com/Werneror/Poetry/master"
SOURCE_NAME = "Werneror · 古诗词"
REPO = "https://github.com/Werneror/Poetry"
OUT = Path("data/zh-werneror.jsonl")
EXISTING = [Path("data/zh-collection.jsonl"), Path("data/zh-classics.jsonl")]

# Every CSV in the repo except those touching the modern (copyright) era.
FILES = [
    "秦", "先秦", "汉", "魏晋", "魏晋末南北朝初", "南北朝", "隋", "隋末唐初",
    "唐", "唐末宋初", "辽", "宋_1", "宋_2", "宋_3", "宋_4", "宋末金初", "金",
    "宋末元初", "金末元初", "元", "元末明初", "明_1", "明_2", "明_3", "明_4",
    "明末清初", "清_1", "清_2",
]
# Skipped (modern / copyright): 清末民国初, 清末近现代初, 近现代, 近现代末当代初,
# 民国末当代初, 当代.


def existing_keys() -> set[str]:
    keys: set[str] = set()
    for path in EXISTING:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                keys.add(normalize(json.loads(line)["full_text"]))
    return keys


def fetch_csv(name: str) -> str:
    url = f"{BASE}/{urllib.parse.quote(name)}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "PoemFerry/0.1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            if attempt == 2:
                raise
    return ""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_keys()
    print(f"existing corpus: {len(seen):,} unique normalized texts")

    written = kept = dup = 0
    with OUT.open("w", encoding="utf-8") as out:
        for name in FILES:
            text = fetch_csv(name)
            rows = list(csv.DictReader(io.StringIO(text)))
            for row in rows:
                content = (row.get("内容") or "").strip()
                if not content:
                    continue
                key = normalize(content)
                if not key or key in seen:
                    dup += 1
                    continue
                seen.add(key)
                rec = {
                    "id": "zh-wer-" + hashlib.md5(key.encode()).hexdigest(),
                    "title": (row.get("题目") or "").strip(),
                    "author": (row.get("作者") or "").strip(),
                    "language": "zh",
                    "era": (row.get("朝代") or "").strip(),
                    "full_text": content,
                    "source_name": SOURCE_NAME,
                    "source_url": REPO,
                    "license": "MIT",
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            kept += len(rows)
            print(f"{name}: {len(rows):,} rows | net new so far {written:,}")

    print(f"\ndone: {written:,} net-new poems written to {OUT.name} "
          f"({dup:,} dropped as duplicates/empty out of {kept:,} scanned)")


if __name__ == "__main__":
    main()
