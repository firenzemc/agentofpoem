"""Ingest poetry-only works from globis-university/aozorabunko-clean.

Aozora Bunko is mostly prose (novels, essays). This script keeps POETRY ONLY,
using the dataset's NDC classification (meta.分類番号) as the authoritative
signal: NDC 911 is Japanese poetry (waka / haiku / senryu / modern shi), and
NDC K911 is children's poetry. Because the records carry only the bare "NDC 911"
code (no .1/.3/.5 decimals), two extra conservative cuts remove the prose that
NDC 911 also covers (911.0 poetics/criticism):

  1. NDC: keep works whose FIRST classification code is 911 or K911.
  2. Title: drop works whose title contains criticism/essay markers
     (論 / 評論 / 研究 / 雑記 / about-X phrases, etc.).
  3. Structure: drop works whose median line length looks like prose (> 70
     chars). Real Japanese verse is short-lined (haiku ~17, waka ~32, free
     verse broken into lines); essays run hundreds of chars per line.

The bias is conservative on purpose: it is better to miss a few dense prose-poems
(e.g. 聖三稜玻璃) than to pollute a poetry corpus with prose.

Granularity: one Aozora work = one record. A 歌集/句集 is many tanka/haiku in a
single text blob, so full_text can be large (some ~20k chars). We do NOT split
collections — there is no reliable cross-work delimiter and splitting risks
fragmenting poems or leaking section prose.

Provenance: every work has meta.作品著作権フラグ == "なし" (out of copyright in
Japan), so all kept records are public domain. source_url is the per-work Aozora
library-card URL (meta.図書カードURL). The globis dataset itself is CC BY 4.0.

The dataset ships as 2 auto-converted parquet shards (~365 MB total); we stream
them from the HF parquet API. Writes data/ja-aozora.jsonl.

Usage: uv run --with pyarrow python scripts/ingest_aozora.py
"""

import json
import statistics
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

PARQUET_URLS = [
    "https://huggingface.co/api/datasets/globis-university/aozorabunko-clean/parquet/default/train/0.parquet",
    "https://huggingface.co/api/datasets/globis-university/aozorabunko-clean/parquet/default/train/1.parquet",
]
SOURCE_NAME = "青空文庫"
LICENSE = "Public Domain (Aozora Bunko); dataset CC BY 4.0"
OUT = Path("data/ja-aozora.jsonl")

# Criticism / essay markers that appear in NDC 911 titles but denote prose.
PROSE_TITLE_MARKERS = (
    "論", "評論", "雑記", "研究", "評伝", "評釈", "概論",
    "について", "に就て", "に就いて", "序説", "小論", "随筆", "史論", "解説",
)
MAX_PROSE_MEDIAN_LINE = 70  # median line length above this looks like prose


def ndc_codes(meta: dict) -> list[str]:
    """Return the NDC codes of a work, e.g. 'NDC 911 914' -> ['911', '914']."""
    return (meta.get("分類番号") or "").replace("NDC", "").split()


def is_poetry(text: str, meta: dict) -> bool:
    codes = ndc_codes(meta)
    # 1. Primary classification must be Japanese poetry (911) or children's (K911).
    if not codes or codes[0].lstrip("K") != "911":
        return False
    # 2. Drop criticism/essay titles.
    title = meta.get("作品名") or ""
    if any(marker in title for marker in PROSE_TITLE_MARKERS):
        return False
    # 3. Drop prose-shaped text (long median line length).
    line_lengths = [len(line.strip()) for line in text.splitlines() if line.strip()]
    if not line_lengths:
        return False
    if statistics.median(line_lengths) > MAX_PROSE_MEDIAN_LINE:
        return False
    return True


def author_name(meta: dict) -> str | None:
    name = (meta.get("姓") or "") + (meta.get("名") or "")
    return name or None


def iter_records():
    for url in PARQUET_URLS:
        req = urllib.request.Request(url, headers={"User-Agent": "PoemFerry/0.1"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        tmp = Path(f"/tmp/aozora-{Path(url).parent.name}-{Path(url).name}")
        tmp.write_bytes(data)
        try:
            for row in pq.read_table(tmp).to_pylist():
                yield row
        finally:
            tmp.unlink(missing_ok=True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8") as f:
        for row in iter_records():
            meta = row.get("meta") or {}
            text = (row.get("text") or "").strip()
            if not text or not is_poetry(text, meta):
                continue
            poem = {
                "id": f"ja-aozora-{meta.get('作品ID')}",
                "title": meta.get("作品名") or None,
                "author": author_name(meta),
                "language": "ja",
                "era": None,
                "full_text": text,
                "source_name": SOURCE_NAME,
                "source_url": meta.get("図書カードURL") or None,
                "license": LICENSE,
            }
            f.write(json.dumps(poem, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} poems to {OUT}")


if __name__ == "__main__":
    main()
