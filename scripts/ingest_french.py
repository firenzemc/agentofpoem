"""Ingest French public-domain poetry from the manu/french_poetry HF dataset.

Source: poesie-francaise.fr scrape on HuggingFace (manu/french_poetry, 1821
poems). The poet field carries life years, e.g. "Victor Hugo (1802-1885)";
we keep only poets dead before 1956 (life+70), so every kept text is public
domain. Fetched via the HF datasets-server rows API (no parquet dependency).
Writes data/fr-poesie.jsonl.

Usage: python scripts/ingest_french.py
"""

import json
import re
import time
import urllib.request
from pathlib import Path

API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=manu%2Ffrench_poetry&config=default&split=train"
)
SOURCE_NAME = "poesie-francaise.fr (manu/french_poetry)"
OUT = Path("data/fr-poesie.jsonl")
PD_DEATH_CUTOFF = 1956

POET_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<birth>\d{4})-(?P<death>\d{4})\)\s*$")
# The scrape prefixes each text with metadata lines: Poésie/Titre/Poète/Recueil.
HEADER_RE = re.compile(r"^(Poésie|Titre|Poète|Recueil)\s*:.*$", re.MULTILINE)


def fetch(offset: int, length: int = 100):
    url = f"{API}&offset={offset}&length={length}"
    req = urllib.request.Request(url, headers={"User-Agent": "PoemFerry/0.1 (ingest)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def clean_text(text: str) -> str:
    body = HEADER_RE.sub("", text)
    # Drop the trailing attribution line ("Victor Hugo.") the site appends.
    lines = [line.rstrip() for line in body.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    first = fetch(0)
    total = first["num_rows_total"]
    written = skipped = 0
    with OUT.open("w", encoding="utf-8") as f:
        offset = 0
        rows = first["rows"]
        while True:
            for r in rows:
                row = r["row"]
                m = POET_RE.match(row.get("poet") or "")
                if not m or int(m.group("death")) >= PD_DEATH_CUTOFF:
                    skipped += 1
                    continue
                text = clean_text(row.get("text") or "")
                if not text:
                    skipped += 1
                    continue
                death = m.group("death")
                poem = {
                    "id": f"fr-pf-{row['id'][:60]}",
                    "title": row.get("title"),
                    "author": m.group("name"),
                    "language": "fr",
                    "era": f"{m.group('birth')}–{death}",
                    "full_text": text,
                    "source_name": SOURCE_NAME,
                    "source_url": row.get("link"),
                    "license": "Public Domain (author d. " + death + ")",
                }
                f.write(json.dumps(poem, ensure_ascii=False) + "\n")
                written += 1
            offset += len(rows)
            if offset >= total:
                break
            rows = fetch(offset)["rows"]
            print(f"  fetched {offset}/{total}  (kept {written})")

    print(f"wrote {written} poems to {OUT} (skipped {skipped} non-PD/empty)")


if __name__ == "__main__":
    main()
