"""Ingest German public-domain poetry from the TextGrid Poetry Corpus.

Source: linhd-postdata/textgrid-poetry — JSON subset of the TextGrid Digital
Library (CC-BY; underlying zeno.org texts are public domain). Per-author files.
We take a curated whitelist of canonical German-language poets, all dead 70+
years, and skip German translations of foreign poets. Writes data/de-textgrid.jsonl.

Usage: python scripts/ingest_german.py [per_author_cap]
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/linhd-postdata/textgrid-poetry/master/json"
SOURCE_NAME = "TextGrid Poetry Corpus (linhd-postdata)"
OUT = Path("data/de-textgrid.jsonl")

# Canonical German-language poets, filename → death year (all life+70 safe).
POETS = {
    "Goethe, Johann Wolfgang": 1832,
    "Schiller, Friedrich": 1805,
    "Heine, Heinrich": 1856,
    "Hölderlin, Friedrich": 1843,
    "Novalis": 1801,
    "Eichendorff, Joseph von": 1857,
    "Mörike, Eduard": 1875,
    "Storm, Theodor": 1888,
    "Droste-Hülshoff, Annette von": 1848,
    "Rilke, Rainer Maria": 1926,
    "Trakl, Georg": 1914,
    "Klopstock, Friedrich Gottlieb": 1803,
    "Brentano, Clemens": 1842,
    "Uhland, Ludwig": 1862,
    "Lenau, Nikolaus": 1850,
    "Fontane, Theodor": 1898,
    "Meyer, Conrad Ferdinand": 1898,
    "Keller, Gottfried": 1890,
    "George, Stefan": 1933,
    "Hofmannsthal, Hugo von": 1929,
    "Morgenstern, Christian": 1914,
    "Claudius, Matthias": 1815,
    "Hebbel, Friedrich": 1863,
    "Chamisso, Adelbert von": 1838,
}


def fetch(author: str):
    url = f"{BASE}/{urllib.parse.quote(author)}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "PoemFerry/0.1 (ingest)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return url, json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8") as f:
        for author, death_yr in POETS.items():
            url, items = fetch(author)
            display = author.split(",")[0] if "," in author else author
            surname, _, forename = author.partition(", ")
            display = f"{forename} {surname}".strip() if forename else surname
            kept = 0
            for n, item in enumerate(items, 1):
                if kept >= cap:
                    break
                stanzas = item.get("text") or []
                lines = [line for stanza in stanzas for line in stanza]
                if not lines or len(lines) > 80:
                    continue
                title = (item.get("title") or "").strip() or None
                poem = {
                    "id": f"de-tg-{urllib.parse.quote(surname.lower(), safe='')}-{n:04d}",
                    "title": title,
                    "author": display,
                    "language": "de",
                    "era": item.get("publicationDate") or None,
                    "full_text": "\n\n".join("\n".join(s) for s in stanzas if s),
                    "source_name": SOURCE_NAME,
                    "source_url": url,
                    "license": "CC-BY (TextGrid); underlying text Public Domain",
                }
                f.write(json.dumps(poem, ensure_ascii=False) + "\n")
                kept += 1
                written += 1
            print(f"  {display} (d.{death_yr}): {kept}  (total {written})")

    print(f"wrote {written} poems to {OUT}")


if __name__ == "__main__":
    main()
