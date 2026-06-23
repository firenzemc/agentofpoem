"""Ingest Italian classic poetry: Dante's Divina Commedia (100 cantos).

Source: fabiovalse/Divina-Commedia-Visualization — structured JSON of the
14th-century poem (public domain; transcription carries no new copyright).
Each canto becomes one poem (~140 lines — a deliberate long-text case).
Writes data/it-dante.jsonl.

Usage: python scripts/ingest_italian.py
"""

import json
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/fabiovalse/"
    "Divina-Commedia-Visualization/master/divina_commedia.json"
)
SOURCE_NAME = "Divina Commedia (fabiovalse structured JSON)"
OUT = Path("data/it-dante.jsonl")


def main() -> None:
    with urllib.request.urlopen(URL, timeout=120) as resp:
        tree = json.loads(resp.read().decode("utf-8"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8") as f:
        for volume in tree["children"]:
            vol_name = volume["name"]
            for canto in volume["children"]:
                lines = [
                    verso["text"]
                    for terzina in canto["children"]
                    for verso in terzina["children"]
                ]
                if not lines:
                    continue
                written += 1
                poem = {
                    "id": f"it-dante-{written:03d}",
                    "title": f"{vol_name} · {canto['name']}",
                    "author": "Dante Alighieri",
                    "language": "it",
                    "era": "1308–1321",
                    "full_text": "\n".join(lines),
                    "source_name": SOURCE_NAME,
                    "source_url": URL,
                    "license": "Public Domain (14th century)",
                }
                f.write(json.dumps(poem, ensure_ascii=False) + "\n")
            print(f"  {vol_name}: total {written}")

    print(f"wrote {written} poems to {OUT}")


if __name__ == "__main__":
    main()
