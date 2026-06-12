"""Ingest Spanish public-domain sonnets from DISCO.

Source: linhd-postdata/disco — Diachronic Spanish Sonnet Corpus, 15th–19th
century, 4,087 sonnets (canonical and minor authors, Spain + Latin America).
Texts from Biblioteca Virtual Miguel de Cervantes / Wikisource; all authors
pre-20th century, so the poems are public domain. Shallow-clones the repo and
parses txt/<period>/per-sonnet/ files named "Author__id~~Title__num.txt".
Writes data/es-disco.jsonl.

Usage: python scripts/ingest_spanish.py
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/linhd-postdata/disco"
SOURCE_NAME = "DISCO · Diachronic Spanish Sonnet Corpus"
OUT = Path("data/es-disco.jsonl")
PERIODS = {"15th-17th": "Siglo de Oro", "18th": "s. XVIII", "19th": "s. XIX"}

FILENAME_RE = re.compile(r"^(?P<author>.+?)__\w+~~(?P<title>.+?)__(?P<num>\d+)\.txt$")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO, tmp],
            check=True,
            capture_output=True,
        )
        with OUT.open("w", encoding="utf-8") as f:
            for period, era in PERIODS.items():
                folder = Path(tmp) / "txt" / period / "per-sonnet"
                count = 0
                for path in sorted(folder.glob("*.txt")):
                    m = FILENAME_RE.match(path.name)
                    if not m:
                        continue
                    text = path.read_text(encoding="utf-8").strip()
                    if not text:
                        continue
                    author = m.group("author").replace("_", " ").strip()
                    title = m.group("title").replace("_", " ").strip()
                    # "Surname, Forename" → "Forename Surname"
                    if "," in author:
                        surname, _, forename = author.partition(",")
                        author = f"{forename.strip()} {surname.strip()}".strip()
                    poem = {
                        "id": f"es-disco-{m.group('num')}",
                        "title": title or None,
                        "author": author or None,
                        "language": "es",
                        "era": era,
                        "full_text": text,
                        "source_name": SOURCE_NAME,
                        "source_url": REPO,
                        "license": "CC-BY 4.0 (DISCO); underlying text Public Domain",
                    }
                    f.write(json.dumps(poem, ensure_ascii=False) + "\n")
                    count += 1
                    written += 1
                print(f"  {period} ({era}): {count}  (total {written})")

    print(f"wrote {written} poems to {OUT}")


if __name__ == "__main__":
    main()
