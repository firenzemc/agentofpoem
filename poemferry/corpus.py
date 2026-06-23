import json
from pathlib import Path

from .models import Poem


def load_poems(path: str) -> list[Poem]:
    """Load the poem corpus from a JSONL file, or every *.jsonl in a directory.

    A directory lets each language/source ingest into its own file (e.g.
    data/zh-tang300.jsonl, data/en-poetrydb.jsonl) and have them all loaded.
    """
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    poems: list[Poem] = []
    for f in files:
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    poems.append(Poem(**json.loads(line)))
    return poems
