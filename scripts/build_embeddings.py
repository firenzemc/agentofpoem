"""Offline: build the retrieval indices for the A/B-testable funnel modes.

- embeddings_doc.npz : one vector per poem over the English key
  (gist + themes + images + title) — semantic / cross-lingual, compressed.
- embeddings_line.npz : one vector per representative line, mapped to its parent
  poem — minimal-compression, captures incidental imagery.

Run after scripts/enrich.py. Usage: python scripts/build_embeddings.py [doc|line|all]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poemferry.config import load_settings  # noqa: E402
from poemferry.corpus import load_poems  # noqa: E402
from poemferry.vectors import (  # noqa: E402
    DOC_FILE, LINE_FILE, embed_texts, poem_lines, retrieval_key, save_index,
)


def build_doc(settings, poems) -> None:
    print(f"[doc] embedding {len(poems)} poem keys ...")
    matrix = embed_texts(settings, [retrieval_key(p) for p in poems])
    save_index([p.id for p in poems], matrix, DOC_FILE)
    print(f"[doc] wrote {DOC_FILE} {matrix.shape}")


def build_line(settings, poems) -> None:
    ids: list[str] = []
    texts: list[str] = []
    for p in poems:
        for ln in poem_lines(p):
            ids.append(p.id)
            texts.append(ln)
    print(f"[line] embedding {len(texts)} lines from {len(poems)} poems ...")
    matrix = embed_texts(settings, texts)
    save_index(ids, matrix, LINE_FILE)
    print(f"[line] wrote {LINE_FILE} {matrix.shape}")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    settings = load_settings()
    if not settings.embed_api_key:
        sys.exit("EMBED_API_KEY is not set")
    poems = load_poems(settings.poems_path)
    if which in ("doc", "all"):
        build_doc(settings, poems)
    if which in ("line", "all"):
        build_line(settings, poems)


if __name__ == "__main__":
    main()
