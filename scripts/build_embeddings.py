"""Offline: build the doc retrieval index (one vector per poem over the English
gist + themes + images + title). Run after scripts/enrich.py.

Usage: python scripts/build_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poemferry.config import load_settings  # noqa: E402
from poemferry.corpus import load_poems  # noqa: E402
from poemferry.vectors import DOC_FILE, embed_texts, retrieval_key, save_index  # noqa: E402


def build_doc(settings, poems) -> None:
    print(f"[doc] embedding {len(poems)} poem keys ...")
    matrix = embed_texts(settings, [retrieval_key(p) for p in poems])
    save_index([p.id for p in poems], matrix, DOC_FILE)
    print(f"[doc] wrote {DOC_FILE} {matrix.shape}")


def main() -> None:
    settings = load_settings()
    if not settings.embed_api_key:
        sys.exit("EMBED_API_KEY is not set")
    build_doc(settings, load_poems(settings.poems_path))


if __name__ == "__main__":
    main()
