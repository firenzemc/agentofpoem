"""Offline: embed every poem's composite retrieval key into data/embeddings.npz.

Uses the cloud embedding endpoint (EMBED_* env). The key is gist + themes +
title/author — English-dominant so a Chinese query (embedded the same way)
matches across languages. Run after scripts/enrich.py. Idempotent: overwrites
the npz. Poems with no embedding row are simply absent from the index.

Usage: EMBED_API_KEY=... python scripts/build_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poemferry.config import load_settings  # noqa: E402
from poemferry.corpus import load_poems  # noqa: E402
from poemferry.vectors import embed_texts, retrieval_key, save_index  # noqa: E402


def main() -> None:
    settings = load_settings()
    if not settings.embed_api_key:
        sys.exit("EMBED_API_KEY is not set")
    poems = load_poems(settings.poems_path)
    print(f"embedding {len(poems)} poems with {settings.embed_model} ...")
    keys = [retrieval_key(p) for p in poems]
    matrix = embed_texts(settings, keys)
    save_index([p.id for p in poems], matrix)
    print(f"wrote data/embeddings.npz  shape={matrix.shape}")


if __name__ == "__main__":
    main()
