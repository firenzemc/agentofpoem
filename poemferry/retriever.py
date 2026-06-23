from typing import Protocol

from .models import Poem


class Retriever(Protocol):
    """Coarse pre-filter that narrows the corpus to swarm candidates.

    Increment 1 ships NaiveRetriever (small corpus, scan all). The embedding
    pre-filter (pgvector) plugs in here later without touching the swarm.
    """

    def retrieve(self, intent: dict, poems: list[Poem]) -> list[Poem]: ...


class NaiveRetriever:
    """Return every poem, optionally restricted to languages the query hints at.

    Viable only while the corpus is small enough for the swarm to read in full.
    """

    def retrieve(self, intent: dict, poems: list[Poem]) -> list[Poem]:
        lang_hint = intent.get("lang_hint")
        if lang_hint:
            wanted = {code.lower() for code in lang_hint}
            filtered = [p for p in poems if p.language.lower() in wanted]
            if filtered:
                return filtered
        return poems
