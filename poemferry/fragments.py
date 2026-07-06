import re

from opencc import OpenCC

from .models import Poem

_t2s = OpenCC("t2s")
_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Fold text for exact-fragment matching: lowercase, drop whitespace and
    punctuation, convert traditional Chinese to simplified (so a simplified
    quote like 在天愿作比翼鸟 matches the stored 在天願作比翼鳥)."""
    return _t2s.convert(_STRIP_RE.sub("", text).lower())


class FragmentIndex:
    """Substring index over normalized poem texts, built once at startup.

    A user-quoted line is matched verbatim (after folding) against every
    poem — zero LLM cost, immune to scout recall misses.
    """

    def __init__(self, poems: list[Poem]):
        self._entries = [(p.id, normalize(p.full_text)) for p in poems]

    def find(self, fragments: list[str]) -> list[str]:
        needles = [n for n in (normalize(f) for f in fragments) if len(n) >= 4]
        if not needles:
            return []
        return [pid for pid, text in self._entries if any(n in text for n in needles)]

    def search_exact(self, query: str, min_len: int = 2) -> list[tuple[str, int]]:
        """Strict word-for-word search: every poem whose normalized full text
        contains the whole query as one contiguous substring, with occurrence
        count, most-hits first. Zero LLM — pure literal matching (trad/simp folded
        so a simplified query still hits a traditional poem)."""
        needle = normalize(query)
        if len(needle) < min_len:
            return []
        hits = [(pid, text.count(needle)) for pid, text in self._entries if needle in text]
        hits.sort(key=lambda pc: -pc[1])
        return hits
