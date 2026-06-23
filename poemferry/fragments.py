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
