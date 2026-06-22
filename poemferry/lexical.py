import math
import re
import unicodedata

from opencc import OpenCC

from .models import Poem

# Lexical retrieval channel: judgment-free term overlap on the raw text. It
# captures concrete surface details (a query for 河 matches 黄河) that semantic
# embeddings compress away — at the cost of being same-language only.

_t2s = OpenCC("t2s")
_LATIN = re.compile(r"[a-z]{3,}")
_CJK_RUN = re.compile(r"[㐀-鿿]+")
# ultra-common classical-Chinese function characters, skipped as single-char grams
_STOP_CJK = set("的了和与與而之其於于以為为在不無无有也者乎兮即既乃且然則则")


def _fold_latin(t: str) -> str:
    """Strip diacritics and fold ß so accented words tokenize: río→rio, Fluß→fluss,
    été→ete. Without this, [a-z]{3,} silently drops every accented word."""
    t = t.replace("ß", "ss")
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _grams(text: str) -> set[str]:
    t = _t2s.convert(text.lower())
    grams: set[str] = set(_LATIN.findall(_fold_latin(t)))
    for run in _CJK_RUN.findall(t):
        for ch in run:
            if ch not in _STOP_CJK:
                grams.add(ch)
        grams.update(run[i : i + 2] for i in range(len(run) - 1))  # bigrams
    return grams


class LexicalIndex:
    """Inverted-ish term index with idf weighting. Rarer shared terms (the rare
    bigram 黄河) count for more than common single characters (河)."""

    def __init__(self, poems: list[Poem]):
        self.ids = [p.id for p in poems]
        self.doc_grams = [_grams(p.full_text) for p in poems]
        df: dict[str, int] = {}
        for grams in self.doc_grams:
            for g in grams:
                df[g] = df.get(g, 0) + 1
        n = max(1, len(poems))
        self.idf = {g: math.log(1 + n / c) for g, c in df.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = _grams(query)
        if not q:
            return []
        scored: list[tuple[str, float]] = []
        for pid, grams in zip(self.ids, self.doc_grams):
            common = q & grams
            if common:
                scored.append((pid, sum(self.idf.get(g, 0.0) for g in common)))
        scored.sort(key=lambda t: -t[1])
        return scored[:k]
