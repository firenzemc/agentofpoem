from poemferry.fragments import FragmentIndex, normalize
from poemferry.models import Poem

CHANGHENGE = Poem(
    id="zh-tang300-0178", title="長恨歌", author="白居易", language="zh",
    full_text="漢皇重色思傾國，御宇多年求不得。\n在天願作比翼鳥，在地願爲連理枝。\n天長地久有時盡，此恨綿綿無絕期。",
    source_name="s", license="MIT",
)
OTHER = Poem(
    id="p2", title="B", author="Y", language="fr",
    full_text="Demain, dès l'aube, à l'heure où blanchit la campagne,\nJe partirai.",
    source_name="s", license="PD",
)


def test_normalize_folds_punctuation_and_traditional():
    assert normalize("在天願作比翼鳥，") == normalize("在天 愿作比翼鸟")


def test_simplified_quote_finds_traditional_poem():
    idx = FragmentIndex([CHANGHENGE, OTHER])
    assert idx.find(["在天愿作比翼鸟"]) == ["zh-tang300-0178"]


def test_accented_quote_matches_case_insensitively():
    idx = FragmentIndex([CHANGHENGE, OTHER])
    assert idx.find(["Demain, dès l'aube"]) == ["p2"]


def test_short_or_unknown_fragments_match_nothing():
    idx = FragmentIndex([CHANGHENGE, OTHER])
    assert idx.find(["月"]) == []  # too short after folding
    assert idx.find(["不存在的句子也不会命中"]) == []
