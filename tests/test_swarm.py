import poemferry.swarm as swarm
from poemferry.config import Settings
from poemferry.models import Poem, Verdict
from poemferry.retriever import NaiveRetriever


def make_settings(**overrides) -> Settings:
    base = dict(
        deepseek_api_key="test",
        deepseek_base_url="https://example.invalid",
        deepseek_model="deepseek-v4-flash",
        swarm_batch_size=2,
        swarm_concurrency=4,
        swarm_max_agents=128,
        scout_batch_size=2,
        shortlist_size=10,
        poems_path="data",
    )
    base.update(overrides)
    return Settings(**base)


POEMS = [
    Poem(id="p1", title="A", author="X", language="zh", full_text="床前明月光",
         source_name="s", license="MIT"),
    Poem(id="p2", title="B", author="Y", language="fr", full_text="Demain, dès l'aube",
         source_name="s", license="PD"),
    Poem(id="p3", title="C", author="Z", language="en", full_text="The woods are lovely",
         source_name="s", license="PD"),
]


def test_naive_retriever_returns_all_without_hint():
    r = NaiveRetriever()
    assert len(r.retrieve({}, POEMS)) == 3


def test_naive_retriever_filters_by_lang_hint():
    r = NaiveRetriever()
    out = r.retrieve({"lang_hint": ["fr"]}, POEMS)
    assert [p.id for p in out] == ["p2"]


def test_naive_retriever_ignores_unmatched_hint():
    r = NaiveRetriever()
    # No Japanese poems: fall back to the full corpus rather than returning nothing.
    assert len(r.retrieve({"lang_hint": ["ja"]}, POEMS)) == 3


async def fake_understand(client, settings, description, usage=None):
    return {"intent_summary": "moon", "query_lang": "zh", "lang_hint": None}


async def fake_match(client, settings, description, query_lang, batch, usage=None):
    return [
        Verdict(poem_id=p.id, match=True, confidence=0.9,
                matched_description_aspects=["月"], evidence_lines=["床前明月光"],
                explanation="月光", explanation_lang="zh")
        for p in batch if p.id == "p1"
    ]


async def test_small_corpus_skips_scout_stage(monkeypatch):
    monkeypatch.setattr(swarm, "understand_query", fake_understand)
    monkeypatch.setattr(swarm, "match_batch", fake_match)

    settings = make_settings(shortlist_size=10)  # corpus of 3 ≤ 10 → no stage 1
    events = [e async for e in swarm.search_stream(None, settings, NaiveRetriever(), POEMS, "月亮")]
    types = [e["type"] for e in events]

    assert "shortlist" not in types
    stages = {e["stage"] for e in events if e["type"] == "swarm_dispatched"}
    assert stages == {2}
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert len(verdicts) == 1 and verdicts[0]["poem"]["id"] == "p1"
    assert events[-1]["type"] == "done" and events[-1]["matched"] == 1


async def test_large_corpus_runs_scout_then_judge(monkeypatch):
    async def fake_scout(client, settings, description, batch, usage=None):
        # Scouts shortlist p1 and p2 only.
        return [(p.id, 0.8) for p in batch if p.id in ("p1", "p2")]

    monkeypatch.setattr(swarm, "understand_query", fake_understand)
    monkeypatch.setattr(swarm, "scout_batch", fake_scout)
    monkeypatch.setattr(swarm, "match_batch", fake_match)

    settings = make_settings(shortlist_size=2)  # corpus of 3 > 2 → stage 1 runs
    events = [e async for e in swarm.search_stream(None, settings, NaiveRetriever(), POEMS, "月亮")]
    types = [e["type"] for e in events]

    stages = [e["stage"] for e in events if e["type"] == "swarm_dispatched"]
    assert stages == [1, 2]
    shortlist = next(e for e in events if e["type"] == "shortlist")
    assert shortlist["count"] == 2
    # Judges only saw the shortlist, and only p1 matched.
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert [v["poem"]["id"] for v in verdicts] == ["p1"]
    assert types[-1] == "done"
