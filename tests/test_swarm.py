import poemferry.swarm as swarm
from poemferry.config import Settings
from poemferry.models import Poem
from poemferry.retriever import NaiveRetriever

SETTINGS = Settings(
    deepseek_api_key="test",
    deepseek_base_url="https://example.invalid",
    deepseek_model="deepseek-v4-flash",
    swarm_batch_size=2,
    swarm_concurrency=4,
    swarm_max_agents=128,
    poems_path="data/poems.jsonl",
)

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


async def test_search_stream_emits_pipeline_events(monkeypatch):
    async def fake_understand(client, settings, description):
        return {"intent_summary": "moon", "query_lang": "zh", "lang_hint": None}

    async def fake_match(client, settings, description, query_lang, batch):
        from poemferry.models import Verdict
        return [
            Verdict(poem_id=p.id, match=(p.id == "p1"), confidence=0.9,
                    matched_description_aspects=["月"], evidence_lines=["床前明月光"],
                    explanation="月光", explanation_lang="zh")
            for p in batch
        ]

    monkeypatch.setattr(swarm, "understand_query", fake_understand)
    monkeypatch.setattr(swarm, "match_batch", fake_match)

    events = [e async for e in swarm.search_stream(None, SETTINGS, NaiveRetriever(), POEMS, "月亮")]
    types = [e["type"] for e in events]

    assert types[0] == "start"
    assert "understanding" in types
    assert any(e["type"] == "candidates" and e["count"] == 3 for e in events)
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert len(verdicts) == 1 and verdicts[0]["poem"]["id"] == "p1"
    assert events[-1] == {"type": "done", "matched": 1}
