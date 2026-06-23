import numpy as np

import poemferry.swarm as swarm
from poemferry.config import Settings
from poemferry.models import Poem
from poemferry.retriever import NaiveRetriever
from poemferry.vectors import VectorIndex


def make_settings(**overrides) -> Settings:
    base = dict(
        deepseek_api_key="test", deepseek_base_url="https://example.invalid",
        deepseek_model="deepseek-v4-flash", swarm_batch_size=2, swarm_concurrency=4,
        swarm_max_agents=128, scout_batch_size=2, shortlist_size=10, poems_path="data",
        retrieval_mode="image", embed_base_url="https://example.invalid", embed_api_key="k",
        embed_model="embed-v-4-0", vec_topk=10, trim_topn=10, expert_batch=2,
        swarm_provider="deepseek", glm_base_url="https://example.invalid", glm_api_key="",
        glm_model="glm-4.7-flash",
    )
    base.update(overrides)
    return Settings(**base)


POEMS = [
    Poem(id="p1", title="A", author="X", language="zh", full_text="酒入愁肠\n泪如雨下",
         source_name="s", license="MIT"),
    Poem(id="p2", title="B", author="Y", language="fr", full_text="Demain, dès l'aube",
         source_name="s", license="PD"),
    Poem(id="p3", title="C", author="Z", language="en", full_text="The woods are lovely",
         source_name="s", license="PD"),
]


async def fake_understand(client, settings, description, usage=None):
    return {
        "intent_summary": "crying then drinking wine", "query_lang": "zh",
        "criteria": [
            {"aspect": "weeping", "kind": "emotion", "weight": 1.0},
            {"aspect": "drinking wine", "kind": "action", "weight": 1.0},
        ],
        "fragments": [], "lang_hint": None,
    }


async def fake_expert(client, settings, criterion, query_lang, batch, usage=None):
    # Both criteria hit only p1.
    return {p.id: {"evidence_lines": ["酒入愁肠"], "note": "命中"} for p in batch if p.id == "p1"}


# ── retriever / fragments unchanged behaviour ──
def test_naive_retriever_filters_by_lang_hint():
    assert [p.id for p in NaiveRetriever().retrieve({"lang_hint": ["fr"]}, POEMS)] == ["p2"]


# ── aggregation ──
def test_aggregate_weighted_vote_and_sort():
    criteria = [{"aspect": "weep", "kind": "emotion", "weight": 1.0},
                {"aspect": "drink", "kind": "action", "weight": 1.0}]
    hits = [{"p1": {"evidence_lines": ["泪如雨下"], "note": "哭"}},
            {"p1": {"evidence_lines": ["酒入愁肠"], "note": "喝"}, "p3": {"evidence_lines": ["x"], "note": "?"}}]
    out = swarm.aggregate(POEMS, criteria, hits, "zh")
    assert out[0].poem_id == "p1" and out[0].confidence == 1.0
    # p3 hit only one of two criteria → 0.5, kept but ranked below p1.
    assert [v.poem_id for v in out] == ["p1", "p3"]
    assert "哭" in out[0].explanation and "喝" in out[0].explanation


def test_aggregate_ignores_poems_with_no_hits():
    criteria = [{"aspect": "x", "kind": "imagery", "weight": 1.0}]
    out = swarm.aggregate(POEMS, criteria, [{}], "zh")
    assert out == []


# ── vector index ──
def test_vector_index_cosine_topk():
    ids = ["a", "b", "c"]
    mat = np.array([[1, 0], [0, 1], [0.7071, 0.7071]], dtype=np.float32)
    idx = VectorIndex(ids, mat)
    res = idx.search_unique(np.array([1, 0], dtype=np.float32), k=2)
    assert res[0][0] == "a" and res[1][0] == "c"


# ── full pipeline (vector retrieval mocked via a tiny index) ──
async def test_funnel_pipeline_runs_experts_and_aggregates(monkeypatch):
    monkeypatch.setattr(swarm, "understand_query", fake_understand)
    monkeypatch.setattr(swarm, "expert_batch", fake_expert)
    monkeypatch.setattr("poemferry.vectors.embed_query",
                        lambda settings, text: np.array([1, 0], dtype=np.float32))

    index = VectorIndex(["p1", "p2", "p3"],
                        np.array([[1, 0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32))
    settings = make_settings()
    events = [e async for e in swarm.search_stream(
        None, settings, NaiveRetriever(), POEMS, "哭了之后喝酒", doc_index=index)]
    types = [e["type"] for e in events]

    assert "criteria" in types and "retrieved" in types
    assert [e["stage"] for e in events if e["type"] == "swarm_dispatched"] == [2]
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert [v["poem"]["id"] for v in verdicts] == ["p1"]
    assert events[-1]["type"] == "done" and events[-1]["matched"] == 1


async def test_fragment_channel_forces_into_shortlist(monkeypatch):
    from poemferry.fragments import FragmentIndex

    async def understand_with_fragment(client, settings, description, usage=None):
        d = await fake_understand(client, settings, description, usage)
        d["fragments"] = ["酒入愁肠"]
        return d

    monkeypatch.setattr(swarm, "understand_query", understand_with_fragment)
    monkeypatch.setattr(swarm, "expert_batch", fake_expert)
    monkeypatch.setattr("poemferry.vectors.embed_query",
                        lambda settings, text: np.array([0, 1], dtype=np.float32))

    # Vector ranks p1 last, but the quoted line forces it into the shortlist.
    index = VectorIndex(["p2", "p3", "p1"],
                        np.array([[0, 1], [0.1, 0.9], [0.2, 0.8]], dtype=np.float32))
    settings = make_settings()
    events = [e async for e in swarm.search_stream(
        None, settings, NaiveRetriever(), POEMS, "我记得一句酒入愁肠",
        fragment_index=FragmentIndex(POEMS), doc_index=index)]

    assert any(e["type"] == "fragment_hits" and e["count"] == 1 for e in events)
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert [v["poem"]["id"] for v in verdicts] == ["p1"]
