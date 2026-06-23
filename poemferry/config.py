import os
from dataclasses import dataclass


MAX_CONCURRENCY = 64


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    swarm_batch_size: int
    swarm_concurrency: int
    swarm_max_agents: int
    scout_batch_size: int
    shortlist_size: int
    poems_path: str
    retrieval_mode: str
    embed_base_url: str
    embed_api_key: str
    embed_model: str
    vec_topk: int
    trim_topn: int
    expert_batch: int
    swarm_provider: str
    glm_base_url: str
    glm_api_key: str
    glm_model: str
    # scan mode: read full text in retrieval-order until satisfied (see swarm.scan_stream)
    scan_order_size: int = 1500
    scan_max_read: int = 768
    scan_target_hits: int = 12
    scan_patience: int = 3
    scan_strong_score: int = 4


def load_settings() -> Settings:
    return Settings(
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        swarm_batch_size=int(os.environ.get("SWARM_BATCH_SIZE", "4")),
        swarm_concurrency=min(int(os.environ.get("SWARM_CONCURRENCY", "48")), MAX_CONCURRENCY),
        swarm_max_agents=int(os.environ.get("SWARM_MAX_AGENTS", "128")),
        scout_batch_size=int(os.environ.get("SWARM_SCOUT_BATCH", "80")),
        shortlist_size=int(os.environ.get("SWARM_SHORTLIST", "240")),
        poems_path=os.environ.get("POEMS_PATH", "data"),
        retrieval_mode=os.environ.get("RETRIEVAL_MODE", "hybrid"),
        embed_base_url=os.environ.get("EMBED_BASE_URL", "https://aihubmix.com/v1"),
        embed_api_key=os.environ.get("EMBED_API_KEY", ""),
        embed_model=os.environ.get("EMBED_MODEL", "embed-v-4-0"),
        vec_topk=int(os.environ.get("VEC_TOPK", "80")),
        trim_topn=int(os.environ.get("TRIM_TOPN", "30")),
        expert_batch=int(os.environ.get("EXPERT_BATCH", "8")),
        swarm_provider=os.environ.get("SWARM_PROVIDER", "deepseek"),
        glm_base_url=os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        glm_api_key=os.environ.get("GLM_API_KEY", ""),
        glm_model=os.environ.get("GLM_MODEL", "glm-4.7-flash"),
        scan_order_size=int(os.environ.get("SCAN_ORDER_SIZE", "1500")),
        scan_max_read=int(os.environ.get("SCAN_MAX_READ", "768")),
        scan_target_hits=int(os.environ.get("SCAN_TARGET_HITS", "12")),
        scan_patience=int(os.environ.get("SCAN_PATIENCE", "3")),
        scan_strong_score=int(os.environ.get("SCAN_STRONG_SCORE", "4")),
    )
