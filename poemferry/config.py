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
    )
