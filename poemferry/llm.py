import json

from openai import AsyncOpenAI

from .config import Settings


def make_client(settings: Settings) -> AsyncOpenAI:
    """Client + model for the swarm, picked by SWARM_PROVIDER (deepseek|glm)."""
    if settings.swarm_provider == "glm" and settings.glm_api_key:
        return AsyncOpenAI(api_key=settings.glm_api_key, base_url=settings.glm_base_url)
    return AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def swarm_model(settings: Settings) -> str:
    if settings.swarm_provider == "glm" and settings.glm_api_key:
        return settings.glm_model
    return settings.deepseek_model


async def chat_json(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2000,
    usage: dict | None = None,
) -> dict:
    """Call the chat API in JSON mode and parse the response object.

    DeepSeek is OpenAI-compatible. v4-flash is a hybrid reasoning model; its
    reasoning lives in `reasoning_content`, while `content` holds the JSON.
    When a `usage` dict is given, token counts (incl. DeepSeek prefix-cache
    hits) are accumulated into it.
    """
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=max_tokens,
    )
    if usage is not None and resp.usage:
        usage["prompt"] = usage.get("prompt", 0) + (resp.usage.prompt_tokens or 0)
        usage["completion"] = usage.get("completion", 0) + (resp.usage.completion_tokens or 0)
        cache_hit = getattr(resp.usage, "prompt_cache_hit_tokens", 0) or 0
        usage["cache_hit"] = usage.get("cache_hit", 0) + cache_hit
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)
