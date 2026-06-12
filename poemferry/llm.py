import json

from openai import AsyncOpenAI

from .config import Settings


def make_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


async def chat_json(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2000,
) -> dict:
    """Call the chat API in JSON mode and parse the response object.

    DeepSeek is OpenAI-compatible. v4-flash is a hybrid reasoning model; its
    reasoning lives in `reasoning_content`, while `content` holds the JSON.
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
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)
