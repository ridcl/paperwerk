import asyncio
from typing import Any, Optional, Sequence

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion


def _strictify(schema: Any) -> Any:
    """Recursively ensure every ``object`` sets ``additionalProperties: false``.

    Strict ``json_schema`` structured output (required by Anthropic's
    OpenAI-compatible endpoint) rejects object schemas that don't
    explicitly forbid extra properties. vLLM is lenient and accepts the
    schema either way, so normalizing here keeps both backends working.
    """
    if isinstance(schema, dict):
        schema = {k: _strictify(v) for k, v in schema.items()}
        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        return schema
    if isinstance(schema, list):
        return [_strictify(item) for item in schema]
    return schema


class LLM:
    """Thin wrapper around an OpenAI-compatible chat endpoint.

    Bundles ``base_url``, ``api_key``, and ``model`` into a single object
    so callers don't have to thread them around separately. Exposes both
    sync (``invoke``) and async (``ainvoke``) entry points and supports
    structured output via ``schema`` and tool calling via ``tools``.

    Works with any provider that speaks the OpenAI chat-completions API
    (OpenAI, Anthropic's OpenAI-compatible endpoint, vLLM, etc.).

    ``max_concurrency`` caps the number of in-flight ``ainvoke`` calls,
    so callers can ``asyncio.gather`` arbitrarily many requests without
    overwhelming the endpoint. The cap is global to this ``LLM`` instance.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_concurrency: int = 8,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def __repr__(self):
        return f"LLM(model={self.model!r}, base_url={self.base_url!r})"

    def _params(
        self,
        messages: list[dict],
        schema: Optional[dict],
        tools: Optional[Sequence[dict]],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        if schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title", "response"),
                    "schema": _strictify(schema),
                    "strict": True,
                },
            }
        if tools is not None:
            params["tools"] = list(tools)
        params.update(extra)
        return params

    def invoke(
        self,
        messages: list[dict],
        *,
        schema: Optional[dict] = None,
        tools: Optional[Sequence[dict]] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        params = self._params(messages, schema, tools, kwargs)
        return self.client.chat.completions.create(**params)

    async def ainvoke(
        self,
        messages: list[dict],
        *,
        schema: Optional[dict] = None,
        tools: Optional[Sequence[dict]] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        params = self._params(messages, schema, tools, kwargs)
        async with self._semaphore:
            return await self.aclient.chat.completions.create(**params)
