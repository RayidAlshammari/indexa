"""Provider-agnostic chat client for any OpenAI-compatible endpoint.

One code path serves Groq (including its in-Kingdom region when enabled for
the account), Azure AI Foundry, HUMAIN, or a self-hosted vLLM server, so the
model can be swapped by configuration without touching the engine.

    INDEXA_LLM_BASE_URL   e.g. https://api.groq.com/openai/v1
    INDEXA_LLM_API_KEY
    INDEXA_LLM_MODEL      e.g. openai/gpt-oss-120b
"""

from __future__ import annotations

import os

import httpx

DEFAULTS = {
    "base_url": "https://api.groq.com/openai/v1",
    "model": "openai/gpt-oss-120b",
}


class ChatClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 900,
        timeout: float = 90.0,
    ):
        self.base_url = (base_url or os.getenv("INDEXA_LLM_BASE_URL") or DEFAULTS["base_url"]).rstrip("/")
        self.api_key = api_key or os.getenv("INDEXA_LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("INDEXA_LLM_MODEL") or DEFAULTS["model"]
        self.temperature, self.max_tokens = temperature, max_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __call__(self, messages: list[dict]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        for attempt in range(4):
            r = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if r.status_code == 429 and attempt < 3:
                import asyncio

                await asyncio.sleep(float(r.headers.get("retry-after", 2 * (attempt + 1))))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        raise RuntimeError("rate limited")

    async def aclose(self) -> None:
        await self._client.aclose()
