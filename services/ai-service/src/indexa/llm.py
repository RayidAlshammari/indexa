"""Provider-agnostic chat client for any OpenAI-compatible endpoint.

One code path serves Groq (including its in-Kingdom region when enabled for
the account), Azure AI Foundry, HUMAIN, or a self-hosted vLLM server, so the
model can be swapped by configuration without touching the engine.

    INDEXA_LLM_BASE_URL   e.g. https://api.groq.com/openai/v1
    INDEXA_LLM_API_KEY
    INDEXA_LLM_MODEL      e.g. openai/gpt-oss-120b
"""

from __future__ import annotations

import asyncio
import json
import os
import random

import httpx


class ProviderLimit(RuntimeError):
    """The provider asked us to wait longer than an assessment can (e.g. a daily token cap)."""


class RequestTooLarge(RuntimeError):
    """The provider refused the request size (Groq: prompt + max_tokens above the TPM limit)."""


MAX_WAIT = 90.0  # seconds; a longer retry-after means a quota, not a burst

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
        max_tokens: int | None = None,
        timeout: float = 90.0,
    ):
        self.base_url = (
            base_url or os.getenv("INDEXA_LLM_BASE_URL") or DEFAULTS["base_url"]
        ).rstrip("/")
        self.api_key = api_key or os.getenv("INDEXA_LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("INDEXA_LLM_MODEL") or DEFAULTS["model"]
        # Reasoning models spend output tokens thinking first; give them room on paid providers.
        self.temperature = temperature
        self.max_tokens = max_tokens or int(os.getenv("INDEXA_LLM_MAX_TOKENS", "900"))
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __call__(self, messages: list[dict]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Provider-specific options, e.g. OpenRouter {"reasoning": {"effort": "low"}}.
            **json.loads(os.getenv("INDEXA_LLM_EXTRA") or "{}"),
        }
        attempts = 10  # free tiers rate-limit hard; waiting is cheaper than a failed judgement
        for attempt in range(attempts):
            r = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if r.status_code == 413:
                raise RequestTooLarge(r.text[:300])
            if r.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
                wait = float(r.headers.get("retry-after") or min(30, 2 * (attempt + 1)))
                if wait > MAX_WAIT:
                    raise ProviderLimit(f"provider limit reached, retry after {wait:.0f}s")
                await asyncio.sleep(wait + random.uniform(0.5, 2.0))
                continue
            r.raise_for_status()
            choice = r.json()["choices"][0]
            content = choice["message"].get("content") or ""
            if not content.strip() and choice.get("finish_reason") == "length":
                raise RuntimeError("model output truncated before the answer (raise max tokens)")
            return content
        raise RuntimeError("rate limited")

    async def aclose(self) -> None:
        await self._client.aclose()
