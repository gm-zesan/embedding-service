import httpx
import logging
from typing import Optional
from app.llm.models import LLMRequest, LLMResponse
from app.llm.providers.base import BaseLLMProvider

logger = logging.getLogger("llm.deepseek")


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, api_key: str, base_url: Optional[str] = None, default_model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = (base_url or "https://api.deepseek.com/v1").rstrip("/")
        self.default_model = default_model

    @property
    def name(self) -> str:
        return "deepseek"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8001",
            "X-Title": "Chatbot Retrieval Escape Hatch",
        }

        payload = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        timeout_cfg = httpx.Timeout(1.5, connect=0.8)
        async with httpx.AsyncClient(timeout=timeout_cfg) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"DeepSeek API HTTP {resp.status_code}: {resp.text}")

            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"].strip()
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                provider=self.name,
                model=model,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                finish_reason=choice.get("finish_reason"),
                raw_response=data,
            )
