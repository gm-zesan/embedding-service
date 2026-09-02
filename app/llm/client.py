import logging
from typing import Dict, Optional
from app import config
from app.llm.models import LLMRequest, LLMResponse
from app.llm.providers.base import BaseLLMProvider
from app.llm.providers.deepseek import DeepSeekProvider
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.providers.openai import OpenAIProvider

logger = logging.getLogger("llm.client")


class LLMClient:
    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}

    def resolve_provider(self, name: str) -> BaseLLMProvider:
        clean_name = name.lower().strip()
        if clean_name in self._providers:
            return self._providers[clean_name]

        if clean_name == "deepseek":
            provider = DeepSeekProvider(
                api_key=config.RETRIEVAL_LLM_API_KEY,
                base_url=config.RETRIEVAL_LLM_BASE_URL,
                default_model=config.RETRIEVAL_LLM_MODEL,
            )
        elif clean_name == "openrouter":
            provider = OpenRouterProvider(
                api_key=config.RETRIEVAL_LLM_API_KEY,
                base_url=config.RETRIEVAL_LLM_BASE_URL,
                default_model=config.RETRIEVAL_LLM_MODEL,
            )
        elif clean_name == "openai":
            provider = OpenAIProvider(
                api_key=config.RETRIEVAL_LLM_API_KEY,
                base_url=config.RETRIEVAL_LLM_BASE_URL,
                default_model=config.RETRIEVAL_LLM_MODEL,
            )
        else:
            raise ValueError(f"Unsupported retrieval LLM provider: '{name}'")

        self._providers[clean_name] = provider
        return provider

    @staticmethod
    def is_fallback_eligible(e: Exception) -> bool:
        msg = str(e).lower()
        if "401" in msg or "403" in msg or "unauthorized" in msg or "invalid api key" in msg:
            return False
        if "400" in msg or "422" in msg or "bad request" in msg:
            return False
        return True

    async def generate(self, request: LLMRequest, provider_name: Optional[str] = None) -> LLMResponse:
        import time

        t_start = time.perf_counter()
        primary = (provider_name or config.RETRIEVAL_LLM_PROVIDER).lower().strip()
        fallback = config.RETRIEVAL_LLM_FALLBACK_PROVIDER.lower().strip()
        fallback_used = False

        try:
            prov = self.resolve_provider(primary)
            response = await prov.generate(request)
        except Exception as e_primary:
            eligible = self.is_fallback_eligible(e_primary)
            logger.warning("Primary LLM provider '%s' failed (fallback_eligible=%s): %s", primary, eligible, e_primary)

            if eligible and fallback and fallback != primary:
                try:
                    logger.info("Attempting configured fallback LLM provider '%s'...", fallback)
                    prov_fallback = self.resolve_provider(fallback)
                    response = await prov_fallback.generate(request)
                    fallback_used = True
                except Exception as e_fallback:
                    logger.error("Fallback LLM provider '%s' also failed: %s", fallback, e_fallback)
                    raise RuntimeError(f"All LLM providers failed. Primary: {e_primary} | Fallback: {e_fallback}") from e_fallback
            else:
                raise e_primary

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        response.telemetry = {
            "latency_ms": latency_ms,
            "fallback_used": fallback_used,
            "primary_provider": primary,
            "active_provider": response.provider,
            "model": response.model,
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
        }
        return response


# Global singleton client
default_client = LLMClient()
