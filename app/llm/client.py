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

    async def generate(self, request: LLMRequest, provider_name: Optional[str] = None) -> LLMResponse:
        primary = (provider_name or config.RETRIEVAL_LLM_PROVIDER).lower().strip()
        fallback = config.RETRIEVAL_LLM_FALLBACK_PROVIDER.lower().strip()

        try:
            prov = self.resolve_provider(primary)
            return await prov.generate(request)
        except Exception as e_primary:
            logger.warning("Primary LLM provider '%s' failed: %s", primary, e_primary)
            if fallback and fallback != primary:
                try:
                    logger.info("Attempting configured fallback LLM provider '%s'...", fallback)
                    prov_fallback = self.resolve_provider(fallback)
                    return await prov_fallback.generate(request)
                except Exception as e_fallback:
                    logger.error("Fallback LLM provider '%s' also failed: %s", fallback, e_fallback)
                    raise RuntimeError(f"All LLM providers failed. Primary: {e_primary} | Fallback: {e_fallback}") from e_fallback
            raise e_primary


# Global singleton client
default_client = LLMClient()
