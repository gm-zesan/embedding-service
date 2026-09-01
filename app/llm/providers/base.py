from abc import ABC, abstractmethod
from app.llm.models import LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute completion request and return standardized LLMResponse."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string."""
        pass
