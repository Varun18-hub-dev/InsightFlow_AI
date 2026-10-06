from app.core.config import settings
from app.llm.base import BaseLLMProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.openai_provider import OpenAIProvider

_llm_instance = None

class LLMProviderFactory:
    @staticmethod
    def get_provider(provider_name: str = None) -> BaseLLMProvider:
        global _llm_instance
        if _llm_instance is not None and provider_name is None:
            return _llm_instance

        name = provider_name or settings.LLM_PROVIDER
        if name.lower() == "openai":
            provider = OpenAIProvider()
        elif name.lower() == "gemini":
            provider = GeminiProvider()
        else:
            provider = GeminiProvider() # Default

        if provider_name is None:
            _llm_instance = provider

        return provider
