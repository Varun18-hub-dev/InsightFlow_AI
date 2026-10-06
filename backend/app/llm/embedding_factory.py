from app.core.config import settings
from app.llm.base import BaseEmbeddingProvider
from app.llm.gemini_provider import GeminiEmbeddingProvider
from app.llm.openai_provider import OpenAIEmbeddingProvider

_embedding_instance = None

class EmbeddingProviderFactory:
    @staticmethod
    def get_provider(provider_name: str = None) -> BaseEmbeddingProvider:
        global _embedding_instance
        if _embedding_instance is not None and provider_name is None:
            return _embedding_instance

        name = provider_name or settings.EMBEDDING_PROVIDER
        if name.lower() == "openai":
            provider = OpenAIEmbeddingProvider()
        elif name.lower() == "gemini":
            provider = GeminiEmbeddingProvider()
        else:
            provider = GeminiEmbeddingProvider() # Default

        if provider_name is None:
            _embedding_instance = provider

        return provider
