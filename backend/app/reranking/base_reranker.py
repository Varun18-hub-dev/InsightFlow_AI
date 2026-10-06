from abc import ABC, abstractmethod

from app.retrieval.base_retriever import RetrievedChunk


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        pass
