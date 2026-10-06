from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)
    rank: int = 0

class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int, filters: dict | None = None) -> list[RetrievedChunk]:
        pass
