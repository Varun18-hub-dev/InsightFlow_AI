from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class PineconeService:
    backend_name = "pinecone"

    def __init__(self):
        try:
            from pinecone import Pinecone
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index_name = settings.get_pinecone_index()
            self.index = self.pc.Index(self.index_name)
            self.dimension = 3072
        except ImportError:
            raise RuntimeError("pinecone package is not installed. Please install pinecone.") from None

    async def upsert_vectors(self, vectors: list[dict[str, Any]], namespace: str = None) -> None:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            kwargs = {"vectors": batch}
            if namespace:
                kwargs["namespace"] = namespace
            self.index.upsert(**kwargs)
            logger.info("upserted_batch", count=len(batch), namespace=namespace)

    async def upsert(self, vectors: list[dict[str, Any]], namespace: str = None) -> None:
        await self.upsert_vectors(vectors, namespace=namespace)

    async def search(self, query_vector: list[float], top_k: int, filter: dict = None, namespace: str = None) -> list[dict]:
        kwargs = {
            "vector": query_vector,
            "top_k": top_k,
            "filter": filter,
            "include_metadata": True,
        }
        if namespace:
            kwargs["namespace"] = namespace
        response = self.index.query(**kwargs)
        return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in response.matches]

    async def query(self, query_vector: list[float], top_k: int, filter: dict = None, namespace: str = None) -> list[dict]:
        return await self.search(query_vector=query_vector, top_k=top_k, filter=filter, namespace=namespace)

    async def delete_document(self, document_id: str, namespace: str = None) -> None:
        kwargs = {"filter": {"document_id": document_id}}
        if namespace:
            kwargs["namespace"] = namespace
        self.index.delete(**kwargs)

    async def delete_all(self, namespace: str = None) -> None:
        if namespace:
            self.index.delete(delete_all=True, namespace=namespace)
        else:
            self.index.delete(delete_all=True)

    async def update_metadata(self, vector_id: str, metadata: dict, namespace: str = None) -> None:
        kwargs = {"id": vector_id, "set_metadata": metadata}
        if namespace:
            kwargs["namespace"] = namespace
        self.index.update(**kwargs)

class InMemoryVectorStore:
    backend_name = "in_memory"

    def __init__(self):
        self.vectors = []

    async def upsert_vectors(self, vectors: list[dict[str, Any]], namespace: str = None) -> None:
        for v in vectors:
            item = dict(v)
            if namespace:
                item["namespace"] = namespace
            self.vectors.append(item)

    async def upsert(self, vectors: list[dict[str, Any]], namespace: str = None) -> None:
        await self.upsert_vectors(vectors, namespace=namespace)

    async def search(self, query_vector: list[float], top_k: int, filter: dict = None, namespace: str = None) -> list[dict]:
        results = []
        qv = np.array(query_vector)
        for v in self.vectors:
            if namespace and v.get("namespace") != namespace:
                continue
            if filter:
                match = all(v["metadata"].get(k) == val for k, val in filter.items())
                if not match:
                    continue

            vv = np.array(v["values"])
            norm_qv = np.linalg.norm(qv)
            norm_vv = np.linalg.norm(vv)
            if norm_qv == 0 or norm_vv == 0:
                score = 0.0
            else:
                score = np.dot(qv, vv) / (norm_qv * norm_vv)
            results.append({"id": v["id"], "score": float(score), "metadata": v["metadata"]})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def query(self, query_vector: list[float], top_k: int, filter: dict = None, namespace: str = None) -> list[dict]:
        return await self.search(query_vector=query_vector, top_k=top_k, filter=filter, namespace=namespace)

    async def delete_document(self, document_id: str, namespace: str = None) -> None:
        self.vectors = [
            v for v in self.vectors
            if not (v["metadata"].get("document_id") == document_id and (not namespace or v.get("namespace") == namespace))
        ]

    async def delete_all(self, namespace: str = None) -> None:
        if namespace:
            self.vectors = [v for v in self.vectors if v.get("namespace") != namespace]
        else:
            self.vectors.clear()

    async def update_metadata(self, vector_id: str, metadata: dict, namespace: str = None) -> None:
        for v in self.vectors:
            if v["id"] == vector_id and (not namespace or v.get("namespace") == namespace):
                v["metadata"].update(metadata)

_pinecone_service_instance = None
_in_memory_store_instance = None

def get_vector_store():
    global _pinecone_service_instance, _in_memory_store_instance
    if settings.PINECONE_API_KEY:
        if _pinecone_service_instance is None:
            try:
                _pinecone_service_instance = PineconeService()
            except Exception as e:
                if settings.ENVIRONMENT == "production":
                    logger.error("pinecone_init_failed_in_production", error=str(e))
                    raise RuntimeError(f"Pinecone initialization failed in production environment: {e}") from e
                logger.warning("pinecone_init_failed_fallback_in_memory", error=str(e))
                if _in_memory_store_instance is None:
                    _in_memory_store_instance = InMemoryVectorStore()
                return _in_memory_store_instance
        return _pinecone_service_instance

    if settings.ENVIRONMENT == "production":
        logger.error("pinecone_api_key_missing_in_production")
        raise RuntimeError("PINECONE_API_KEY is not configured in production environment. Refusing silent in-memory fallback.")

    if _in_memory_store_instance is None:
        _in_memory_store_instance = InMemoryVectorStore()
    return _in_memory_store_instance

