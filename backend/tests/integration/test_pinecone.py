"""
Integration tests for Pinecone vector store service and fallbacks.
Runs against real Pinecone index when PINECONE_API_KEY is configured.
Skips cleanly when running in environments without Pinecone credentials.
"""
import time
import uuid

import pytest

import app.services.pinecone_service as ps_mod
from app.core.config import settings
from app.services.pinecone_service import InMemoryVectorStore, PineconeService, get_vector_store

HAS_PINECONE = bool(settings.PINECONE_API_KEY and settings.PINECONE_INDEX)
pytestmark = pytest.mark.asyncio


class TestInMemoryVectorStore:
    """Tests for the InMemory fallback vector store."""

    async def test_in_memory_upsert_and_search(self):
        store = InMemoryVectorStore()
        v1 = {"id": "v1", "values": [1.0, 0.0, 0.0], "metadata": {"user_id": "u1", "doc": "a"}}
        v2 = {"id": "v2", "values": [0.0, 1.0, 0.0], "metadata": {"user_id": "u1", "doc": "b"}}
        v3 = {"id": "v3", "values": [0.0, 0.0, 1.0], "metadata": {"user_id": "u2", "doc": "c"}}

        await store.upsert_vectors([v1, v2, v3])
        assert len(store.vectors) == 3

        # Query aligned with v1
        results = await store.search([1.0, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "v1"
        assert results[0]["score"] > 0.9

    async def test_in_memory_metadata_filter(self):
        store = InMemoryVectorStore()
        v1 = {"id": "v1", "values": [0.5, 0.5], "metadata": {"user_id": "u1"}}
        v2 = {"id": "v2", "values": [0.5, 0.5], "metadata": {"user_id": "u2"}}
        await store.upsert_vectors([v1, v2])

        results = await store.search([0.5, 0.5], top_k=5, filter={"user_id": "u1"})
        assert len(results) == 1
        assert results[0]["id"] == "v1"

    async def test_in_memory_delete_document(self):
        store = InMemoryVectorStore()
        v1 = {"id": "v1", "values": [0.1, 0.2], "metadata": {"document_id": "doc1"}}
        v2 = {"id": "v2", "values": [0.3, 0.4], "metadata": {"document_id": "doc2"}}
        await store.upsert_vectors([v1, v2])

        await store.delete_document("doc1")
        assert len(store.vectors) == 1
        assert store.vectors[0]["id"] == "v2"


class TestPineconeFallback:
    """Verify clean fallback behavior when PINECONE_API_KEY is not configured."""

    async def test_get_vector_store_fallback_when_empty_key(self):
        saved_key = settings.PINECONE_API_KEY
        try:
            settings.PINECONE_API_KEY = ""
            ps_mod._pinecone_service_instance = None
            ps_mod._in_memory_store_instance = None

            store = get_vector_store()
            assert isinstance(store, InMemoryVectorStore)
            assert store.backend_name == "in_memory"
        finally:
            settings.PINECONE_API_KEY = saved_key
            ps_mod._pinecone_service_instance = None
            ps_mod._in_memory_store_instance = None


@pytest.mark.skipif(not HAS_PINECONE, reason="PINECONE_API_KEY not configured — skipping live Pinecone tests")
class TestLivePineconeIntegration:
    """Live integration tests running against configured Pinecone index."""

    async def test_pinecone_connectivity_and_index_specs(self):
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        desc = pc.describe_index(settings.PINECONE_INDEX)
        assert desc.dimension == 3072, f"Dimension mismatch: expected 3072, got {desc.dimension}"
        assert desc.metric == "cosine", f"Metric mismatch: expected cosine, got {desc.metric}"
        assert desc.status.state in ("Ready", "ready")

    async def test_pinecone_service_singleton(self):
        store = get_vector_store()
        assert isinstance(store, PineconeService)
        assert store.backend_name == "pinecone"
        assert store.dimension == 3072

    async def test_pinecone_crud_and_metadata_isolation(self):
        store = PineconeService()
        run_id = uuid.uuid4().hex[:8]
        test_namespace = f"pytest-{run_id}"

        # 3072-dimensional normalized probe vectors
        u1_vec = [0.0] * 3072
        u1_vec[0] = 1.0
        u2_vec = [0.0] * 3072
        u2_vec[1] = 1.0

        v1_id = f"vec-u1-{run_id}"
        v2_id = f"vec-u2-{run_id}"

        try:
            # 1. Upsert
            await store.upsert_vectors(
                [
                    {
                        "id": v1_id,
                        "values": u1_vec,
                        "metadata": {"user_id": f"u1-{run_id}", "document_id": f"d1-{run_id}", "run": run_id}
                    },
                    {
                        "id": v2_id,
                        "values": u2_vec,
                        "metadata": {"user_id": f"u2-{run_id}", "document_id": f"d2-{run_id}", "run": run_id}
                    }
                ],
                namespace=test_namespace
            )
            time.sleep(3) # Allow vector propagation

            # 2. Search with User 1 filter
            q_res_u1 = await store.search(
                query_vector=u1_vec,
                top_k=5,
                filter={"user_id": f"u1-{run_id}"},
                namespace=test_namespace
            )
            assert len(q_res_u1) >= 1
            assert q_res_u1[0]["id"] == v1_id
            assert q_res_u1[0]["score"] > 0.99
            # Verify user 2 vector was not returned (user isolation)
            assert not any(r["id"] == v2_id for r in q_res_u1)

            # 3. Search with User 2 filter
            q_res_u2 = await store.search(
                query_vector=u2_vec,
                top_k=5,
                filter={"user_id": f"u2-{run_id}"},
                namespace=test_namespace
            )
            assert len(q_res_u2) >= 1
            assert q_res_u2[0]["id"] == v2_id
            assert q_res_u2[0]["score"] > 0.99

        finally:
            # 4. Clean up test vectors
            try:
                await store.delete_all(namespace=test_namespace)
            except Exception:
                await store.delete_vectors([v1_id, v2_id], namespace=test_namespace)
