import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_provider():
    from unittest.mock import AsyncMock
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value="This is a test answer from the mock LLM provider.")
    provider.get_model_name = lambda: "mock-model"
    return provider


@pytest.fixture
def mock_embedding_provider():
    from unittest.mock import AsyncMock
    provider = AsyncMock()
    provider.embed_query = AsyncMock(return_value=[0.1] * 768)
    provider.embed_documents = AsyncMock(return_value=[[0.1] * 768])
    provider.get_dimension = lambda: 768
    return provider


@pytest.fixture
def sample_chunks():
    from app.retrieval.base_retriever import RetrievedChunk
    return [
        RetrievedChunk(id=f"chunk_{i}", content=f"Sample content {i}", score=1.0 - i * 0.1, metadata={"filename": "test.pdf", "page_number": i})
        for i in range(5)
    ]
