"""Unit tests for DocumentChunker."""
import pytest
from app.ingestion.chunker import DocumentChunker


def test_basic_chunking_produces_multiple_chunks():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "word " * 200  # ~1000 chars
    chunks = chunker.chunk(text, metadata={"source": "test"})
    assert len(chunks) > 1


def test_empty_text_returns_empty():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk("", metadata={})
    assert chunks == []


def test_chunk_index_sequential():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    text = "hello world " * 50
    chunks = chunker.chunk(text, metadata={})
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_metadata_preserved_in_all_chunks():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=0)
    text = "sample content " * 30
    chunks = chunker.chunk(text, metadata={"filename": "test.pdf", "page": 1})
    assert all(chunks), "Should produce at least one chunk"
    for c in chunks:
        assert c["metadata"]["filename"] == "test.pdf"
        assert c["metadata"]["page"] == 1


def test_content_field_present():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "enterprise AI platform " * 20
    chunks = chunker.chunk(text, metadata={})
    for c in chunks:
        assert "content" in c
        assert len(c["content"]) > 0


def test_chunk_documents_multi_page():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    pages = [
        {"content": "Page one content. " * 20, "page_number": 1, "metadata": {}},
        {"content": "Page two content. " * 20, "page_number": 2, "metadata": {}},
    ]
    chunks = chunker.chunk_documents(pages)
    assert len(chunks) >= 2
    assert all("chunk_index" in c for c in chunks)
