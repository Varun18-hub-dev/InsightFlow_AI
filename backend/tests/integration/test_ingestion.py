"""Integration tests for document ingestion pipeline."""
import os
import tempfile


def test_text_extractor_reads_file():
    from app.ingestion.extractors import TextExtractor
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Enterprise AI systems require robust document processing pipelines.\n\nThis is page one content.")
        path = f.name
    try:
        extractor = TextExtractor()
        pages = extractor.extract(path)
        assert len(pages) > 0
        assert any("document" in p["content"].lower() for p in pages)
    finally:
        os.unlink(path)


def test_text_cleaner_removes_extra_whitespace():
    from app.ingestion.cleaner import TextCleaner
    cleaner = TextCleaner()
    dirty = "This has  extra   spaces\t\tand\ttabs\n\n\n\nmultiple newlines"
    clean = cleaner.clean(dirty)
    assert "  " not in clean
    assert len(clean) < len(dirty)


def test_chunker_produces_metadata():
    from app.ingestion.chunker import DocumentChunker
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    pages = [{"content": "word " * 100, "page_number": 1, "metadata": {}}]
    chunks = chunker.chunk_documents(pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "chunk_index" in chunk
        assert "content" in chunk
        assert len(chunk["content"]) > 0


def test_end_to_end_txt_pipeline():
    """Full pipeline: extract → clean → chunk → metadata."""
    from app.ingestion.chunker import DocumentChunker
    from app.ingestion.cleaner import TextCleaner
    from app.ingestion.extractors import TextExtractor

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Enterprise AI Platform InsightFlow " * 60)
        path = f.name

    try:
        extractor = TextExtractor()
        cleaner = TextCleaner()
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)

        pages = extractor.extract(path)
        for page in pages:
            page["content"] = cleaner.clean(page["content"])

        chunks = chunker.chunk_documents(pages)

        assert len(chunks) > 0
        assert all("content" in c for c in chunks)
        assert all(len(c["content"]) > 0 for c in chunks)
    finally:
        os.unlink(path)


def test_metadata_enricher():
    import uuid

    from app.ingestion.metadata_enricher import MetadataEnricher

    enricher = MetadataEnricher()
    chunks = [
        {"content": "test content", "chunk_index": 0, "metadata": {"page_number": 1}},
        {"content": "more content", "chunk_index": 1, "metadata": {"page_number": 1}},
    ]

    class FakeDoc:
        id = uuid.uuid4()
        filename = "test_doc.pdf"
        document_type = "pdf"
        created_at = None

    enriched = enricher.enrich(chunks, FakeDoc(), "user-123")
    assert len(enriched) == 2
    for c in enriched:
        assert "document_id" in c["metadata"]
        assert c["metadata"]["user_id"] == "user-123"
