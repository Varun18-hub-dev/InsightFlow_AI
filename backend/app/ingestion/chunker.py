"""
Document chunker using LangChain's RecursiveCharacterTextSplitter.
Falls back gracefully if the package layout changes across LangChain versions.
"""
from typing import Any

# LangChain moved text splitters to its own package in v0.2+
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Pure-Python fallback — no external dependency required
        class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
            def __init__(self, chunk_size=800, chunk_overlap=150, length_function=len, **kwargs):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function

            def split_text(self, text: str) -> list[str]:
                if not text:
                    return []
                chunks = []
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunk = text[start:end]
                    if chunk.strip():
                        chunks.append(chunk)
                    start = end - self.chunk_overlap
                    if start <= 0 and end >= len(text):
                        break
                return chunks

from app.core.config import settings


class DocumentChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        cs = chunk_size or settings.CHUNK_SIZE
        co = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs,
            chunk_overlap=co,
            length_function=len,
        )

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk a single text string."""
        if not text or not text.strip():
            return []
        chunks = self.splitter.split_text(text)
        return [
            {"content": chunk, "chunk_index": i, "metadata": metadata.copy()}
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]

    def chunk_documents(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Chunk a list of pages (each page has 'content', 'page_number', 'metadata')."""
        all_chunks = []
        global_index = 0
        for page in pages:
            content = page.get("content", "")
            if not content or not content.strip():
                continue
            chunks = self.splitter.split_text(content)
            for chunk in chunks:
                if not chunk.strip():
                    continue
                meta = page.get("metadata", {}).copy()
                meta["page_number"] = page.get("page_number", 1)
                all_chunks.append({
                    "content": chunk,
                    "chunk_index": global_index,
                    "metadata": meta,
                })
                global_index += 1
        return all_chunks
