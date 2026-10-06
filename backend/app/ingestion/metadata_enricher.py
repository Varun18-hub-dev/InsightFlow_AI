"""
MetadataEnricher — adds standard metadata fields to each chunk before ingestion.
Uses a duck-typed document object so it works with both ORM models and test stubs.
"""
import uuid
from datetime import datetime
from typing import Any


class MetadataEnricher:
    """Enriches chunk metadata with document-level context."""

    @staticmethod
    def enrich(
        chunks: list[dict[str, Any]],
        document,  # accepts any object with .id, .filename, .document_type
        user_id: str,
    ) -> list[dict[str, Any]]:
        """
        Add document_id, filename, document_type, user_id, upload_time, chunk_id
        to each chunk's metadata dict in-place.
        """
        now_str = datetime.utcnow().isoformat()
        document_id = str(document.id)
        filename = getattr(document, "filename", "unknown")
        doc_type = getattr(document, "document_type", "unknown")

        for chunk in chunks:
            chunk_meta = chunk.get("metadata", {})
            chunk_meta.update({
                "document_id": document_id,
                "filename": filename,
                "document_type": doc_type,
                "user_id": str(user_id),
                "upload_time": now_str,
                "chunk_id": str(uuid.uuid4()),
                "content": chunk.get("content", ""),
            })
            chunk["metadata"] = chunk_meta
        return chunks
