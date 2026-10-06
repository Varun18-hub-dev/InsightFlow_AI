from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.base import AsyncSessionLocal
from app.db.models import Document, DocumentChunk
from app.ingestion.chunker import DocumentChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.extractors import ExtractorFactory
from app.ingestion.metadata_enricher import MetadataEnricher
from app.llm.embedding_factory import EmbeddingProviderFactory
from app.services.pinecone_service import get_vector_store

logger = get_logger(__name__)


async def _process_document_internal(document_id: UUID, file_path: str, document_type: str, db: AsyncSession) -> bool:
    doc = None
    try:
        # 1. Update status to 'processing'
        doc = await db.get(Document, document_id)
        if not doc:
            logger.error("document_not_found", document_id=str(document_id))
            return False

        doc.status = "processing"
        await db.commit()

        # 2. Extract text
        extractor = ExtractorFactory.get_extractor(document_type)
        pages = extractor.extract(file_path)

        # 3. Clean text
        for page in pages:
            page["content"] = TextCleaner.clean(page["content"])

        # 4. Chunk
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(pages)

        # 5. Enrich metadata
        chunks = MetadataEnricher.enrich(chunks, doc, str(doc.user_id))

        # 6. Generate embeddings
        embedding_provider = EmbeddingProviderFactory.get_provider()
        texts = [c["content"] for c in chunks]
        embeddings = await embedding_provider.embed_documents(texts)

        # 7. Upsert to Pinecone
        vector_store = get_vector_store()
        vectors = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["metadata"]["chunk_id"]
            vectors.append({
                "id": chunk_id,
                "values": embeddings[i],
                "metadata": chunk["metadata"],
            })

            # 8. Save chunks to PostgreSQL
            db_chunk = DocumentChunk(
                document_id=doc.id,
                user_id=doc.user_id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                page_number=chunk["metadata"].get("page_number"),
                chunk_metadata=chunk["metadata"],
                embedding_id=chunk_id,
            )
            db.add(db_chunk)

        await vector_store.upsert_vectors(vectors)

        # 9. Update document status
        doc.status = "processed"
        doc.page_count = len(pages)
        doc.chunk_count = len(chunks)
        await db.commit()
        logger.info("document_processed_successfully", document_id=str(document_id), chunks=len(chunks))
        return True

    except Exception as e:
        logger.error("document_processing_failed", error=str(e), document_id=str(document_id))
        if doc is not None:
            try:
                doc.status = "failed"
                doc.error_message = str(e)
                await db.commit()
            except Exception as commit_err:
                logger.error("failed_to_save_error_status", error=str(commit_err))
        return False


async def process_document(
    document_id: UUID,
    file_path: str,
    document_type: str,
    db: AsyncSession | None = None,
) -> bool:
    """Entry point for document processing; manages independent session for background tasks."""
    if db is None:
        async with AsyncSessionLocal() as session:
            return await _process_document_internal(document_id, file_path, document_type, session)
    return await _process_document_internal(document_id, file_path, document_type, db)
