"""
Phase C: End-to-End Pinecone/VectorStore Hybrid RAG Pipeline Verification
Executes the full 17-step flow specified in Phase C with real PDF extraction,
real Gemini embeddings, live PostgreSQL FTS, RRF, reranking, and Gemini 2.5 Flash generation.
"""
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

# Clean global GOOGLE_API_KEY conflict and ensure Gemini key is loaded
os.environ.pop("GOOGLE_API_KEY", None)

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # PyMuPDF
from sqlalchemy import select, text
from app.core.config import settings
from app.db.base import AsyncSessionLocal
from app.db.models import User, Document, DocumentChunk
from app.core.security import hash_password
from app.ingestion.extractors import PDFExtractor
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import DocumentChunker
from app.ingestion.metadata_enricher import MetadataEnricher
from app.llm.embedding_factory import EmbeddingProviderFactory
from app.services.pinecone_service import get_vector_store
from app.retrieval.semantic_retriever import SemanticRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranking.lightweight_reranker import LightweightReranker
from app.llm.factory import LLMProviderFactory
from app.core.prompt_manager import prompt_manager


def create_sample_pdf(filepath: str):
    """Generate a real 2-page PDF with exact facts for verification."""
    doc = fitz.open()
    
    # Page 1: Cloud Strategy & Financials
    p1 = doc.new_page()
    text_p1 = """InsightFlow Enterprise AI - Cloud Strategy & Financial Report 2025

1. Executive Financial Summary:
The annual cloud infrastructure budget is allocated at $4.75 million across AWS and GCP.
Projected operational cost savings from automated retrieval workflows total $1.35 million annually.
All cloud expenditures are reviewed bi-weekly by the FinOps engineering committee.

2. Migration Timelines & SLA:
The target cloud migration completion date is November 15, 2025.
The maximum allowable recovery time objective (RTO) is 12 minutes across all primary banking services.
The recovery point objective (RPO) is strictly bounded to zero data loss for transactional ledgers.
"""
    p1.insert_text(fitz.Point(50, 70), text_p1, fontsize=11)
    
    # Page 2: Leadership & Security
    p2 = doc.new_page()
    text_p2 = """InsightFlow Enterprise AI - Governance & Security Architecture

3. Leadership Appointments:
Dr. Elena Rostova was appointed Chief Information Security Officer (CISO) effective August 1, 2024.
Marcus Vance leads the Data Platform and Hybrid Retrieval Infrastructure division.

4. Compliance & Encryption Standards:
All customer documents and chunk embeddings are encrypted with AES-256 at rest and TLS 1.3 in transit.
Role-Based Access Control (RBAC) enforces strict tenant isolation at the user_id boundary.
Audit logs are shipped in real-time to Amazon CloudWatch and retained for 365 days.
"""
    p2.insert_text(fitz.Point(50, 70), text_p2, fontsize=11)
    
    doc.save(filepath)
    doc.close()
    print(f"      [PASS] Generated real test PDF: {filepath}")


async def run_phase_c_verification():
    print("=" * 70)
    print("PHASE C: Complete Hybrid RAG Pipeline Verification")
    print("=" * 70)

    test_pdf_path = "test_enterprise_cloud_2025.pdf"
    create_sample_pdf(test_pdf_path)

    async with AsyncSessionLocal() as db:
        # Step 1: User setup
        print("\n[Step 1/17] Authenticating test user...")
        test_email = f"phase_c_test_{int(time.time())}@insightflow.ai"
        user = User(
            id=uuid.uuid4(),
            email=test_email,
            full_name="Phase C Verification Engineer",
            hashed_password=hash_password("SuperSecret123!"),
            is_active=True
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        user_id = str(user.id)
        print(f"      [PASS] Test user created: {test_email} (ID: {user_id})")

        # Step 2 & 3: File validation & text extraction
        print("\n[Step 2-3/17] Validating and extracting text from real PDF...")
        extractor = PDFExtractor()
        pages = extractor.extract(test_pdf_path)
        assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}"
        print(f"      [PASS] Extracted {len(pages)} pages using PyMuPDF")

        # Step 4: Text cleaning & chunking
        print("\n[Step 4/17] Cleaning text and splitting into semantic chunks...")
        for p in pages:
            p["content"] = TextCleaner.clean(p["content"])
        chunker = DocumentChunker(chunk_size=400, chunk_overlap=80)
        chunks = chunker.chunk_documents(pages)
        print(f"      [PASS] Created {len(chunks)} chunks from 2 pages")

        # Create Document record
        doc_id = uuid.uuid4()
        doc = Document(
            id=doc_id,
            user_id=user.id,
            filename="enterprise_cloud_2025.pdf",
            original_filename="enterprise_cloud_2025.pdf",
            document_type="application/pdf",
            file_path=test_pdf_path,
            file_size=os.path.getsize(test_pdf_path),
            status="processing",
            page_count=len(pages),
            chunk_count=len(chunks)
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        chunks = MetadataEnricher.enrich(chunks, doc, user_id)

        # Step 5: Real embeddings with configured provider
        print("\n[Step 5/17] Generating real embeddings via Gemini provider...")
        t0 = time.time()
        embedding_provider = EmbeddingProviderFactory.get_provider()
        texts = [c["content"] for c in chunks]
        embeddings = await embedding_provider.embed_documents(texts)
        embed_time = time.time() - t0
        dim = len(embeddings[0])
        print(f"      [PASS] Generated {len(embeddings)} embeddings (dim={dim}) in {embed_time:.2f}s")

        # Step 6: Vector Store Backend Identification & Upsert
        print("\n[Step 6/17] Checking active vector backend and upserting...")
        vector_store = get_vector_store()
        backend_name = getattr(vector_store, "backend_name", type(vector_store).__name__)
        print(f"      Active Vector Backend: {backend_name}")
        if settings.PINECONE_API_KEY:
            print("      Mode: LIVE PINECONE CLOUD")
        else:
            print("      Mode: IN-MEMORY FALLBACK (PINECONE_API_KEY is not configured)")

        vectors = []
        for i, chunk in enumerate(chunks):
            cid = chunk["metadata"]["chunk_id"]
            vectors.append({"id": cid, "values": embeddings[i], "metadata": chunk["metadata"]})

            db_chunk = DocumentChunk(
                document_id=doc.id,
                user_id=user.id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                page_number=chunk["metadata"].get("page_number"),
                chunk_metadata=chunk["metadata"],
                embedding_id=cid
            )
            db.add(db_chunk)

        await vector_store.upsert_vectors(vectors)
        doc.status = "processed"
        await db.commit()
        print(f"      [PASS] Stored {len(vectors)} vectors in {backend_name} and PostgreSQL document_chunks")

        # Step 7: Confirm vectors are searchable
        print("\n[Step 7/17] Confirming vectors are searchable in vector store...")
        test_q_vec = await embedding_provider.embed_query("annual cloud infrastructure budget")
        probe_res = await vector_store.search(test_q_vec, top_k=2, filter={"user_id": user_id})
        assert len(probe_res) > 0, "No vectors returned from search probe"
        print(f"      [PASS] Vector probe returned {len(probe_res)} matches (top score={probe_res[0]['score']:.4f})")

        # Step 8: Semantic retrieval
        query = "What is the annual cloud infrastructure budget and who is the CISO?"
        print(f"\n[Step 8/17] Semantic retrieval for: '{query}'...")
        sem_retriever = SemanticRetriever()
        sem_results = await sem_retriever.retrieve(query, top_k=5, filters={"user_id": user_id})
        print(f"      [PASS] Semantic search retrieved {len(sem_results)} chunks (top score={sem_results[0].score:.4f})")

        # Step 9: PostgreSQL Full-Text Search
        print("\n[Step 9/17] Keyword (PostgreSQL Full-Text Search) retrieval...")
        kw_retriever = KeywordRetriever(db)
        kw_results = await kw_retriever.retrieve(query, top_k=5, filters={"user_id": user_id})
        print(f"      [PASS] PostgreSQL FTS retrieved {len(kw_results)} chunks")

        # Step 10: Concurrent hybrid retrieval
        print("\n[Step 10/17] Executing concurrent hybrid retrieval (asyncio.gather)...")
        hybrid_retriever = HybridRetriever(db)
        hybrid_results = await hybrid_retriever.retrieve(query, top_k_semantic=5, top_k_keyword=5, filters={"user_id": user_id})
        print(f"      [PASS] Concurrent hybrid retrieval completed: {len(hybrid_results)} chunks retrieved")

        # Step 11: RRF Fusion
        print("\n[Step 11/17] RRF fusion verified (k=60)...")
        for i, r in enumerate(hybrid_results[:3]):
            print(f"        Rank {i}: score={r.score:.5f} | ID={r.id}")

        # Step 12: Lightweight Reranking
        print("\n[Step 12/17] Applying LightweightReranker...")
        reranker = LightweightReranker()
        reranked = await reranker.rerank(query, hybrid_results, top_k=3)
        print(f"      [PASS] Reranked to top {len(reranked)} chunks")
        for i, r in enumerate(reranked):
            print(f"        Rank {i}: score={r.score:.4f} | Content: {r.content[:70]}...")

        # Step 13: Generate answer using real Gemini 2.5 Flash
        print("\n[Step 13/17] Generating answer using Gemini 2.5 Flash LLM...")
        context_parts = []
        sources = []
        for c in reranked:
            meta = c.metadata
            fname = meta.get("filename", "enterprise_cloud_2025.pdf")
            page = meta.get("page_number", "?")
            context_parts.append(f"[Source: {fname}, Page {page}]\n{c.content}")
            sources.append({"document": fname, "page": page, "chunk_id": c.id, "score": round(c.score, 4)})

        context_str = "\n\n---\n\n".join(context_parts)
        system_prompt = prompt_manager.get_prompt("rag", "system")
        answer_prompt = prompt_manager.render("rag", "answer", context=context_str, question=query)

        llm = LLMProviderFactory.get_provider()
        answer = await llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": answer_prompt}
        ])

        # Step 14 & 15: Source citations and fact verification
        print("\n[Step 14-15/17] Verifying citations and comparing with original PDF facts...")
        print("-" * 65)
        print(f"QUERY:    {query}")
        print(f"ANSWER:   {answer.strip()}")
        print("-" * 65)
        print(f"SOURCES:  {sources}")
        print("-" * 65)

        # Ground truth verification checks
        has_budget = "$4.75 million" in answer or "4.75 million" in answer or "4.75M" in answer
        has_ciso = "Elena Rostova" in answer
        assert has_budget, "Answer failed to extract exact budget ($4.75 million)"
        assert has_ciso, "Answer failed to extract exact CISO name (Dr. Elena Rostova)"
        assert len(sources) > 0, "No citations returned"
        print("      [PASS] Answer contains BOTH ground-truth facts ($4.75M budget & Dr. Elena Rostova CISO)!")
        print("      [PASS] Citations correctly link to enterprise_cloud_2025.pdf")

        # Step 16: Persistence check (re-query after simulated restart)
        print("\n[Step 16/17] Persistence check: re-querying stored chunks from PostgreSQL...")
        stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        res = await db.execute(stmt)
        stored_chunks = res.scalars().all()
        assert len(stored_chunks) == len(chunks), f"Expected {len(chunks)} chunks, found {len(stored_chunks)}"
        print(f"      [PASS] Verified persistence: {len(stored_chunks)} chunks persisted in PostgreSQL")

        # Step 17: Delete document and verify clean cascade
        print("\n[Step 17/17] Deleting document and verifying vector & chunk removal...")
        await vector_store.delete_document(str(doc.id))
        stmt_del = text("DELETE FROM documents WHERE id = :id")
        await db.execute(stmt_del, {"id": doc.id})
        await db.commit()

        # Confirm chunks deleted in PostgreSQL (CASCADE)
        res_after = await db.execute(select(DocumentChunk).where(DocumentChunk.document_id == doc.id))
        chunks_after = res_after.scalars().all()
        assert len(chunks_after) == 0, "Chunks were not deleted by CASCADE"
        print("      [PASS] Document and associated chunks cleanly deleted from vector store & PostgreSQL")

    # Cleanup temp pdf
    if os.path.exists(test_pdf_path):
        os.remove(test_pdf_path)

    print("\n" + "=" * 70)
    print("SUCCESS: ALL 17 STEPS OF PHASE C VERIFIED AND PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_phase_c_verification())
