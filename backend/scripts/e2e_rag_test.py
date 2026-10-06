"""
Standalone end-to-end RAG test — no Docker, no Postgres required.
Tests: Embed → InMemoryVectorStore → Retrieve → Rerank → Gemini LLM → Answer with citations

Run from: insightflow-ai/backend/
"""
import asyncio
import os
import sys
import time

# Remove conflicting env key
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("EMBEDDING_PROVIDER", "gemini")
os.environ.setdefault("PINECONE_API_KEY", "")  # Force InMemory fallback
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")  # Not used in this test
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")

sys.path.insert(0, ".")

from app.core.config import settings

if not settings.GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY must be set in environment or .env file.")
    sys.exit(1)

SAMPLE_DOCUMENT = """
InsightFlow AI Platform — Q3 2024 Performance Report

Executive Summary:
InsightFlow AI processed 2.4 million queries in Q3 2024, achieving an average response
latency of 320ms. The hybrid retrieval system demonstrated a 94.7% accuracy rate on
the internal evaluation benchmark.

Revenue Impact:
The platform generated $1.2 million in cost savings by automating document review
workflows that previously required 3 full-time analysts. Customer satisfaction scores
improved by 23% after deploying the AI-powered search feature.

Technical Metrics:
- Documents indexed: 847,000
- Average chunk retrieval score: 0.87
- LLM provider: Gemini 2.5 Flash
- Reranker type: Lightweight (no GPU)
- P99 latency: 1,240ms
- Cache hit rate: 67%

Upcoming Features (Q4 2024):
- Multi-modal document support (images in PDFs)
- SQL agent integration for database queries
- Real-time streaming responses via SSE
- MLflow experiment tracking dashboard
"""

async def run_e2e_test():
    print("=" * 60)
    print("InsightFlow AI — End-to-End RAG Test")
    print("=" * 60)

    # ── Step 1: Chunk the document ──────────────────────────────
    print("\n[1/6] Chunking document...")
    from app.ingestion.chunker import DocumentChunker
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk(SAMPLE_DOCUMENT, metadata={"filename": "q3_report.txt", "document_id": "doc-001", "user_id": "user-test"})
    print(f"      ✓ {len(chunks)} chunks created")
    for i, c in enumerate(chunks):
        print(f"        Chunk {i}: {c['content'][:60].strip()}...")

    # ── Step 2: Embed chunks ────────────────────────────────────
    print("\n[2/6] Embedding chunks with Gemini text-embedding-004...")
    t0 = time.time()
    from app.llm.embedding_factory import EmbeddingProviderFactory
    embedder = EmbeddingProviderFactory.get_provider()
    texts = [c["content"] for c in chunks]
    embeddings = await embedder.embed_documents(texts)
    elapsed = round(time.time() - t0, 2)
    print(f"      ✓ {len(embeddings)} embeddings created, dim={len(embeddings[0])}, took {elapsed}s")

    # ── Step 3: Store in InMemoryVectorStore ────────────────────
    print("\n[3/6] Storing in InMemoryVectorStore (Pinecone fallback)...")
    from app.services.pinecone_service import InMemoryVectorStore
    store = InMemoryVectorStore()
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        meta = chunk["metadata"].copy()
        meta["content"] = chunk["content"]
        meta["chunk_index"] = i
        vectors.append({"id": f"chunk-{i}", "values": emb, "metadata": meta})
    await store.upsert(vectors)
    print(f"      ✓ {len(vectors)} vectors stored in InMemoryVectorStore")

    # ── Step 4: Retrieve with hybrid search ─────────────────────
    print("\n[4/6] Retrieving with semantic search...")
    query = "What was the revenue impact and cost savings?"
    query_embedding = await embedder.embed_query(query)
    results = await store.query(query_embedding, top_k=4, filter={})
    print(f"      ✓ Retrieved {len(results)} chunks")
    for r in results:
        print(f"        score={r['score']:.4f} | {r['metadata']['content'][:70].strip()}...")

    # ── Step 5: Rerank ──────────────────────────────────────────
    print("\n[5/6] Reranking with LightweightReranker...")
    from app.retrieval.base_retriever import RetrievedChunk
    from app.reranking.lightweight_reranker import LightweightReranker

    retrieved_chunks = [
        RetrievedChunk(
            id=r["id"],
            content=r["metadata"]["content"],
            score=r["score"],
            metadata=r["metadata"],
        )
        for r in results
    ]
    reranker = LightweightReranker()
    reranked = await reranker.rerank(query, retrieved_chunks, top_k=3)
    print(f"      ✓ Reranked to top {len(reranked)} chunks")
    for r in reranked:
        print(f"        rank={r.rank} score={r.score:.4f} | {r.content[:70].strip()}...")

    # ── Step 6: Generate answer with Gemini ─────────────────────
    print("\n[6/6] Generating answer with Gemini 2.5 Flash...")
    context_parts = []
    sources = []
    for chunk in reranked:
        fname = chunk.metadata.get("filename", "doc")
        context_parts.append(f"[Source: {fname}]\n{chunk.content}")
        sources.append({"document": fname, "chunk_id": chunk.id, "score": round(chunk.score, 4)})

    context = "\n\n---\n\n".join(context_parts)
    system_prompt = (
        "You are InsightFlow AI, an enterprise document assistant. "
        "Answer questions using ONLY the provided document context. "
        "Always cite your sources. If the answer is not in the context, say so."
    )
    user_prompt = f"""Context from documents:
{context}

Question: {query}

Provide a specific, cited answer."""

    from app.llm.factory import LLMProviderFactory
    t0 = time.time()
    llm = LLMProviderFactory.get_provider()
    answer = await llm.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    elapsed = round(time.time() - t0, 2)

    print(f"\n{'=' * 60}")
    print("QUERY:", query)
    print("-" * 60)
    print("ANSWER:", answer)
    print("-" * 60)
    print(f"SOURCES ({len(sources)}):")
    for s in sources:
        print(f"  - {s['document']} | chunk {s['chunk_id']} | score {s['score']}")
    print(f"LATENCY: {elapsed}s")
    print("=" * 60)
    print("\n✅ END-TO-END TEST PASSED — Full RAG pipeline working!")

if __name__ == "__main__":
    asyncio.run(run_e2e_test())
