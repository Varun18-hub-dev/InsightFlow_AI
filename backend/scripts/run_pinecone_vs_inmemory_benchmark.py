"""
Sections 10 & 11: Comprehensive Real Pinecone vs InMemory Vector Store Benchmark Comparison
InsightFlow AI Production Evaluation Suite
"""
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Clean global GOOGLE_API_KEY conflict
os.environ.pop("GOOGLE_API_KEY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.db.base import AsyncSessionLocal
from app.db.models import User, Document, DocumentChunk
from app.core.security import hash_password
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import DocumentChunker
from app.ingestion.metadata_enricher import MetadataEnricher
from app.llm.embedding_factory import EmbeddingProviderFactory
from app.services.pinecone_service import PineconeService, InMemoryVectorStore
import app.services.pinecone_service as ps_mod
from app.retrieval.semantic_retriever import SemanticRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranking.lightweight_reranker import LightweightReranker
from app.llm.factory import LLMProviderFactory
from app.core.prompt_manager import prompt_manager


BENCHMARK_CORPUS = [
    {
        "filename": "annual_report_2025.txt",
        "content": (
            "Annual Financial Report 2025: Northstar Technologies recorded total annual gross revenue of "
            "$34.2 million, with Q3 revenue standing at $8.4 million and Q4 reaching $9.8 million. Operating margin "
            "improved to 31.4% driven by enterprise software subscription expansion."
        )
    },
    {
        "filename": "executive_leadership.txt",
        "content": (
            "Corporate Governance & Executive Directory: Dr. Aris Thorne serves as Chief Executive Officer. "
            "Dr. Elena Rostova directs technical innovation as Chief Technology Officer, leading the core "
            "infrastructure modernization. Marcus Vance oversees cloud operations as Vice President."
        )
    },
    {
        "filename": "cloud_infrastructure_cost.txt",
        "content": (
            "Cloud Infrastructure Optimization Initiative: The engineering team executed a strategic infrastructure "
            "consolidation across AWS and GCP, resulting in $4.75 million in recurring annual cost savings. Serverless "
            "containerization and reserved instances contributed 60% of the cost reduction."
        )
    },
    {
        "filename": "product_roadmap_2026.txt",
        "content": (
            "Strategic Product Roadmap: The flagship InsightFlow AI enterprise decision platform successfully launched "
            "in March 2026. Version 2.0 includes multi-tenant Pinecone vector search, hybrid PostgreSQL FTS retrieval, "
            "and real-time LLM streaming response capabilities."
        )
    },
    {
        "filename": "security_compliance.txt",
        "content": (
            "Enterprise Security and Governance: InsightFlow AI maintains full SOC2 Type II certification and GDPR "
            "compliance. All customer documents and vector embeddings are isolated by tenant user_id with AES-256 "
            "encryption at rest and TLS 1.3 in transit."
        )
    }
]

BENCHMARK_QUERIES = [
    {
        "query": "What was the Q3 revenue reported for 2025?",
        "expected_doc": "annual_report_2025.txt",
        "expected_fact": "$8.4 million"
    },
    {
        "query": "Who is the Chief Technology Officer leading the modernization?",
        "expected_doc": "executive_leadership.txt",
        "expected_fact": "Dr. Elena Rostova"
    },
    {
        "query": "How much annual cost savings did the optimization initiative achieve?",
        "expected_doc": "cloud_infrastructure_cost.txt",
        "expected_fact": "$4.75 million"
    },
    {
        "query": "When did the InsightFlow platform launch?",
        "expected_doc": "product_roadmap_2026.txt",
        "expected_fact": "March 2026"
    },
    {
        "query": "What encryption standard is used for data at rest?",
        "expected_doc": "security_compliance.txt",
        "expected_fact": "AES-256"
    },
    {
        "query": "Who is the Chief Executive Officer?",
        "expected_doc": "executive_leadership.txt",
        "expected_fact": "Dr. Aris Thorne"
    },
    {
        "query": "What was the operating margin in the annual financial report?",
        "expected_doc": "annual_report_2025.txt",
        "expected_fact": "31.4%"
    },
    {
        "query": "Which cloud providers were consolidated for cost reduction?",
        "expected_doc": "cloud_infrastructure_cost.txt",
        "expected_fact": "AWS and GCP"
    },
    {
        "query": "What compliance certifications does InsightFlow maintain?",
        "expected_doc": "security_compliance.txt",
        "expected_fact": "SOC2 Type II"
    },
    {
        "query": "What was the total annual gross revenue for 2025?",
        "expected_doc": "annual_report_2025.txt",
        "expected_fact": "$34.2 million"
    }
]


async def run_benchmark_for_store(store_name: str, vs_instance, user_id: str, db_session, test_chunks, chunk_embeddings):
    print(f"\n{'='*25} RUNNING BENCHMARK: {store_name.upper()} {'='*25}")
    
    # Upsert chunks into vector store
    vectors = []
    for i, chunk in enumerate(test_chunks):
        cid = chunk["metadata"]["chunk_id"]
        vectors.append({
            "id": cid,
            "values": chunk_embeddings[i],
            "metadata": {
                "user_id": user_id,
                "document_id": chunk["metadata"]["document_id"],
                "content": chunk["content"],
                "chunk_id": cid,
                "filename": chunk["metadata"]["filename"]
            }
        })
    
    t0_upsert = time.time()
    await vs_instance.upsert_vectors(vectors)
    upsert_latency = time.time() - t0_upsert
    print(f"Upserted {len(vectors)} vectors in {upsert_latency:.3f}s")
    
    if store_name == "Pinecone":
        time.sleep(3) # Wait for Pinecone index propagation

    # Patch ps_mod singleton for SemanticRetriever
    saved_key = settings.PINECONE_API_KEY
    if store_name == "Pinecone":
        ps_mod._pinecone_service_instance = vs_instance
        ps_mod._in_memory_store_instance = None
    else:
        settings.PINECONE_API_KEY = ""
        ps_mod._in_memory_store_instance = vs_instance
        ps_mod._pinecone_service_instance = None

    sem_retriever = SemanticRetriever()
    hybrid_retriever = HybridRetriever(db_session)
    reranker = LightweightReranker()
    llm = LLMProviderFactory.get_provider()

    retrieval_latencies = []
    e2e_latencies = []
    hit_count_at_1 = 0
    hit_count_at_5 = 0
    reciprocal_ranks = []
    precisions_at_5 = []
    recalls_at_5 = []

    for item_idx, item in enumerate(BENCHMARK_QUERIES):
        q = item["query"]
        target_doc = item["expected_doc"]
        target_fact = item["expected_fact"]

        # Measure Retrieval Latency & Quality
        t0_retrieval = time.time()
        retrieved = await sem_retriever.retrieve(q, top_k=5, filters={"user_id": user_id})
        ret_latency = time.time() - t0_retrieval
        retrieval_latencies.append(ret_latency)

        # Evaluate Hit Rate and MRR
        doc_matches = [r for r in retrieved if r.metadata.get("filename") == target_doc]
        fact_matches = [r for r in retrieved if target_fact.lower() in r.content.lower()]
        
        hit_at_1 = 1 if (fact_matches and fact_matches[0] == retrieved[0]) else 0
        hit_at_5 = 1 if len(fact_matches) > 0 else 0
        hit_count_at_1 += hit_at_1
        hit_count_at_5 += hit_at_5

        rank = next((idx + 1 for idx, r in enumerate(retrieved) if target_fact.lower() in r.content.lower()), 0)
        rr = (1.0 / rank) if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        # Precision@5 and Recall@5 (1 relevant chunk per query in this corpus)
        prec_5 = len(fact_matches) / 5.0
        rec_5 = 1.0 if len(fact_matches) > 0 else 0.0
        precisions_at_5.append(prec_5)
        recalls_at_5.append(rec_5)

        # Full RAG Pipeline: Hybrid Retrieval + Reranking (+ Generation on sample)
        t0_e2e = time.time()
        hybrid_chunks = await hybrid_retriever.retrieve(
            query=q,
            top_k_semantic=5,
            top_k_keyword=5,
            filters={"user_id": user_id},
            final_top_k=5
        )
        reranked = await reranker.rerank(q, hybrid_chunks, top_k=3)
        context = "\n\n".join([f"[Source: {c.metadata.get('filename')}]\n{c.content}" for c in reranked])
        system_prompt = prompt_manager.get_prompt("rag", "system")
        answer_prompt = prompt_manager.render("rag", "answer", context=context, question=q)
        if item_idx < 2:
            try:
                answer = await llm.chat([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": answer_prompt}
                ])
                await asyncio.sleep(1.0)
            except Exception as e:
                answer = f"LLM throttled: {e}"
        e2e_lat = time.time() - t0_e2e
        e2e_latencies.append(e2e_lat)

        print(f"Query: '{q[:40]}...' -> Rank: {rank} | Ret Lat: {ret_latency:.3f}s | E2E Lat: {e2e_lat:.3f}s")

    settings.PINECONE_API_KEY = saved_key

    n = len(BENCHMARK_QUERIES)
    metrics = {
        "store": store_name,
        "queries_evaluated": n,
        "hit_rate_at_1": round(hit_count_at_1 / n, 4),
        "hit_rate_at_5": round(hit_count_at_5 / n, 4),
        "recall_at_5": round(sum(recalls_at_5) / n, 4),
        "precision_at_5": round(sum(precisions_at_5) / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "avg_retrieval_latency_ms": round((sum(retrieval_latencies) / n) * 1000, 2),
        "p95_retrieval_latency_ms": round(sorted(retrieval_latencies)[int(0.95 * n)] * 1000, 2),
        "avg_e2e_latency_s": round(sum(e2e_latencies) / n, 3),
        "upsert_latency_s": round(upsert_latency, 3)
    }
    return metrics


async def main():
    print("=" * 75)
    print("  INSIGHTFLOW AI — BENCHMARK: PINECONE vs IN-MEMORY VECTOR STORE")
    print("=" * 75)

    run_id = uuid.uuid4().hex[:8]
    embed_provider = EmbeddingProviderFactory.get_provider()

    async with AsyncSessionLocal() as session:
        # Create test user
        user = User(
            id=uuid.uuid4(),
            email=f"benchmark_user_{run_id}@insightflow.ai",
            full_name="Benchmark Suite",
            hashed_password=hash_password("Bench123!"),
            is_active=True
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        # Ingest documents into DB
        test_chunks = []
        texts_to_embed = []
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

        for item in BENCHMARK_CORPUS:
            doc = Document(
                id=uuid.uuid4(),
                user_id=user.id,
                filename=item["filename"],
                original_filename=item["filename"],
                document_type="text/plain",
                file_path=item["filename"],
                file_size=len(item["content"]),
                status="completed",
                page_count=1,
                chunk_count=1
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)

            cleaned = TextCleaner.clean(item["content"])
            raw_chunks = chunker.chunk(cleaned, metadata={"page_number": 1})
            enriched = MetadataEnricher.enrich(raw_chunks, doc, str(user.id))

            for c in enriched:
                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    user_id=user.id,
                    chunk_index=c["chunk_index"],
                    content=c["content"],
                    page_number=1,
                    chunk_metadata=c["metadata"],
                    embedding_id=c["metadata"]["chunk_id"]
                )
                session.add(db_chunk)
                test_chunks.append(c)
                texts_to_embed.append(c["content"])

        await session.commit()
        print(f"Ingested {len(BENCHMARK_CORPUS)} benchmark documents ({len(test_chunks)} chunks) into DB.")

        # Generate 3072-dim embeddings
        print("Generating real Gemini 3072-dim embeddings for all chunks...")
        t0_emb = time.time()
        chunk_embeddings = await embed_provider.embed_documents(texts_to_embed)
        print(f"Generated {len(chunk_embeddings)} embeddings in {time.time() - t0_emb:.2f}s")

        # PASS 1: InMemoryVectorStore
        in_memory_store = InMemoryVectorStore()
        in_memory_metrics = await run_benchmark_for_store(
            store_name="InMemory",
            vs_instance=in_memory_store,
            user_id=str(user.id),
            db_session=session,
            test_chunks=test_chunks,
            chunk_embeddings=chunk_embeddings
        )

        # PASS 2: Real PineconeService
        pinecone_store = PineconeService()
        pinecone_metrics = await run_benchmark_for_store(
            store_name="Pinecone",
            vs_instance=pinecone_store,
            user_id=str(user.id),
            db_session=session,
            test_chunks=test_chunks,
            chunk_embeddings=chunk_embeddings
        )

        # Clean up Pinecone vectors
        for doc_item in BENCHMARK_CORPUS:
            pass
        await pinecone_store.delete_document(str(user.id)) # Will use filter user_id

        # SUMMARY COMPARISON TABLE
        print("\n" + "=" * 80)
        print(f"{'METRIC':<30} | {'IN-MEMORY STORE':<20} | {'REAL PINECONE STORE':<20}")
        print("-" * 80)
        comparison_fields = [
            ("Hit Rate @ 1", "hit_rate_at_1", ""),
            ("Hit Rate @ 5", "hit_rate_at_5", ""),
            ("Recall @ 5", "recall_at_5", ""),
            ("Precision @ 5", "precision_at_5", ""),
            ("MRR (Mean Reciprocal Rank)", "mrr", ""),
            ("Avg Retrieval Latency", "avg_retrieval_latency_ms", " ms"),
            ("P95 Retrieval Latency", "p95_retrieval_latency_ms", " ms"),
            ("Avg E2E RAG Latency", "avg_e2e_latency_s", " s"),
            ("Batch Upsert Latency", "upsert_latency_s", " s")
        ]

        for label, key, unit in comparison_fields:
            im_val = f"{in_memory_metrics[key]}{unit}"
            pc_val = f"{pinecone_metrics[key]}{unit}"
            print(f"{label:<30} | {im_val:<20} | {pc_val:<20}")
        print("=" * 80)

        # Save to JSON
        output_file = Path("evaluation_benchmark_pinecone_vs_inmemory.json")
        output_file.write_text(json.dumps({
            "in_memory": in_memory_metrics,
            "pinecone": pinecone_metrics,
            "queries": len(BENCHMARK_QUERIES),
            "corpus_size": len(BENCHMARK_CORPUS)
        }, indent=2))
        print(f"Benchmark results successfully saved to {output_file.resolve()}\n")


if __name__ == "__main__":
    asyncio.run(main())
