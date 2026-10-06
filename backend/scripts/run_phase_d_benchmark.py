"""
Phase D: Retrieval & Generation Evaluation Benchmark
Compares:
  - Configuration A: Semantic Only
  - Configuration B: Hybrid (Keyword FTS + Semantic RRF + Lightweight Reranker)
Computes:
  - Hit Rate
  - Recall@5
  - Precision@5
  - MRR (Mean Reciprocal Rank)
  - Faithfulness
  - Answer Relevance
  - Latency (s)
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

# Add backend and project root to sys.path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(backend_path.parent))

from sqlalchemy import select, text
from app.core.config import settings

if not settings.GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY must be set in environment or .env file.")
    sys.exit(1)
from app.db.base import AsyncSessionLocal
from app.db.models import User, Document, DocumentChunk
from app.core.security import hash_password
from app.llm.embedding_factory import EmbeddingProviderFactory
from app.services.pinecone_service import get_vector_store
from app.retrieval.semantic_retriever import SemanticRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranking.lightweight_reranker import LightweightReranker
from app.llm.factory import LLMProviderFactory
from app.core.prompt_manager import prompt_manager
from evaluation.metrics.retrieval_metrics import RecallAtK, PrecisionAtK, MeanReciprocalRank, HitRate
from evaluation.metrics.generation_metrics import FaithfulnessMetric, AnswerRelevanceMetric

# Evaluation corpus for the 15 sample_eval.json questions
CORPUS = [
    {
        "filename": "annual_report.pdf",
        "content": "InsightFlow Annual Financial Report. Company revenue in Q4 reached $14.2 million, representing an 18% year-over-year increase across enterprise subscriptions and AI services."
    },
    {
        "filename": "company_overview.docx",
        "content": "Executive Summary & Organization: Sarah Chen serves as the Chief Executive Officer (CEO) of InsightFlow AI, leading international strategic expansion and corporate strategy."
    },
    {
        "filename": "risk_assessment_2023.pdf",
        "content": "Corporate Risk Assessment 2023: The primary enterprise risk factors identified are market volatility in AI cloud spending and global supply chain disruptions affecting GPU hardware availability."
    },
    {
        "filename": "roadmap.pdf",
        "content": "Product Engineering Roadmap 2025: The next major product launch is scheduled for Q3, introducing multi-modal RAG search, voice synthesis, and autonomous workflow actions."
    },
    {
        "filename": "hr_metrics.csv",
        "content": "Human Resources Annual Review: Employee turnover rate decreased to 5% this year due to improved talent retention, competitive equity compensation, and flexible workplace policies."
    },
    {
        "filename": "pricing_strategy.pdf",
        "content": "Commercial Strategy: InsightFlow uses a tiered subscription-based pricing model with Starter, Professional, and Enterprise plans based on retrieval volume and seats."
    },
    {
        "filename": "compliance_guide.pdf",
        "content": "Regulatory Compliance Handbook: GDPR compliance requirements mandate strict EU data localization, granular user consent tracking, and the right to erasure within 30 days."
    },
    {
        "filename": "budget_2024.xlsx",
        "content": "Financial Budget Allocation 2024: The marketing budget for next year has been increased by 15% to $2.0 million to expand brand awareness and enterprise pipeline."
    },
    {
        "filename": "market_analysis.pdf",
        "content": "Competitive Intelligence: Competitor A and Competitor B currently dominate the legacy enterprise search market, while InsightFlow leads in autonomous hybrid retrieval."
    },
    {
        "filename": "board_minutes_jan.docx",
        "content": "Board Meeting Minutes January: The key strategic takeaways were prioritizing international expansion in APAC and EMEA, and increasing R&D investment by 25%."
    },
    {
        "filename": "support_metrics.pdf",
        "content": "Customer Success Operations: Customer support performance is measured using Customer Satisfaction (CSAT) scores (target >92%) and average first-response resolution time under 15 minutes."
    },
    {
        "filename": "architecture_spec.md",
        "content": "Technical Architecture Specification: The next-generation platform tech stack uses Next.js 14 React frontend, FastAPI backend services, Python 3.11, PostgreSQL, Redis, and vector stores."
    },
    {
        "filename": "esg_report.pdf",
        "content": "Environmental Sustainability Report: InsightFlow is committed to achieving net-zero carbon neutrality by 2030 through green data center partnerships and energy-efficient inference."
    },
    {
        "filename": "ip_portfolio.pdf",
        "content": "Intellectual Property Summary: The company currently holds 45 active patents and 12 pending provisional patent applications in vector search indexing and neural reranking."
    },
    {
        "filename": "employee_handbook.pdf",
        "content": "Workplace Policy Guide: Employees are permitted to work remotely up to 3 days per week, subject to departmental scheduling and team core collaboration hours."
    }
]


async def run_benchmark():
    print("=" * 75)
    print("PHASE D & E: RETRIEVAL & GENERATION BENCHMARK EVALUATION")
    print("=" * 75)

    dataset_path = backend_path.parent / "evaluation" / "datasets" / "sample_eval.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    print(f"Loaded evaluation dataset: {len(dataset)} questions from {dataset_path.name}")

    async with AsyncSessionLocal() as db:
        # Create Eval User
        eval_email = f"benchmark_eval_{int(time.time())}@insightflow.ai"
        user = User(
            id=uuid.uuid4(),
            email=eval_email,
            full_name="Benchmark Evaluator",
            hashed_password=hash_password("EvalPass123!"),
            is_active=True
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        user_id = str(user.id)
        print(f"[1/4] Created benchmark evaluation user: {eval_email}")

        # Ingest 15 corpus documents
        print(f"[2/4] Ingesting {len(CORPUS)} evaluation documents into PostgreSQL & VectorStore...")
        embedding_provider = EmbeddingProviderFactory.get_provider()
        vector_store = get_vector_store()
        
        texts = [doc["content"] for doc in CORPUS]
        t0 = time.time()
        all_embeddings = await embedding_provider.embed_documents(texts)
        print(f"      Generated {len(all_embeddings)} embeddings in {time.time()-t0:.2f}s")

        doc_ids = []
        vectors = []
        for i, item in enumerate(CORPUS):
            d_id = uuid.uuid4()
            doc_ids.append(d_id)
            doc_record = Document(
                id=d_id,
                user_id=user.id,
                filename=item["filename"],
                original_filename=item["filename"],
                document_type="text/plain",
                file_path=item["filename"],
                file_size=len(item["content"]),
                status="processed",
                page_count=1,
                chunk_count=1
            )
            db.add(doc_record)

            cid = str(uuid.uuid4())
            meta = {
                "chunk_id": cid,
                "document_id": str(d_id),
                "filename": item["filename"],
                "user_id": user_id,
                "page_number": 1,
                "content": item["content"]
            }
            vectors.append({"id": cid, "values": all_embeddings[i], "metadata": meta})

            chunk_record = DocumentChunk(
                document_id=d_id,
                user_id=user.id,
                chunk_index=0,
                content=item["content"],
                page_number=1,
                chunk_metadata=meta,
                embedding_id=cid
            )
            db.add(chunk_record)

        await db.commit()
        await vector_store.upsert_vectors(vectors)
        print(f"      Ingestion complete. {len(CORPUS)} documents active in test corpus.")

        # Evaluator metric classes
        recall_metric = RecallAtK()
        precision_metric = PrecisionAtK()
        mrr_metric = MeanReciprocalRank()
        hitrate_metric = HitRate()
        faith_metric = FaithfulnessMetric()
        relevance_metric = AnswerRelevanceMetric()

        # Tools
        semantic_retriever = SemanticRetriever()
        keyword_retriever = KeywordRetriever(db)
        hybrid_retriever = HybridRetriever(db)
        reranker = LightweightReranker()
        llm = LLMProviderFactory.get_provider()

        print("\n[3/4] Evaluating Configuration A: Semantic Only...")
        results_sem = {"hit_rate": [], "recall@5": [], "precision@5": [], "mrr": [], "latency": []}
        for item in dataset:
            q = item["question"]
            expected = item.get("expected_sources", [])
            t_start = time.time()
            res = await semantic_retriever.retrieve(q, top_k=5, filters={"user_id": user_id})
            lat = time.time() - t_start
            retrieved_files = [c.metadata.get("filename", "") for c in res]
            
            results_sem["hit_rate"].append(hitrate_metric.compute(retrieved_files, expected).value)
            results_sem["recall@5"].append(recall_metric.compute(retrieved_files, expected, k=5).value)
            results_sem["precision@5"].append(precision_metric.compute(retrieved_files, expected, k=5).value)
            results_sem["mrr"].append(mrr_metric.compute(retrieved_files, expected).value)
            results_sem["latency"].append(lat)

        print("\n[4/4] Evaluating Configuration B: Hybrid (Keyword + Semantic + RRF + Reranker)...")
        results_hyb = {
            "hit_rate": [], "recall@5": [], "precision@5": [], "mrr": [],
            "faithfulness": [], "relevance": [], "latency": []
        }
        
        system_prompt = prompt_manager.get_prompt("rag", "system")
        for i, item in enumerate(dataset):
            q = item["question"]
            expected = item.get("expected_sources", [])
            t_start = time.time()
            raw_chunks = await hybrid_retriever.retrieve(q, top_k_semantic=5, top_k_keyword=5, filters={"user_id": user_id})
            reranked = await reranker.rerank(q, raw_chunks, top_k=5)
            lat = time.time() - t_start

            retrieved_files = [c.metadata.get("filename", "") for c in reranked]
            results_hyb["hit_rate"].append(hitrate_metric.compute(retrieved_files, expected).value)
            results_hyb["recall@5"].append(recall_metric.compute(retrieved_files, expected, k=5).value)
            results_hyb["precision@5"].append(precision_metric.compute(retrieved_files, expected, k=5).value)
            results_hyb["mrr"].append(mrr_metric.compute(retrieved_files, expected).value)
            results_hyb["latency"].append(lat)

            # Test generation on a sample (first 3 queries) to evaluate faithfulness & relevance
            if i < 3:
                context_str = "\n".join([f"[{c.metadata.get('filename')}]: {c.content}" for c in reranked[:3]])
                user_msg = prompt_manager.render("rag", "answer", context=context_str, question=q)
                ans = await llm.chat([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ])
                faith = faith_metric.compute(ans, context_str).value
                rel = relevance_metric.compute(q, ans).value
                results_hyb["faithfulness"].append(faith)
                results_hyb["relevance"].append(rel)

        # Compute Averages
        avg_sem = {k: sum(v)/len(v) for k, v in results_sem.items()}
        avg_hyb = {k: sum(v)/len(v) for k, v in results_hyb.items()}

        print("\n" + "=" * 75)
        print(f"{'METRIC':<25} | {'SEMANTIC ONLY':<18} | {'HYBRID (RRF+RERANK)':<18} | {'DELTA':<10}")
        print("-" * 75)
        for m in ["hit_rate", "recall@5", "precision@5", "mrr"]:
            v_a = avg_sem[m]
            v_b = avg_hyb[m]
            diff = v_b - v_a
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            print(f"{m.upper():<25} | {v_a:<18.4f} | {v_b:<18.4f} | {diff_str:<10}")

        print(f"{'LATENCY (s)':<25} | {avg_sem['latency']:<18.4f} | {avg_hyb['latency']:<18.4f} | +{avg_hyb['latency']-avg_sem['latency']:.4f}s")
        print(f"{'FAITHFULNESS':<25} | {'N/A':<18} | {avg_hyb['faithfulness']:<18.4f} | N/A")
        print(f"{'ANSWER RELEVANCE':<25} | {'N/A':<18} | {avg_hyb['relevance']:<18.4f} | N/A")
        print("=" * 75)

        # Save results to evaluation_results.json
        out_file = backend_path.parent / "evaluation_results.json"
        summary_payload = {
            "dataset_size": len(dataset),
            "configurations": {
                "semantic_only": avg_sem,
                "hybrid_rrf_reranker": avg_hyb
            },
            "timestamp": time.time()
        }
        with open(out_file, "w") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\n[PASS] Saved evaluation benchmark results to {out_file}")

        # Cleanup test corpus
        print("\nCleaning up evaluation documents...")
        for did in doc_ids:
            await vector_store.delete_document(str(did))
        await db.execute(text("DELETE FROM documents WHERE user_id = :uid"), {"uid": user.id})
        await db.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        await db.commit()
        print("[PASS] Evaluation database cleanly reverted.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
