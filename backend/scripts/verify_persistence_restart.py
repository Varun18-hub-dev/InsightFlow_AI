"""
Section 8 Verification: Pinecone Vector Persistence Across Container Restart
"""
import asyncio
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

# Clean global GOOGLE_API_KEY conflict
os.environ.pop("GOOGLE_API_KEY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.services.pinecone_service import get_vector_store
from app.llm.embedding_factory import EmbeddingProviderFactory
from app.retrieval.semantic_retriever import SemanticRetriever

async def main():
    print("=" * 70)
    print("SECTION 8: Pinecone Vector Persistence Across Container Restart")
    print("=" * 70)

    run_id = uuid.uuid4().hex[:8]
    test_user_id = f"persist-user-{run_id}"
    test_text = f"Persistence verification payload {run_id}: Project Apollo launched in 2026 with 99.99% uptime."

    vs = get_vector_store()
    embedder = EmbeddingProviderFactory.get_provider()
    retriever = SemanticRetriever()

    # Step 1: Upsert test vector
    print(f"Step 1: Upserting test vector for user {test_user_id}...")
    vec = await embedder.embed_documents([test_text])
    chunk_id = f"chunk-{run_id}"
    await vs.upsert_vectors([{
        "id": chunk_id,
        "values": vec[0],
        "metadata": {
            "user_id": test_user_id,
            "content": test_text,
            "document_id": f"doc-{run_id}",
            "filename": "apollo_spec.txt"
        }
    }])
    time.sleep(3) # Pinecone consistency
    print(f"[PASS] Vector {chunk_id} upserted to Pinecone")

    # Step 2: Query before restart
    print("Step 2: Querying before restart...")
    res_before = await retriever.retrieve("Project Apollo uptime", top_k=1, filters={"user_id": test_user_id})
    assert len(res_before) == 1, "Failed to retrieve vector before restart"
    assert "99.99% uptime" in res_before[0].content
    print(f"[PASS] Retrieved before restart: '{res_before[0].content[:60]}...' (Score: {res_before[0].score:.4f})")

    # Step 3: Restart backend container
    print("Step 3: Restarting insightflow-backend-app container...")
    proc = subprocess.run(
        ["docker", "compose", "restart", "backend"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2])
    )
    print(f"Docker restart output: {proc.stdout.strip()}")
    assert proc.returncode == 0, f"Docker restart failed: {proc.stderr}"

    # Step 4: Wait for health check
    print("Step 4: Waiting for backend health check...")
    import urllib.request, json
    healthy = False
    for _ in range(15):
        time.sleep(2)
        try:
            with urllib.request.urlopen("http://localhost:8000/api/health", timeout=3) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok" and data.get("pinecone") == "connected":
                    healthy = True
                    print(f"[PASS] Health check verified: {data}")
                    break
        except Exception:
            pass
    assert healthy, "Backend did not become healthy within 30s"

    # Step 5 & 6: Query after restart WITHOUT re-ingesting
    print("Step 5 & 6: Querying exact same vector after container restart (NO re-ingestion)...")
    res_after = await retriever.retrieve("Project Apollo uptime", top_k=1, filters={"user_id": test_user_id})
    assert len(res_after) == 1, "Failed to retrieve vector after container restart"
    assert "99.99% uptime" in res_after[0].content
    print(f"[PASS] Retrieved after restart: '{res_after[0].content[:60]}...' (Score: {res_after[0].score:.4f})")

    # Step 7: Clean up
    print("Step 7: Cleanup test vector...")
    await vs.delete_document(f"doc-{run_id}")
    print("[PASS] Cleanup complete. Pinecone persistence verified!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
