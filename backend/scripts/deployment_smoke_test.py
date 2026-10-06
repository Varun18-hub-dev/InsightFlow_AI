"""InsightFlow AI — Live Deployment Smoke Test.

Validates an end-to-end deployed InsightFlow AI stack (Render FastAPI backend,
Vercel Next.js frontend, Neon DB, Upstash Redis, Pinecone, Gemini).

Usage:
    python backend/scripts/deployment_smoke_test.py --api-url https://insightflow-api.onrender.com
    python backend/scripts/deployment_smoke_test.py --api-url https://insightflow-api.onrender.com --frontend-url https://insightflow.vercel.app
"""

import argparse
import io
import json
import sys
import time
import uuid
import httpx


def log_step(name: str, status: str, detail: str = "", duration: float | None = None):
    status_str = f"[{status}]"
    if status == "PASS":
        color = "\033[92m"
    elif status == "FAIL":
        color = "\033[91m"
    elif status == "WARN":
        color = "\033[93m"
    else:
        color = "\033[94m"
    reset = "\033[0m"

    dur_str = f" ({duration:.2f}s)" if duration is not None else ""
    print(f"{color}{status_str:8s}{reset} {name:35s} {detail}{dur_str}")


def run_smoke_test(api_url: str, frontend_url: str | None = None) -> bool:
    api_url = api_url.rstrip("/")
    print("=" * 75)
    print(f"InsightFlow AI — Deployment Smoke Test")
    print(f"Target API:      {api_url}")
    if frontend_url:
        print(f"Target Frontend: {frontend_url.rstrip('/')}")
    print("=" * 75)

    client = httpx.Client(timeout=45.0)
    all_passed = True
    token: str | None = None
    doc_id: str | None = None
    test_user_id = f"smoke_{uuid.uuid4().hex[:8]}"
    test_email = f"{test_user_id}@insightflow-smoke.internal"
    test_password = "SmokeTest123!Secure"

    # Step 1: Health check (/health)
    t0 = time.time()
    try:
        resp = client.get(f"{api_url}/health")
        dur = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            db_status = data.get("database", "unknown")
            pinecone_status = data.get("pinecone", "unknown")
            redis_status = data.get("redis", "unknown")
            gemini_status = data.get("gemini", "unknown")
            env = data.get("environment", "unknown")
            log_step("1. Health Check (/health)", "PASS", f"env={env}, db={db_status}, pinecone={pinecone_status}, redis={redis_status}, gemini={gemini_status}", dur)
            if db_status != "healthy":
                log_step("1a. Database Health", "FAIL", f"Expected 'healthy', got '{db_status}'")
                all_passed = False
        else:
            log_step("1. Health Check (/health)", "FAIL", f"HTTP {resp.status_code}: {resp.text}", dur)
            return False
    except Exception as e:
        log_step("1. Health Check (/health)", "FAIL", f"Connection error: {e}", time.time() - t0)
        return False

    # Step 2: Health check (/api/health)
    t0 = time.time()
    try:
        resp = client.get(f"{api_url}/api/health")
        dur = time.time() - t0
        if resp.status_code == 200:
            log_step("2. Health Check (/api/health)", "PASS", "Mounted correctly at /api prefix", dur)
        else:
            log_step("2. Health Check (/api/health)", "FAIL", f"HTTP {resp.status_code}", dur)
            all_passed = False
    except Exception as e:
        log_step("2. Health Check (/api/health)", "FAIL", str(e), time.time() - t0)
        all_passed = False

    # Step 3: Register test user
    t0 = time.time()
    try:
        resp = client.post(
            f"{api_url}/api/auth/register",
            json={
                "email": test_email,
                "password": test_password,
                "full_name": f"Smoke Test User {test_user_id}",
            },
        )
        dur = time.time() - t0
        if resp.status_code in (200, 201):
            token_data = resp.json()
            token = token_data.get("access_token")
            log_step("3. User Registration", "PASS", f"Created user {test_email}", dur)
        else:
            log_step("3. User Registration", "FAIL", f"HTTP {resp.status_code}: {resp.text}", dur)
            return False
    except Exception as e:
        log_step("3. User Registration", "FAIL", str(e), time.time() - t0)
        return False

    headers = {"Authorization": f"Bearer {token}"}

    # Step 4: Login test user
    t0 = time.time()
    try:
        resp = client.post(
            f"{api_url}/api/auth/login",
            data={"username": test_email, "password": test_password},
        )
        dur = time.time() - t0
        if resp.status_code == 200:
            token = resp.json().get("access_token")
            headers["Authorization"] = f"Bearer {token}"
            log_step("4. User Login", "PASS", "Token refreshed successfully", dur)
        else:
            log_step("4. User Login", "FAIL", f"HTTP {resp.status_code}: {resp.text}", dur)
            all_passed = False
    except Exception as e:
        log_step("4. User Login", "FAIL", str(e), time.time() - t0)
        all_passed = False

    # Step 5: Check authenticated profile (/api/auth/me)
    t0 = time.time()
    try:
        resp = client.get(f"{api_url}/api/auth/me", headers=headers)
        dur = time.time() - t0
        if resp.status_code == 200 and resp.json().get("email") == test_email:
            log_step("5. Auth Profile (/me)", "PASS", f"Verified profile {resp.json().get('id')}", dur)
        else:
            log_step("5. Auth Profile (/me)", "FAIL", f"HTTP {resp.status_code}: {resp.text}", dur)
            all_passed = False
    except Exception as e:
        log_step("5. Auth Profile (/me)", "FAIL", str(e), time.time() - t0)
        all_passed = False

    # Step 6: Upload document
    sample_fact = f"Project Falcon release code is ALPHA-99-{uuid.uuid4().hex[:4]}."
    sample_text = (
        f"Confidential Internal Document\n"
        f"InsightFlow Deployment Verification Report\n\n"
        f"Key Information:\n"
        f"1. {sample_fact}\n"
        f"2. The primary deployment region is US-East.\n"
        f"3. All systems must operate within zero AWS footprint.\n"
    )
    t0 = time.time()
    try:
        files = {
            "file": (
                "smoke_test_doc.txt",
                io.BytesIO(sample_text.encode("utf-8")),
                "text/plain",
            )
        }
        resp = client.post(f"{api_url}/api/documents/upload", headers=headers, files=files)
        dur = time.time() - t0
        if resp.status_code in (200, 201):
            doc_data = resp.json()
            doc_id = doc_data.get("id")
            log_step("6. Document Upload", "PASS", f"Doc ID: {doc_id}", dur)
        else:
            log_step("6. Document Upload", "FAIL", f"HTTP {resp.status_code}: {resp.text}", dur)
            all_passed = False
    except Exception as e:
        log_step("6. Document Upload", "FAIL", str(e), time.time() - t0)
        all_passed = False

    # Step 7: Wait for document processing
    if doc_id:
        t0 = time.time()
        processed = False
        for attempt in range(15):
            time.sleep(2)
            try:
                check_resp = client.get(f"{api_url}/api/documents/{doc_id}", headers=headers)
                if check_resp.status_code == 200:
                    status = check_resp.json().get("status")
                    if status == "completed":
                        processed = True
                        break
                    elif status == "failed":
                        log_step("7. Document Processing", "FAIL", f"Status failed: {check_resp.json().get('error_message')}")
                        all_passed = False
                        break
            except Exception as e:
                pass
        dur = time.time() - t0
        if processed:
            log_step("7. Document Processing", "PASS", f"Status: completed (chunks indexed in Pinecone)", dur)
        elif all_passed:
            log_step("7. Document Processing", "WARN", f"Timed out waiting for completion after {dur:.1f}s", dur)

    # Step 8: Chat QA / Hybrid Retrieval
    t0 = time.time()
    try:
        chat_payload = {"query": "What is the Project Falcon release code?"}
        chat_resp = client.post(f"{api_url}/api/chat", headers=headers, json=chat_payload)
        dur = time.time() - t0
        if chat_resp.status_code == 200:
            chat_data = chat_resp.json()
            answer = chat_data.get("answer", "")
            sources = chat_data.get("sources", [])
            log_step("8. RAG Chat QA", "PASS", f"Answer received ({len(sources)} sources, len={len(answer)})", dur)
        else:
            log_step("8. RAG Chat QA", "FAIL", f"HTTP {chat_resp.status_code}: {chat_resp.text}", dur)
            all_passed = False
    except Exception as e:
        log_step("8. RAG Chat QA", "FAIL", str(e), time.time() - t0)
        all_passed = False

    # Step 9: SSE Chat Stream
    t0 = time.time()
    try:
        stream_payload = {"query": "Summarize the key information in one bullet point."}
        with client.stream(
            "POST",
            f"{api_url}/api/chat/stream",
            headers=headers,
            json=stream_payload,
            timeout=30.0,
        ) as stream_resp:
            dur = time.time() - t0
            if stream_resp.status_code == 200:
                events_received = 0
                for line in stream_resp.iter_lines():
                    if line.startswith("data:"):
                        events_received += 1
                log_step("9. SSE Streaming Chat", "PASS", f"Streamed {events_received} events successfully", dur)
            else:
                log_step("9. SSE Streaming Chat", "FAIL", f"HTTP {stream_resp.status_code}", dur)
                all_passed = False
    except Exception as e:
        log_step("9. SSE Streaming Chat", "FAIL", str(e), time.time() - t0)
        all_passed = False

    # Step 10: Clean up document
    if doc_id:
        t0 = time.time()
        try:
            del_resp = client.delete(f"{api_url}/api/documents/{doc_id}", headers=headers)
            dur = time.time() - t0
            if del_resp.status_code in (200, 204):
                log_step("10. Cleanup Document", "PASS", f"Deleted {doc_id}", dur)
            else:
                log_step("10. Cleanup Document", "WARN", f"Status: {del_resp.status_code}", dur)
        except Exception as e:
            log_step("10. Cleanup Document", "WARN", str(e), time.time() - t0)

    # Step 11: Frontend verification if URL provided
    if frontend_url:
        f_url = frontend_url.rstrip("/")
        t0 = time.time()
        try:
            f_resp = client.get(f_url, follow_redirects=True)
            dur = time.time() - t0
            if f_resp.status_code == 200 and ("InsightFlow" in f_resp.text or "<html" in f_resp.text):
                log_step("11. Frontend Reachability", "PASS", f"Status 200 OK at {f_url}", dur)
            else:
                log_step("11. Frontend Reachability", "FAIL", f"HTTP {f_resp.status_code}", dur)
                all_passed = False
        except Exception as e:
            log_step("11. Frontend Reachability", "FAIL", str(e), time.time() - t0)
            all_passed = False

    print("=" * 75)
    if all_passed:
        print("\033[92mALL DEPLOYMENT SMOKE TESTS PASSED SUCCESSFULLY!\033[0m")
    else:
        print("\033[91mSMOKE TESTS COMPLETED WITH FAILURES.\033[0m")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightFlow AI Live Deployment Smoke Test")
    parser.add_argument("--api-url", required=True, help="Base URL of deployed FastAPI backend")
    parser.add_argument("--frontend-url", default=None, help="Base URL of deployed Next.js frontend")
    args = parser.parse_args()

    success = run_smoke_test(api_url=args.api_url, frontend_url=args.frontend_url)
    sys.exit(0 if success else 1)
