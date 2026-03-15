"""Run test queries against the live FastAPI server and log results to JSONL.
Used for testing the API and querying the index.
Run with: python tests/test_query_log.py
Prerequisites:
  - .env file with GOOGLE_API_KEY and PINECONE_API_KEY
  - Pinecone index already populated (run ingest.py first)
  - API server running: uvicorn api:app --reload
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.config import API_BASE_URL, HEALTH_CHECK_TIMEOUT, QUERY_TIMEOUT, TEST_SLEEP_BETWEEN

TESTS_DIR = Path(__file__).parent
OUTPUT_FILE = TESTS_DIR / "query_results.jsonl"

QUESTIONS = [
    "How much did I spend in January 2024?",
    "What did I buy last week?",
    "Show me all receipts from December",
    "Find all Whole Foods receipts",
]


def _check_health() -> bool:
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
        data = resp.json()
        if data.get("status") != "healthy":
            print(f"API unhealthy: {data}")
            return False
        print(f"API healthy -- {data.get('total_vectors', 0)} vectors indexed")
        return True
    except Exception as e:
        print(f"Cannot reach API at {API_BASE_URL}: {e}")
        return False


def _run_query(query: str) -> dict:
    """Send a query and return the full JSON response with timing."""
    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/query",
            json={"query": query},
            timeout=QUERY_TIMEOUT,
        )
        elapsed = round(time.monotonic() - start, 2)
        if resp.status_code == 200:
            return {
                "status": "success",
                "http_code": 200,
                "elapsed_s": elapsed,
                "response": resp.json(),
            }
        return {
            "status": "error",
            "http_code": resp.status_code,
            "elapsed_s": elapsed,
            "error": resp.text,
        }
    except Exception as e:
        elapsed = round(time.monotonic() - start, 2)
        return {
            "status": "exception",
            "http_code": None,
            "elapsed_s": elapsed,
            "error": str(e),
        }


def main():
    print(f"Checking API health at {API_BASE_URL} ...")
    if not _check_health():
        sys.exit(1)

    print(f"\nRunning {len(QUESTIONS)} queries, {TEST_SLEEP_BETWEEN}s sleep between each")
    print(f"Results will be appended to {OUTPUT_FILE}\n")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    with OUTPUT_FILE.open("a") as f:
        for i, question in enumerate(QUESTIONS, 1):
            print(f"[{i}/{len(QUESTIONS)}] {question}")
            result = _run_query(question)

            record = {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "status": result["status"],
                "http_code": result["http_code"],
                "elapsed_s": result["elapsed_s"],
            }

            if result["status"] == "success":
                resp = result["response"]
                record["answer"] = resp.get("answer")
                record["total_results"] = resp.get("query_metadata", {}).get("total_results")
                record["parsed_filters"] = resp.get("query_metadata", {}).get("parsed_filters")
                record["chunk_type"] = resp.get("query_metadata", {}).get("chunk_type")
                record["aggregation_type"] = resp.get("query_metadata", {}).get("aggregation_type")
                record["aggregation"] = resp.get("aggregation")
                record["receipts_count"] = len(resp.get("receipts", []))
                print(f"  -> OK ({result['elapsed_s']}s) -- {record['total_results']} results")
                if record["answer"]:
                    preview = record["answer"][:120].replace("\n", " ")
                    print(f"  -> {preview}...")
            else:
                record["error"] = result.get("error")
                print(f"  -> FAIL ({result['status']}): {result.get('error', '')[:120]}")

            f.write(json.dumps(record) + "\n")

            if i < len(QUESTIONS):
                print(f"  sleeping {TEST_SLEEP_BETWEEN}s ...")
                time.sleep(TEST_SLEEP_BETWEEN)

    print(f"\nDone. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
