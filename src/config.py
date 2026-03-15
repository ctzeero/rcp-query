"""Centralized configuration -- single source of truth for all settings.

Every tunable value lives here, read once from environment variables
(with sensible defaults).  Other modules import from ``src.config``
instead of calling ``os.getenv`` themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECEIPT_DIR = PROJECT_ROOT / "receipt_samples"
RECEIPT_FILE_GLOB = "receipt_*.txt"

# ---------------------------------------------------------------------------
# API keys (required -- fail fast when missing)
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")

REQUIRED_ENV_VARS = ("GOOGLE_API_KEY", "PINECONE_API_KEY")

# ---------------------------------------------------------------------------
# Gemini LLM
# ---------------------------------------------------------------------------
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE_RESPONSE: float = float(os.getenv("LLM_TEMPERATURE_RESPONSE", "0.1"))
LLM_TEMPERATURE_PARSING: float = float(os.getenv("LLM_TEMPERATURE_PARSING", "0"))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "receipt-intelligence")
PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
INDEX_READY_POLL_INTERVAL: int = int(os.getenv("INDEX_READY_POLL_INTERVAL", "5"))
INDEX_READY_TIMEOUT: int = int(os.getenv("INDEX_READY_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# Indexer (batch upsert)
# ---------------------------------------------------------------------------
UPSERT_BATCH_SIZE: int = int(os.getenv("UPSERT_BATCH_SIZE", "50"))
UPSERT_MAX_RETRIES: int = int(os.getenv("UPSERT_MAX_RETRIES", "3"))
UPSERT_BASE_BACKOFF: float = float(os.getenv("UPSERT_BASE_BACKOFF", "2.0"))
UPSERT_VERIFY_ATTEMPTS: int = int(os.getenv("UPSERT_VERIFY_ATTEMPTS", "6"))
UPSERT_VERIFY_SLEEP: int = int(os.getenv("UPSERT_VERIFY_SLEEP", "5"))

# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------
REFERENCE_DATE: str = os.getenv(
    "REFERENCE_DATE",
    __import__("datetime").date.today().isoformat(),
)
DATASET_END_DATE: str = os.getenv("DATASET_END_DATE", "2024-01-31")
DATASET_START_DATE: str = os.getenv("DATASET_START_DATE", "2023-11-01")
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "20"))
SOFT_FILTER_MULTIPLIER: int = int(os.getenv("SOFT_FILTER_MULTIPLIER", "3"))
FUZZY_MATCH_THRESHOLD: int = int(os.getenv("FUZZY_MATCH_THRESHOLD", "70"))
METADATA_QUERY_LIMIT: int = int(os.getenv("METADATA_QUERY_LIMIT", "1000"))

# ---------------------------------------------------------------------------
# Retrieval / display
# ---------------------------------------------------------------------------
MAX_DISPLAY_RECEIPTS: int = int(os.getenv("MAX_DISPLAY_RECEIPTS", "10"))
CONTENT_PREVIEW_LENGTH: int = int(os.getenv("CONTENT_PREVIEW_LENGTH", "200"))

# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------
MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "500"))
MAX_QUERY_SENTENCES: int = int(os.getenv("MAX_QUERY_SENTENCES", "3"))

# ---------------------------------------------------------------------------
# Streamlit / frontend
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
HEALTH_CHECK_TIMEOUT: float = float(os.getenv("HEALTH_CHECK_TIMEOUT", "10.0"))
QUERY_TIMEOUT: float = float(os.getenv("QUERY_TIMEOUT", "60.0"))
MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
TEST_SLEEP_BETWEEN: int = int(os.getenv("TEST_SLEEP_BETWEEN", "5"))
