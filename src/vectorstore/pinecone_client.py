from __future__ import annotations

import logging
import time
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    INDEX_READY_POLL_INTERVAL,
    INDEX_READY_TIMEOUT,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX,
    PINECONE_METRIC,
    PINECONE_REGION,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a cached GoogleGenerativeAIEmbeddings instance for document indexing."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=EMBEDDING_DIMENSION,
    )


@lru_cache(maxsize=1)
def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a cached GoogleGenerativeAIEmbeddings instance for query-time retrieval."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=EMBEDDING_DIMENSION,
    )

def get_pinecone_client() -> Pinecone:
    """Return a Pinecone client instance."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    return Pinecone(api_key=PINECONE_API_KEY)

def _wait_until_ready(pc: Pinecone, index_name: str) -> None:
    """Block until the Pinecone index status is 'Ready'."""
    deadline = time.time() + INDEX_READY_TIMEOUT
    while time.time() < deadline:
        desc = pc.describe_index(index_name)
        status = desc.status
        if status.get("ready"):
            logger.info("Index '%s' is ready.", index_name)
            return
        logger.info(
            "Waiting for index '%s' to be ready (state=%s)...",
            index_name,
            status.get("state", "unknown"),
        )
        time.sleep(INDEX_READY_POLL_INTERVAL)
    raise TimeoutError(
        f"Index '{index_name}' not ready after {INDEX_READY_TIMEOUT}s"
    )

def ensure_index(pc: Pinecone, index_name: str | None = None) -> str:
    """Create the Pinecone index if it doesn't already exist.

    Blocks until the index reports ready so upserts aren't silently dropped.
    Returns the index name.
    """
    index_name = index_name or PINECONE_INDEX

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info("Creating Pinecone index '%s' (dim=%d)", index_name, EMBEDDING_DIMENSION)
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        logger.info("Index '%s' created. Waiting for it to become ready...", index_name)
        _wait_until_ready(pc, index_name)
    else:
        logger.info("Index '%s' already exists.", index_name)

    return index_name

def get_index_stats(pc: Pinecone, index_name: str | None = None) -> dict:
    """Return index statistics."""
    index_name = index_name or PINECONE_INDEX
    index = pc.Index(index_name)
    return index.describe_index_stats().to_dict()
