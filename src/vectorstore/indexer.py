from __future__ import annotations

import hashlib
import logging
import time

from langchain_core.documents import Document
from pinecone import Pinecone

from src.config import (
    PINECONE_INDEX,
    UPSERT_BASE_BACKOFF,
    UPSERT_BATCH_SIZE,
    UPSERT_MAX_RETRIES,
    UPSERT_VERIFY_ATTEMPTS,
    UPSERT_VERIFY_SLEEP,
)
from src.vectorstore.pinecone_client import get_embeddings, get_pinecone_client

logger = logging.getLogger(__name__)


def _make_vector_id(doc: Document, index: int = 0) -> str:
    """Generate a deterministic ID for a document.

    Uses receipt_id, chunk_type, item_name, *and* a positional index so that
    duplicate item names on the same receipt don't collide.
    """
    key = (
        f"{doc.metadata.get('receipt_id', '')}:"
        f"{doc.metadata.get('chunk_type', '')}:"
        f"{doc.metadata.get('item_name', '')}:"
        f"{index}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:32]

def _embed_with_retry(embeddings, texts: list[str], attempt: int = 0) -> list[list[float]]:
    """Embed texts with exponential backoff on rate limit errors."""
    try:
        return embeddings.embed_documents(texts)
    except Exception as e:
        if attempt >= UPSERT_MAX_RETRIES:
            raise
        err_str = str(e).lower()
        if "429" in err_str or "rate" in err_str or "quota" in err_str:
            wait = UPSERT_BASE_BACKOFF ** (attempt + 1)
            logger.warning("Rate limited, retrying in %.1fs (attempt %d/%d)", wait, attempt + 1, UPSERT_MAX_RETRIES)
            time.sleep(wait)
            return _embed_with_retry(embeddings, texts, attempt + 1)
        raise

def upsert_documents(
    documents: list[Document],
    index_name: str | None = None,
    pc: Pinecone | None = None,
) -> int:
    """Embed and upsert documents to Pinecone in batches.

    Returns the number of vectors successfully upserted.
    """
    if not documents:
        return 0

    pc = pc or get_pinecone_client()
    index_name = index_name or PINECONE_INDEX
    index = pc.Index(index_name)
    embeddings = get_embeddings()

    total_upserted = 0

    for batch_start in range(0, len(documents), UPSERT_BATCH_SIZE):
        batch = documents[batch_start : batch_start + UPSERT_BATCH_SIZE]
        texts = [doc.page_content for doc in batch]

        try:
            vectors = _embed_with_retry(embeddings, texts)
        except Exception as e:
            logger.error(
                "Failed to embed batch starting at %d: %s", batch_start, e
            )
            continue

        upsert_data = []
        for doc_idx, (doc, vec) in enumerate(zip(batch, vectors), start=batch_start):
            vec_id = _make_vector_id(doc, index=doc_idx)
            upsert_data.append({
                "id": vec_id,
                "values": vec,
                "metadata": {
                    **doc.metadata,
                    "text": doc.page_content,
                },
            })

        try:
            index.upsert(vectors=upsert_data)
            total_upserted += len(upsert_data)
            logger.info(
                "Upserted batch %d-%d (%d vectors)",
                batch_start,
                batch_start + len(batch),
                len(upsert_data),
            )
        except Exception as e:
            logger.error("Failed to upsert batch starting at %d: %s", batch_start, e)
            continue

    logger.info("Total vectors upserted: %d/%d", total_upserted, len(documents))

    for attempt in range(UPSERT_VERIFY_ATTEMPTS):
        time.sleep(UPSERT_VERIFY_SLEEP)
        stats = index.describe_index_stats()
        count = stats.get("total_vector_count", 0)
        if count >= total_upserted:
            logger.info("Verified: index contains %d vectors.", count)
            break
        logger.info(
            "Waiting for vectors to be indexed (%d/%d, attempt %d)...",
            count, total_upserted, attempt + 1,
        )
    else:
        logger.warning(
            "Index reports %d vectors after waiting; expected %d. "
            "Vectors may still be indexing.",
            count, total_upserted,
        )

    return total_upserted
