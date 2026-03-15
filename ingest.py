"""
One-time CLI script: parse all 100 receipts, chunk, embed, and upsert to Pinecone.

Run with:
    python ingest.py

To delete the Pinecone index before re-ingesting (clean slate):
    python -c "from src.config import PINECONE_INDEX; from src.vectorstore.pinecone_client import get_pinecone_client; pc = get_pinecone_client(); pc.delete_index(PINECONE_INDEX); print('Index deleted.')"
"""

from __future__ import annotations

import logging
import sys

from src.config import RECEIPT_DIR
from src.ingestion.chunker import chunk_receipts
from src.ingestion.parser import ReceiptParser
from src.vectorstore.indexer import upsert_documents
from src.vectorstore.pinecone_client import (
    ensure_index,
    get_index_stats,
    get_pinecone_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting receipt ingestion pipeline")

    pc = get_pinecone_client()
    index_name = ensure_index(pc)
    logger.info("Using Pinecone index: %s", index_name)

    parser = ReceiptParser()
    receipts, failures = parser.parse_directory(RECEIPT_DIR)

    if failures:
        logger.warning("Failed to parse %d receipts:", len(failures))
        for filename, error in failures:
            logger.warning("  %s: %s", filename, error)

    if not receipts:
        logger.error("No receipts parsed successfully. Exiting.")
        sys.exit(1)

    documents = chunk_receipts(receipts)
    receipt_chunks = sum(1 for d in documents if d.metadata["chunk_type"] == "receipt")
    item_chunks = sum(1 for d in documents if d.metadata["chunk_type"] == "item")
    logger.info(
        "Created %d documents (%d receipt-level, %d item-level)",
        len(documents),
        receipt_chunks,
        item_chunks,
    )

    upserted = upsert_documents(documents, index_name=index_name, pc=pc)
    logger.info("Upserted %d vectors to Pinecone", upserted)

    stats = get_index_stats(pc, index_name)
    logger.info("Index stats: %s", stats)
    logger.info(
        "Ingestion complete. %d/%d receipts processed. %d skipped.",
        len(receipts),
        len(receipts) + len(failures),
        len(failures),
    )


if __name__ == "__main__":
    main()
