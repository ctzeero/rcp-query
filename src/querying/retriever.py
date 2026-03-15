from __future__ import annotations

import calendar
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from dateutil import parser as dateparser
from langchain_core.documents import Document
from pinecone import Pinecone

from rapidfuzz import fuzz

from src.config import (
    EMBEDDING_DIMENSION,
    FUZZY_MATCH_THRESHOLD,
    MAX_DISPLAY_RECEIPTS,
    METADATA_QUERY_LIMIT,
    PINECONE_INDEX,
    SOFT_FILTER_MULTIPLIER,
)
from src.querying.query_parser import ParsedQuery
from src.vectorstore.pinecone_client import get_pinecone_client, get_query_embeddings


def _sanitize_date_str(iso_str: str) -> str:
    """Strip malformed timezone suffixes that Gemini sometimes emits."""
    cleaned = re.sub(r"(\[UTC\]|Z)+$", "", iso_str.strip())
    if cleaned.endswith("T00:00:00"):
        cleaned = cleaned[:10]
    return cleaned


def _iso_to_epoch(iso_str: str) -> float:
    """Convert an ISO date/datetime string to Unix epoch seconds."""
    iso_str = _sanitize_date_str(iso_str)
    dt = dateparser.parse(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

logger = logging.getLogger(__name__)

_STRING_FILTER_KEYS = {"merchant", "category", "city", "payment_method", "location"}

def _build_pinecone_filter(parsed: ParsedQuery) -> tuple[dict, dict[str, str]]:
    """Convert ParsedQuery filters into a Pinecone metadata filter dict.

    String-valued filters (merchant, category, etc.) are separated out for
    post-retrieval fuzzy matching since Pinecone only supports exact $eq.

    Returns (pinecone_filter, soft_filters) where soft_filters maps
    field names to the user's search term for case-insensitive substring matching.
    """
    pc_filter: dict = {}
    soft_filters: dict[str, str] = {}

    if parsed.chunk_type:
        pc_filter["chunk_type"] = {"$eq": parsed.chunk_type}

    for key, value in parsed.filters.items():
        if isinstance(value, bool):
            pc_filter[key] = {"$eq": value}
        elif isinstance(value, str):
            if key in _STRING_FILTER_KEYS:
                soft_filters[key] = value
            else:
                pc_filter[key] = {"$eq": value}
        elif isinstance(value, (int, float)):
            pc_filter[key] = {"$eq": value}

    if parsed.date_range:
        start = parsed.date_range.start
        end = parsed.date_range.end

        if start and not end and re.fullmatch(r"\d{4}-\d{2}-01", start):
            dt = datetime.strptime(start, "%Y-%m-%d")
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            end = f"{dt.year}-{dt.month:02d}-{last_day:02d}"
            logger.info("Inferred missing end date: %s", end)

        if start and end:
            pc_filter["date"] = {
                "$gte": _iso_to_epoch(start),
                "$lte": _iso_to_epoch(end),
            }
        elif start:
            pc_filter["date"] = {"$gte": _iso_to_epoch(start)}
        elif end:
            pc_filter["date"] = {"$lte": _iso_to_epoch(end)}

    if parsed.price_filter:
        op_map = {"gt": "$gt", "lt": "$lt", "gte": "$gte", "lte": "$lte", "eq": "$eq"}
        field = parsed.price_filter.get("field", "total_amount")
        operator = op_map.get(parsed.price_filter.get("operator", ""), "$gt")
        value = parsed.price_filter.get("value", 0)
        pc_filter[field] = {operator: value}

    if parsed.tip_filter:
        op_map = {"gt": "$gt", "lt": "$lt", "gte": "$gte", "lte": "$lte", "eq": "$eq"}
        operator = op_map.get(parsed.tip_filter.get("operator", ""), "$gt")
        value = parsed.tip_filter.get("value", 0)
        pc_filter["tip_percentage"] = {operator: value}

    return pc_filter, soft_filters

_FOOD_CATEGORIES = {"grocery", "restaurant", "coffee", "fast_food"}

def _fuzzy_match(term: str, stored: str, threshold: int = FUZZY_MATCH_THRESHOLD) -> bool:
    """Check if *term* fuzzy-matches *stored* using rapidfuzz.

    Uses partial_ratio (best substring alignment) so that
    "Akiko Sushi" matches "AKIKO'S SUSHI" and "McDonalds" matches "MCDONALD'S".
    """
    if not term or not stored:
        return False
    return fuzz.partial_ratio(term.lower(), stored.lower()) >= threshold


def _apply_soft_filters(docs: list[Document], soft_filters: dict[str, str]) -> list[Document]:
    """Post-filter documents using fuzzy matching (rapidfuzz)."""
    if not soft_filters:
        return docs
    filtered = []
    for doc in docs:
        match = True
        for key, term in soft_filters.items():
            stored = str(doc.metadata.get(key, ""))
            if not _fuzzy_match(term, stored):
                match = False
                break
        if match:
            filtered.append(doc)
    return filtered


def _rerank_items_by_category(docs: list[Document]) -> list[Document]:
    """Boost item results from food-related categories over non-food stores.

    Items like 'Chicken Breast' appear in synthetic data at hardware/retail
    stores; this re-rank pushes food-category items to the top.
    """
    food_docs = [d for d in docs if d.metadata.get("category", "") in _FOOD_CATEGORIES]
    other_docs = [d for d in docs if d.metadata.get("category", "") not in _FOOD_CATEGORIES]
    return food_docs + other_docs

def _query_pinecone(
    query_text: str,
    pc_filter: dict,
    top_k: int,
    index_name: str,
    pc: Pinecone,
) -> list[Document]:
    """Execute a query against Pinecone and return Documents."""
    embeddings = get_query_embeddings()
    query_vector = embeddings.embed_query(query_text)

    index = pc.Index(index_name)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=pc_filter if pc_filter else None,
    )

    documents = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        text = metadata.pop("text", "")
        doc = Document(
            page_content=text,
            metadata={**metadata, "score": match.get("score", 0.0)},
        )
        documents.append(doc)

    return documents


def _metadata_only_query(
    pc_filter: dict,
    index_name: str,
    pc: Pinecone,
    limit: int = METADATA_QUERY_LIMIT,
) -> list[Document]:
    """Query Pinecone using only metadata filters (for aggregation).

    Uses a dummy vector to satisfy the API while relying entirely on filters.
    """
    dummy_vector = [0.0] * EMBEDDING_DIMENSION
    index = pc.Index(index_name)

    results = index.query(
        vector=dummy_vector,
        top_k=limit,
        include_metadata=True,
        filter=pc_filter if pc_filter else None,
    )

    documents = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        text = metadata.pop("text", "")
        doc = Document(
            page_content=text,
            metadata={**metadata, "score": match.get("score", 0.0)},
        )
        documents.append(doc)

    return documents

def retrieve(
    parsed: ParsedQuery,
    index_name: Optional[str] = None,
    pc: Optional[Pinecone] = None,
) -> list[Document]:
    """Retrieve documents from Pinecone based on parsed query.

    Routes between vector search, metadata-only filter, or hybrid
    depending on the query type.
    """
    pc = pc or get_pinecone_client()
    index_name = index_name or PINECONE_INDEX

    pc_filter, soft_filters = _build_pinecone_filter(parsed)
    logger.info("Pinecone filter: %s | soft filters: %s", pc_filter, soft_filters)

    if parsed.needs_all_results:
        if not pc_filter:
            pc_filter["chunk_type"] = {"$eq": "receipt"}
        docs = _metadata_only_query(pc_filter, index_name, pc)
        docs = _apply_soft_filters(docs, soft_filters)
        logger.info("Metadata-only query returned %d documents", len(docs))
        return docs

    extra_k = parsed.top_k * SOFT_FILTER_MULTIPLIER if soft_filters else parsed.top_k
    docs = _query_pinecone(
        query_text=parsed.search_text,
        pc_filter=pc_filter,
        top_k=extra_k,
        index_name=index_name,
        pc=pc,
    )
    docs = _apply_soft_filters(docs, soft_filters)
    if parsed.chunk_type == "item":
        docs = _rerank_items_by_category(docs)
    docs = docs[: parsed.top_k]
    logger.info("Vector query returned %d documents", len(docs))
    return docs
