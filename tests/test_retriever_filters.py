"""
Unit tests for filter building and soft filtering in src.querying.retriever.

These tests exercise the pure-logic functions without touching Pinecone.

Run with: python -m pytest tests/test_receipt_samples.py -v -s
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.querying.query_parser import DateRange, ParsedQuery
from src.querying.retriever import _apply_soft_filters, _build_pinecone_filter


def _parsed(
    search_text: str = "test",
    chunk_type: str | None = None,
    filters: dict | None = None,
    date_range: DateRange | None = None,
    price_filter: dict | None = None,
) -> ParsedQuery:
    return ParsedQuery(
        search_text=search_text,
        chunk_type=chunk_type,
        filters=filters or {},
        date_range=date_range,
        price_filter=price_filter,
    )


class TestBuildPineconeFilter:
    def test_empty_query_produces_empty_filter(self):
        pc_filter, soft = _build_pinecone_filter(_parsed())
        assert pc_filter == {}
        assert soft == {}

    def test_chunk_type_becomes_eq_filter(self):
        pc_filter, _ = _build_pinecone_filter(_parsed(chunk_type="receipt"))
        assert pc_filter["chunk_type"] == {"$eq": "receipt"}

    def test_boolean_filter_goes_to_pinecone(self):
        pc_filter, soft = _build_pinecone_filter(
            _parsed(filters={"has_warranty": True})
        )
        assert pc_filter["has_warranty"] == {"$eq": True}
        assert soft == {}

    def test_string_filter_becomes_soft_filter(self):
        pc_filter, soft = _build_pinecone_filter(
            _parsed(filters={"merchant": "Whole Foods"})
        )
        assert "merchant" not in pc_filter
        assert soft["merchant"] == "Whole Foods"

    def test_category_is_soft_filter(self):
        _, soft = _build_pinecone_filter(_parsed(filters={"category": "grocery"}))
        assert soft["category"] == "grocery"

    def test_date_range_both(self):
        dr = DateRange(start="2023-12-01", end="2023-12-31")
        pc_filter, _ = _build_pinecone_filter(_parsed(date_range=dr))
        assert "$gte" in pc_filter["date"]
        assert "$lte" in pc_filter["date"]

    def test_date_range_start_only(self):
        dr = DateRange(start="2023-12-15", end=None)
        pc_filter, _ = _build_pinecone_filter(_parsed(date_range=dr))
        assert "$gte" in pc_filter["date"]
        assert "$lte" not in pc_filter["date"]

    def test_date_range_end_only(self):
        dr = DateRange(start=None, end="2023-12-31")
        pc_filter, _ = _build_pinecone_filter(_parsed(date_range=dr))
        assert "$lte" in pc_filter["date"]
        assert "$gte" not in pc_filter["date"]

    def test_price_filter(self):
        pf = {"operator": "gt", "value": 100, "field": "total_amount"}
        pc_filter, _ = _build_pinecone_filter(_parsed(price_filter=pf))
        assert pc_filter["total_amount"] == {"$gt": 100}

    def test_combined_filters(self):
        dr = DateRange(start="2023-12-01", end="2023-12-31")
        pc_filter, soft = _build_pinecone_filter(
            _parsed(
                chunk_type="receipt",
                filters={"category": "grocery", "has_warranty": False},
                date_range=dr,
            )
        )
        assert pc_filter["chunk_type"] == {"$eq": "receipt"}
        assert pc_filter["has_warranty"] == {"$eq": False}
        assert "date" in pc_filter
        assert soft["category"] == "grocery"


class TestApplySoftFilters:
    def _doc(self, merchant: str, category: str = "grocery") -> Document:
        return Document(
            page_content="test",
            metadata={"merchant": merchant, "category": category},
        )

    def test_no_filters_returns_all(self):
        docs = [self._doc("A"), self._doc("B")]
        assert _apply_soft_filters(docs, {}) == docs

    def test_filters_by_merchant_substring(self):
        docs = [
            self._doc("Whole Foods Market"),
            self._doc("Target"),
            self._doc("Whole Foods Express"),
        ]
        result = _apply_soft_filters(docs, {"merchant": "Whole Foods"})
        assert len(result) == 2

    def test_case_insensitive_matching(self):
        docs = [self._doc("WHOLE FOODS"), self._doc("Target")]
        result = _apply_soft_filters(docs, {"merchant": "whole foods"})
        assert len(result) == 1

    def test_multiple_soft_filters_and_together(self):
        docs = [
            self._doc("Whole Foods", "grocery"),
            self._doc("Whole Foods", "pharmacy"),
            self._doc("Target", "grocery"),
        ]
        result = _apply_soft_filters(docs, {"merchant": "Whole Foods", "category": "grocery"})
        assert len(result) == 1
        assert result[0].metadata["merchant"] == "Whole Foods"

    def test_no_match_returns_empty(self):
        docs = [self._doc("Target")]
        result = _apply_soft_filters(docs, {"merchant": "Costco"})
        assert result == []


class TestQueryValidation:
    """Tests for QueryRequest validation in models.py (exercises the guard we added)."""

    def test_empty_query_rejected(self):
        from src.models import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(query="")

    def test_whitespace_query_rejected(self):
        from src.models import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(query="   ")

    def test_valid_query_accepted(self):
        from src.models import QueryRequest
        req = QueryRequest(query="How much did I spend at Whole Foods?")
        assert req.query.strip() == "How much did I spend at Whole Foods?"

    def test_too_many_sentences_rejected(self):
        from src.models import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(query="One. Two. Three. Four.")

    def test_too_long_query_rejected(self):
        from src.models import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(query="a" * 501)
