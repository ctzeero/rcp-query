"""
Unit tests for src.querying.aggregator -- pure Python, no external deps.

Run with: python -m pytest tests/test_aggregator.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document

from src.models import AggregationResult, GroupDetail
from src.querying.aggregator import aggregate


def _receipt_doc(
    receipt_id: str,
    merchant: str,
    total: float,
    category: str = "grocery",
    date_epoch: float | None = None,
) -> Document:
    """Helper to build a receipt-level Document with typical metadata."""
    epoch = date_epoch or datetime(2023, 12, 15, tzinfo=timezone.utc).timestamp()
    return Document(
        page_content=f"{merchant} receipt",
        metadata={
            "chunk_type": "receipt",
            "receipt_id": receipt_id,
            "merchant": merchant,
            "total_amount": total,
            "category": category,
            "date": epoch,
            "date_str": "2023-12-15",
        },
    )

def _item_doc(receipt_id: str, item_name: str, price: float) -> Document:
    return Document(
        page_content=f"{item_name} at store",
        metadata={
            "chunk_type": "item",
            "receipt_id": receipt_id,
            "item_name": item_name,
            "item_price": price,
        },
    )

class TestAggregateReturnType:
    def test_returns_aggregation_result_model(self):
        result = aggregate([], "sum")
        assert isinstance(result, AggregationResult)

    def test_group_by_returns_group_detail_values(self):
        docs = [
            _receipt_doc("r1", "WF", 50.00, category="grocery"),
            _receipt_doc("r2", "SB", 8.00, category="coffee"),
        ]
        result = aggregate(docs, "group_by_category")
        assert isinstance(result.result, dict)
        for detail in result.result.values():
            assert isinstance(detail, GroupDetail)

class TestAggregateEmpty:
    def test_returns_none_result_on_empty(self):
        result = aggregate([], "sum")
        assert result.type == "sum"
        assert result.result is None
        assert "No matching" in result.message

class TestAggregateSum:
    def test_sums_receipt_totals(self):
        docs = [
            _receipt_doc("r1", "Store A", 10.50),
            _receipt_doc("r2", "Store B", 20.00),
            _receipt_doc("r3", "Store C", 5.25),
        ]
        result = aggregate(docs, "sum")
        assert result.type == "sum"
        assert result.result == 35.75
        assert result.count == 3

    def test_deduplicates_receipts(self):
        docs = [
            _receipt_doc("r1", "Store A", 10.00),
            _receipt_doc("r1", "Store A", 10.00),
            _receipt_doc("r2", "Store B", 20.00),
        ]
        result = aggregate(docs, "sum")
        assert result.result == 30.00
        assert result.count == 2

class TestAggregateAvg:
    def test_computes_average(self):
        docs = [
            _receipt_doc("r1", "A", 10.00),
            _receipt_doc("r2", "B", 30.00),
        ]
        result = aggregate(docs, "avg")
        assert result.result == 20.00
        assert result.count == 2

class TestAggregateCount:
    def test_counts_documents(self):
        docs = [_receipt_doc(f"r{i}", "Store", 10.0) for i in range(7)]
        result = aggregate(docs, "count")
        assert result.result == 7

class TestAggregateMinMax:
    def test_max_finds_highest(self):
        docs = [
            _receipt_doc("r1", "Cheap", 5.00),
            _receipt_doc("r2", "Expensive", 99.99),
            _receipt_doc("r3", "Medium", 50.00),
        ]
        result = aggregate(docs, "max")
        assert result.result == 99.99
        assert result.merchant == "Expensive"

    def test_min_finds_lowest(self):
        docs = [
            _receipt_doc("r1", "Cheap", 5.00),
            _receipt_doc("r2", "Expensive", 99.99),
        ]
        result = aggregate(docs, "min")
        assert result.result == 5.00
        assert result.merchant == "Cheap"

class TestAggregateGroupBy:
    def test_group_by_category(self):
        docs = [
            _receipt_doc("r1", "WF", 50.00, category="grocery"),
            _receipt_doc("r2", "WF", 30.00, category="grocery"),
            _receipt_doc("r3", "Starbucks", 8.00, category="coffee"),
        ]
        result = aggregate(docs, "group_by_category")
        assert result.type == "group_by_category"
        assert result.total == 88.00
        assert result.result["grocery"].total == 80.00
        assert result.result["grocery"].count == 2
        assert result.result["coffee"].total == 8.00

    def test_group_by_merchant(self):
        docs = [
            _receipt_doc("r1", "Whole Foods", 50.00),
            _receipt_doc("r2", "Whole Foods", 30.00),
            _receipt_doc("r3", "Target", 20.00),
        ]
        result = aggregate(docs, "group_by_merchant")
        assert result.result["Whole Foods"].total == 80.00
        assert result.result["Target"].count == 1

    def test_group_by_month(self):
        nov = datetime(2023, 11, 15, tzinfo=timezone.utc).timestamp()
        dec = datetime(2023, 12, 10, tzinfo=timezone.utc).timestamp()
        docs = [
            _receipt_doc("r1", "A", 10.00, date_epoch=nov),
            _receipt_doc("r2", "B", 20.00, date_epoch=dec),
            _receipt_doc("r3", "C", 30.00, date_epoch=dec),
        ]
        result = aggregate(docs, "group_by_month")
        assert "2023-11" in result.result
        assert "2023-12" in result.result
        assert result.result["2023-12"].total == 50.00

class TestAggregateItems:
    def test_falls_back_to_item_prices_when_no_receipts(self):
        docs = [
            _item_doc("r1", "Chicken", 12.00),
            _item_doc("r1", "Pasta", 4.00),
        ]
        result = aggregate(docs, "sum")
        assert result.result == 16.00

class TestAggregateUnknownType:
    def test_unknown_type_returns_unsupported(self):
        docs = [_receipt_doc("r1", "A", 10.00)]
        result = aggregate(docs, "percentile_90")
        assert result.result is None
        assert "Unsupported" in result.message
