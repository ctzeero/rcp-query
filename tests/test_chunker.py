"""
Unit tests for src.ingestion.chunker -- pure Python, no external deps.

Run with: python -m pytest tests/test_chunker.py
"""

from __future__ import annotations

from datetime import date

import pytest

from src.ingestion.chunker import chunk_receipt, chunk_receipts
from src.models import Receipt, ReceiptItem

def _make_receipt(**overrides) -> Receipt:
    """Build a minimal Receipt with sensible defaults."""
    defaults = dict(
        receipt_id="receipt_001",
        merchant="Test Store",
        address="123 Main St",
        city="San Francisco",
        state="CA",
        date=date(2023, 12, 15),
        category="grocery",
        items=[
            ReceiptItem(name="Chicken Breast", price=12.07),
            ReceiptItem(name="Pasta", price=4.21),
        ],
        subtotal=16.28,
        tax=1.43,
        total=17.71,
        payment_method="VISA",
        items_count=2,
    )
    defaults.update(overrides)
    return Receipt(**defaults)

class TestChunkReceipt:
    def test_produces_receipt_plus_item_chunks(self):
        receipt = _make_receipt()
        docs = chunk_receipt(receipt)
        assert len(docs) == 3  # 1 receipt + 2 items

    def test_first_chunk_is_receipt_level(self):
        docs = chunk_receipt(_make_receipt())
        assert docs[0].metadata["chunk_type"] == "receipt"
        assert docs[0].metadata["receipt_id"] == "receipt_001"

    def test_item_chunks_have_correct_metadata(self):
        docs = chunk_receipt(_make_receipt())
        item_docs = [d for d in docs if d.metadata["chunk_type"] == "item"]
        assert len(item_docs) == 2
        names = {d.metadata["item_name"] for d in item_docs}
        assert names == {"Chicken Breast", "Pasta"}

    def test_receipt_metadata_contains_expected_fields(self):
        docs = chunk_receipt(_make_receipt())
        meta = docs[0].metadata
        assert meta["merchant"] == "Test Store"
        assert meta["category"] == "grocery"
        assert meta["city"] == "San Francisco"
        assert meta["total_amount"] == 17.71
        assert meta["payment_method"] == "VISA"
        assert "date" in meta
        assert "date_str" in meta

    def test_grand_total_used_when_present(self):
        receipt = _make_receipt(
            tip=5.00,
            grand_total=22.71,
        )
        docs = chunk_receipt(receipt)
        assert docs[0].metadata["total_amount"] == 22.71

    def test_item_price_in_metadata(self):
        docs = chunk_receipt(_make_receipt())
        chicken = next(d for d in docs if d.metadata.get("item_name") == "Chicken Breast")
        assert chicken.metadata["item_price"] == 12.07

    def test_prescription_metadata_propagated(self):
        receipt = _make_receipt(
            items=[
                ReceiptItem(name="Amoxicillin", price=15.00, is_prescription=True, rx_number="RX123"),
            ],
            items_count=1,
            has_prescription=True,
        )
        docs = chunk_receipt(receipt)
        rx_doc = next(d for d in docs if d.metadata.get("item_name") == "Amoxicillin")
        assert rx_doc.metadata["is_prescription"] is True
        assert rx_doc.metadata["rx_number"] == "RX123"

    def test_warranty_flag_on_receipt(self):
        receipt = _make_receipt(has_warranty=True)
        docs = chunk_receipt(receipt)
        assert docs[0].metadata["has_warranty"] is True

    def test_receipt_text_contains_merchant_and_items(self):
        docs = chunk_receipt(_make_receipt())
        text = docs[0].page_content
        assert "Test Store" in text
        assert "Chicken Breast" in text
        assert "Pasta" in text

    def test_date_epoch_is_numeric(self):
        docs = chunk_receipt(_make_receipt())
        assert isinstance(docs[0].metadata["date"], (int, float))

    def test_zero_items_produces_receipt_only(self):
        receipt = _make_receipt(items=[], items_count=0)
        docs = chunk_receipt(receipt)
        assert len(docs) == 1
        assert docs[0].metadata["chunk_type"] == "receipt"

class TestChunkReceipts:
    def test_processes_multiple_receipts(self):
        r1 = _make_receipt(receipt_id="r1", items_count=2)
        r2 = _make_receipt(
            receipt_id="r2",
            items=[ReceiptItem(name="Coffee", price=5.00)],
            items_count=1,
        )
        docs = chunk_receipts([r1, r2])
        assert len(docs) == 5  # (1+2) + (1+1)

    def test_empty_list_returns_empty(self):
        assert chunk_receipts([]) == []
