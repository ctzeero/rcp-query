from __future__ import annotations

from datetime import date, datetime, timezone

from langchain_core.documents import Document

from src.models import Receipt


def _date_to_epoch(d: date) -> float:
    """Convert a date to Unix epoch seconds (midnight UTC)."""
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


def _receipt_metadata(receipt: Receipt) -> dict:
    """Build the shared metadata dict for a receipt-level chunk."""
    effective_total = receipt.grand_total if receipt.grand_total else receipt.total
    return {
        "chunk_type": "receipt",
        "receipt_id": receipt.receipt_id,
        "merchant": receipt.merchant,
        "date": _date_to_epoch(receipt.date),
        "date_str": receipt.date.isoformat(),
        "category": receipt.category,
        "total_amount": effective_total,
        "pre_tip_total": receipt.total,
        "tax": receipt.tax,
        "tip_amount": receipt.tip or 0.0,
        "tip_percentage": receipt.tip_percentage or 0.0,
        "payment_method": receipt.payment_method,
        "location": f"{receipt.city}, {receipt.state}",
        "city": receipt.city,
        "items_count": receipt.items_count,
        "has_warranty": receipt.has_warranty,
        "has_prescription": receipt.has_prescription,
        "has_loyalty_discount": receipt.has_loyalty_discount,
    }


def _receipt_text(receipt: Receipt) -> str:
    """Build the text content for a receipt-level chunk."""
    items_text = "\n".join(
        f"  - {item.name}: ${item.price:.2f}" for item in receipt.items
    )
    lines = [
        f"{receipt.merchant} - {receipt.category} receipt",
        f"Date: {receipt.date.isoformat()}",
        f"Location: {receipt.city}, {receipt.state}",
        f"Items ({receipt.items_count}):",
        items_text,
        f"Subtotal: ${receipt.subtotal:.2f}",
        f"Tax: ${receipt.tax:.2f}",
    ]
    if receipt.tip is not None:
        lines.append(f"Tip: ${receipt.tip:.2f}")
    if receipt.grand_total is not None:
        lines.append(f"Grand Total: ${receipt.grand_total:.2f}")
    else:
        lines.append(f"Total: ${receipt.total:.2f}")
    lines.append(f"Payment: {receipt.payment_method}")
    if receipt.has_warranty:
        lines.append("Includes extended warranty")
    if receipt.has_prescription:
        lines.append("Includes pharmacy prescription pickup")
    return "\n".join(lines)


def _item_document(receipt: Receipt, idx: int) -> Document:
    """Build a Document for a single item-level chunk."""
    item = receipt.items[idx]
    text = (
        f"{item.name} - ${item.price:.2f} at {receipt.merchant} "
        f"on {receipt.date.isoformat()} in {receipt.city} ({receipt.category})"
    )
    metadata = {
        "chunk_type": "item",
        "receipt_id": receipt.receipt_id,
        "parent_receipt_id": receipt.receipt_id,
        "merchant": receipt.merchant,
        "date": _date_to_epoch(receipt.date),
        "date_str": receipt.date.isoformat(),
        "category": receipt.category,
        "city": receipt.city,
        "item_name": item.name,
        "item_price": item.price,
    }
    if item.sku:
        metadata["sku"] = item.sku
    if item.is_prescription:
        metadata["is_prescription"] = True
    if item.rx_number:
        metadata["rx_number"] = item.rx_number

    return Document(page_content=text, metadata=metadata)


def chunk_receipt(receipt: Receipt) -> list[Document]:
    """Convert a Receipt into a list of LangChain Documents (hybrid chunks).

    Returns one receipt-level Document + one Document per item.
    """
    docs: list[Document] = []

    receipt_doc = Document(
        page_content=_receipt_text(receipt),
        metadata=_receipt_metadata(receipt),
    )
    docs.append(receipt_doc)

    for i in range(len(receipt.items)):
        docs.append(_item_document(receipt, i))

    return docs


def chunk_receipts(receipts: list[Receipt]) -> list[Document]:
    """Chunk a list of receipts into Documents."""
    all_docs: list[Document] = []
    for receipt in receipts:
        all_docs.extend(chunk_receipt(receipt))
    return all_docs
