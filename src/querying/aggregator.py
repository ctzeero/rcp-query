from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from langchain_core.documents import Document

from src.models import AggregationResult, GroupDetail

logger = logging.getLogger(__name__)


def _get_amount(doc: Document) -> float:
    """Extract the monetary amount from a document's metadata."""
    if doc.metadata.get("chunk_type") == "item":
        return doc.metadata.get("item_price", 0.0)
    return doc.metadata.get("total_amount", 0.0)

def _deduplicate_receipts(docs: list[Document]) -> list[Document]:
    """Keep only one document per receipt_id (for receipt-level aggregation)."""
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        rid = doc.metadata.get("receipt_id", "")
        if rid and rid not in seen:
            seen.add(rid)
            unique.append(doc)
    return unique

def _parse_date(raw) -> datetime | None:
    """Parse a date value that may be an epoch float, int, or ISO string."""
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(raw, str):
        try:
            return datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            return None
    return None

def _week_key(raw) -> str:
    """Convert a date value to a week key (Monday start)."""
    dt = _parse_date(raw)
    if dt is None:
        return "unknown"
    monday = dt - timedelta(days=dt.weekday())
    return f"Week of {monday.strftime('%Y-%m-%d')}"

def _month_key(raw) -> str:
    """Convert a date value to a month key."""
    dt = _parse_date(raw)
    if dt is None:
        return "unknown"
    return dt.strftime("%Y-%m")

def aggregate(docs: list[Document], aggregation_type: str) -> AggregationResult:
    """Compute aggregation over retrieved documents.

    All monetary computations happen in Python -- never delegated to the LLM.
    """
    if not docs:
        return AggregationResult(
            type=aggregation_type, result=None, message="No matching receipts found."
        )

    receipt_docs = [d for d in docs if d.metadata.get("chunk_type") == "receipt"]
    if receipt_docs:
        receipt_docs = _deduplicate_receipts(receipt_docs)

    working_docs = receipt_docs if receipt_docs else docs
    amounts = [_get_amount(d) for d in working_docs]

    if aggregation_type == "sum":
        grand_total = round(sum(amounts), 2)
        tips = [d.metadata.get("tip_amount", 0.0) or 0.0 for d in working_docs]
        tips_total = round(sum(tips), 2)
        pre_tip = round(grand_total - tips_total, 2)

        if tips_total > 0:
            msg = (
                f"Subtotals + tax: ${pre_tip:.2f}, "
                f"Tips: ${tips_total:.2f}, "
                f"Grand total: ${grand_total:.2f} "
                f"across {len(amounts)} receipts"
            )
        else:
            msg = f"Total: ${grand_total:.2f} across {len(amounts)} receipts"

        return AggregationResult(
            type="sum",
            result=grand_total,
            count=len(amounts),
            pre_tip_total=pre_tip if tips_total > 0 else None,
            tips_total=tips_total if tips_total > 0 else None,
            message=msg,
        )

    if aggregation_type == "avg":
        avg = sum(amounts) / len(amounts)
        return AggregationResult(
            type="avg",
            result=round(avg, 2),
            count=len(amounts),
            message=f"Average: ${avg:.2f} across {len(amounts)} receipts",
        )

    if aggregation_type == "count":
        return AggregationResult(
            type="count",
            result=len(working_docs),
            message=f"Found {len(working_docs)} matching receipts",
        )

    if aggregation_type == "max":
        max_val = max(amounts)
        max_doc = working_docs[amounts.index(max_val)]
        return AggregationResult(
            type="max",
            result=round(max_val, 2),
            receipt_id=max_doc.metadata.get("receipt_id", ""),
            merchant=max_doc.metadata.get("merchant", ""),
            message=f"Most expensive: ${max_val:.2f} at {max_doc.metadata.get('merchant', 'unknown')}",
        )

    if aggregation_type == "min":
        min_val = min(amounts)
        min_doc = working_docs[amounts.index(min_val)]
        return AggregationResult(
            type="min",
            result=round(min_val, 2),
            receipt_id=min_doc.metadata.get("receipt_id", ""),
            merchant=min_doc.metadata.get("merchant", ""),
            message=f"Least expensive: ${min_val:.2f} at {min_doc.metadata.get('merchant', 'unknown')}",
        )

    if aggregation_type == "group_by_category":
        groups: dict[str, list[float]] = defaultdict(list)
        for doc in working_docs:
            groups[doc.metadata.get("category", "unknown")].append(_get_amount(doc))
        breakdown = {
            cat: GroupDetail(
                total=round(sum(vals), 2),
                count=len(vals),
                avg=round(sum(vals) / len(vals), 2),
            )
            for cat, vals in sorted(groups.items(), key=lambda x: sum(x[1]), reverse=True)
        }
        return AggregationResult(
            type="group_by_category",
            result=breakdown,
            total=round(sum(amounts), 2),
            message="Spending breakdown by category",
        )

    if aggregation_type == "group_by_merchant":
        groups = defaultdict(list)
        for doc in working_docs:
            groups[doc.metadata.get("merchant", "unknown")].append(_get_amount(doc))
        breakdown = {
            m: GroupDetail(
                total=round(sum(vals), 2),
                count=len(vals),
                avg=round(sum(vals) / len(vals), 2),
            )
            for m, vals in sorted(groups.items(), key=lambda x: sum(x[1]), reverse=True)
        }
        return AggregationResult(
            type="group_by_merchant",
            result=breakdown,
            total=round(sum(amounts), 2),
            message="Spending breakdown by merchant",
        )

    if aggregation_type in ("group_by_week", "group_by_month"):
        key_fn = _week_key if aggregation_type == "group_by_week" else _month_key
        groups = defaultdict(list)
        for doc in working_docs:
            date_val = doc.metadata.get("date") or doc.metadata.get("date_str", "")
            if date_val:
                groups[key_fn(date_val)].append(_get_amount(doc))
        breakdown = {
            period: GroupDetail(total=round(sum(vals), 2), count=len(vals))
            for period, vals in sorted(groups.items())
        }
        return AggregationResult(
            type=aggregation_type,
            result=breakdown,
            total=round(sum(amounts), 2),
            message=f"Spending breakdown by {'week' if 'week' in aggregation_type else 'month'}",
        )

    logger.warning("Unknown aggregation type: %s", aggregation_type)
    return AggregationResult(
        type=aggregation_type,
        result=None,
        message=f"Unsupported aggregation type: {aggregation_type}",
    )
