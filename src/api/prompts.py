"""LLM response prompt and receipt/aggregation formatting helpers."""

from __future__ import annotations

from datetime import datetime

from src.models import AggregationResult

RESPONSE_PROMPT = """\
You are a helpful receipt analysis assistant. Answer the user's question using ONLY the data below.

IMPORTANT RULES:
- When aggregation data is present, ALWAYS use those computed numbers as your primary source of truth. \
They are pre-computed from ALL matching receipts, not just the sample shown below.
- The "Retrieved receipts" section is a SAMPLE (up to {max_display}). The aggregation covers ALL {count} results.
- For group-by breakdowns, summarize ALL groups in the aggregation, not just the sample receipts.
- When a spending breakdown shows subtotals+tax, tips, and grand total separately, \
report ALL three numbers so the user sees the full picture.
- Include specific numbers, dates, and merchant names.
- Never fabricate data.
- When no results are found, always mention today's date and the exact date range that was searched \
so the user understands what was looked for. Example: "As of March 15, 2026, I searched \
March 8–14, 2026 but found no matching receipts."

{search_context}

User question: {query}

Retrieved receipts (sample of {count} total results):
{context}

{aggregation_info}

Respond in 2-3 concise sentences. Be specific and data-driven."""


def fmt_date(iso: str) -> str:
    """Format 'YYYY-MM-DD' as 'Month Day, Year' (e.g. 'March 15, 2026')."""
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%B %-d, %Y")


def build_search_context(reference_date: str, date_range) -> str:
    """Build a search context string with current date and searched range."""
    parts = [f"Today's date: {fmt_date(reference_date)}"]
    if date_range:
        start = fmt_date(date_range.start) if date_range.start else "earliest"
        end = fmt_date(date_range.end) if date_range.end else "latest"
        parts.append(f"Date range searched: {start} to {end}")
    return "\n".join(parts)


def build_out_of_range_answer(
    today: str, start: str | None, end: str | None,
    dataset_start: str, dataset_end: str,
) -> str:
    """Human-readable message when the query dates fall outside the dataset."""
    searched_start = fmt_date(start) if start else "earliest"
    searched_end = fmt_date(end) if end else "latest"
    return (
        f"As of {fmt_date(today)}, I searched "
        f"{searched_start} to {searched_end} "
        f"but the receipt data only covers "
        f"{fmt_date(dataset_start)} to "
        f"{fmt_date(dataset_end)}, so there are no matching receipts."
    )


def format_receipt_for_context(doc_metadata: dict) -> str:
    """Format a receipt document's metadata for LLM context."""
    chunk_type = doc_metadata.get("chunk_type", "")
    display_date = doc_metadata.get("date_str") or doc_metadata.get("date", "?")
    if chunk_type == "item":
        return (
            f"- {doc_metadata.get('item_name', '?')} "
            f"${doc_metadata.get('item_price', 0):.2f} "
            f"at {doc_metadata.get('merchant', '?')} "
            f"on {display_date}"
        )
    parts = [
        f"- {doc_metadata.get('merchant', '?')}",
        f"({doc_metadata.get('category', '?')})",
        f"${doc_metadata.get('total_amount', 0):.2f}",
        f"on {display_date}",
        f"in {doc_metadata.get('city', '?')}",
    ]

    tip = doc_metadata.get("tip_amount", 0.0)
    tip_pct = doc_metadata.get("tip_percentage", 0.0)
    if tip and tip > 0:
        parts.append(f"tip: ${tip:.2f} ({tip_pct:.0f}%)")

    payment = doc_metadata.get("payment_method")
    if payment:
        parts.append(f"paid: {payment}")

    if doc_metadata.get("has_warranty"):
        parts.append("[WARRANTY]")
    if doc_metadata.get("has_prescription"):
        parts.append("[PRESCRIPTION]")
    if doc_metadata.get("has_loyalty_discount"):
        parts.append("[LOYALTY DISCOUNT]")

    parts.append(f"[{doc_metadata.get('receipt_id', '?')}]")
    return " ".join(parts)


def format_aggregation_info(aggregation_result: AggregationResult | None) -> str:
    """Convert an AggregationResult into a text block for the LLM prompt."""
    if not aggregation_result:
        return ""

    if aggregation_result.type.startswith("group_by"):
        breakdown = aggregation_result.result
        total = aggregation_result.total or 0
        lines = [f"AGGREGATION ({aggregation_result.message}, total: ${total:.2f}):"]
        if isinstance(breakdown, dict):
            for key, detail in breakdown.items():
                lines.append(f"  {key}: ${detail.total:.2f} ({detail.count} receipts)")
        return "\n".join(lines)

    if aggregation_result.type == "sum" and aggregation_result.tips_total:
        lines = [
            "AGGREGATION (spending breakdown):",
            f"  Subtotals + tax: ${aggregation_result.pre_tip_total:.2f}",
            f"  Tips: ${aggregation_result.tips_total:.2f}",
            f"  Grand total: ${aggregation_result.result:.2f}",
            f"  ({aggregation_result.count} receipts)",
        ]
        return "\n".join(lines)

    if aggregation_result.type in ("sum", "avg", "count", "max", "min"):
        return f"AGGREGATION: {aggregation_result.message}"

    return f"Aggregation: {aggregation_result.model_dump_json()}"
