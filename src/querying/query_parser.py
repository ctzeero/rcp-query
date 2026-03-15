from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

from datetime import datetime

from src.config import (
    DATASET_END_DATE,
    DATASET_START_DATE,
    DEFAULT_TOP_K,
    GEMINI_MODEL,
    LLM_TEMPERATURE_PARSING,
    REFERENCE_DATE,
)
from src.querying.date_resolver import resolve_dates

logger = logging.getLogger(__name__)

CATEGORIES = [
    "grocery", "restaurant", "coffee", "fast_food",
    "electronics", "pharmacy", "retail", "hardware", "gas",
]

_TZ_JUNK_RE = re.compile(r"(T\d{2}:\d{2}:\d{2})?(Z|\[UTC\])+$")


class DateRange(BaseModel):
    start: Optional[str] = Field(None, description="ISO date YYYY-MM-DD only, no timezone")
    end: Optional[str] = Field(None, description="ISO date YYYY-MM-DD only, no timezone")

    @field_validator("start", "end", mode="before")
    @classmethod
    def strip_tz_junk(cls, v: str | None) -> str | None:
        if v is None:
            return v
        cleaned = _TZ_JUNK_RE.sub("", v.strip())
        if len(cleaned) > 10:
            cleaned = cleaned[:10]
        return cleaned


class ParsedQuery(BaseModel):
    """Structured representation of a user's natural language query."""

    search_text: str = Field(
        description="The core semantic search text to use for vector similarity. "
        "Rephrase for embedding search -- keep item names, descriptions, merchant names."
    )
    chunk_type: Optional[str] = Field(
        None,
        description="'receipt' for merchant/total/date queries, 'item' for specific item searches, null for both",
    )
    filters: dict = Field(
        default_factory=dict,
        description="Metadata filters: merchant, category, city, payment_method, "
        "has_warranty, has_prescription, has_loyalty_discount. "
        "Use exact values. category must be one of: "
        "grocery, restaurant, coffee, fast_food, electronics, pharmacy, retail, hardware, gas",
    )
    date_range: Optional[DateRange] = Field(
        None,
        description="Date range filter with start and/or end in ISO format",
    )
    aggregation: Optional[str] = Field(
        None,
        description="Aggregation type if needed: sum, avg, count, min, max, "
        "group_by_category, group_by_merchant, group_by_month, group_by_week",
    )
    price_filter: Optional[dict] = Field(
        None,
        description="Price filter: {'operator': 'gt'|'lt'|'gte'|'lte'|'eq', 'value': float, 'field': 'total_amount'|'item_price'}",
    )
    tip_filter: Optional[dict] = Field(
        None,
        description="Tip percentage filter: {'operator': 'gt'|'lt'|'gte'|'lte'|'eq', 'value': float}. "
        "Use for queries like 'tip over 20%' -> {'operator': 'gt', 'value': 20}",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        description="Number of results to retrieve. Use higher values (50-100) for aggregation queries.",
    )
    needs_all_results: bool = Field(
        default=False,
        description="True if the query needs ALL matching results (aggregation, totals, counting). "
        "When true, retriever should use metadata-only filter without top-k limit.",
    )


_PARSE_PROMPT = """\
You are a query parser for a receipt query system. The dataset contains 100 receipts \
from Nov 2023 to Jan 2024 in the San Francisco Bay Area.

Today's date is {today}. The dataset covers {dataset_start} to {dataset_end}.
Use today's date to resolve relative expressions ("last week", "recently", "this month").
For bare month names without a year ("December", "November"), assume the year that \
falls within the dataset range.

CRITICAL: Date values in date_range MUST be plain YYYY-MM-DD format. \
NO timezone suffixes, NO "T" times, NO "Z", NO "[UTC]". Examples: "2023-12-01", "2024-01-31".

Date resolution examples (ALWAYS set BOTH start AND end for bounded periods):
- "December" or "December 2023" -> start="2023-12-01", end="2023-12-31"
- "November" -> start="2023-11-01", end="2023-11-30"
- "January" or "January 2024" -> start="2024-01-01", end="2024-01-31"
- "Q4 2023" -> start="2023-10-01", end="2023-12-31"
- "Thanksgiving week" -> start="2023-11-20", end="2023-11-26"
- "Christmas week" -> start="2023-12-18", end="2023-12-25"
- "week before Christmas" -> start="2023-12-18", end="2023-12-24"
- "first week of January" -> start="2024-01-01", end="2024-01-07"
- "last week of November" -> start="2023-11-27", end="2023-11-30"
- "before Christmas" -> end="2023-12-24" (open start is OK)
- "recently" -> last 2 weeks from today's date
- "11/07/2023" or "November 7, 2023" -> start="2023-11-07", end="2023-11-07"

CRITICAL: When a user mentions a month name (e.g. "December", "in January"), \
you MUST set BOTH start and end to the first and last day of that month. \
Never leave end=null for a month query -- that returns everything after the start date.

ALWAYS set date_range when the query mentions ANY time period, month, week, date, \
holiday, or season. Never leave date_range null if the user is asking about a time period.

Valid categories: grocery, restaurant, coffee, fast_food, electronics, pharmacy, retail, hardware, gas

Parse this user query into structured filters. Be precise with filters -- only add filters you're confident about.

For tip-related queries ("tip over 20%", "tipped 15%", "tip at least 18%"), set tip_filter \
with the appropriate operator and value. Example: "tip over 20%" -> tip_filter={{"operator": "gt", "value": 20}}. \
Also set needs_all_results=true and category="restaurant" since only restaurants have tips.

For aggregation queries (total spending, average, how many, most/least expensive, breakdown by), \
set needs_all_results=true and appropriate aggregation type.

For item-specific queries ("find chicken", "show me coffee drinks"), set chunk_type="item".
For receipt-level queries ("show me all Whole Foods receipts", "how much at restaurants"), set chunk_type="receipt".
For ambiguous queries, leave chunk_type=null to search both.

User query: {query}"""


class QueryParser:
    """Parses natural language queries into structured query parameters."""

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None):
        if llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                temperature=LLM_TEMPERATURE_PARSING,
            )
        else:
            self._llm = llm

        self._structured_llm = self._llm.with_structured_output(ParsedQuery)

    def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query into structured parameters."""
        fast = self._try_fast_parse(query)
        if fast is not None:
            logger.info("Fast-path parse (skipped LLM): %s", fast.model_dump_json(indent=2))
            return fast

        prompt = _PARSE_PROMPT.format(
            today=REFERENCE_DATE,
            dataset_start=DATASET_START_DATE,
            dataset_end=DATASET_END_DATE,
            query=query,
        )
        result: ParsedQuery = self._structured_llm.invoke(prompt)
        result = self._post_parse_fixes(query, result)
        logger.info("Parsed query: %s", result.model_dump_json(indent=2))
        return result

    @staticmethod
    def _try_fast_parse(query: str) -> ParsedQuery | None:
        """Attempt to parse the query without an LLM call.

        Returns a ``ParsedQuery`` when the date resolver handles the temporal
        expression and no complex filters (merchant, category, aggregation,
        tips, price, item names) are detected.  Returns ``None`` to fall
        through to the LLM.
        """
        resolved = resolve_dates(query, REFERENCE_DATE, dataset_end=DATASET_END_DATE)
        if resolved is None:
            return None

        q = query.lower()
        complex_signals = [
            "total", "spending", "spent", "average", "how many", "count",
            "most expensive", "least expensive", "cheapest", "breakdown",
            "group", "tip", "tipped", "warranty", "prescription", "loyalty",
            "category", "merchant", "restaurant", "grocery", "coffee",
            "pharmacy", "electronics", "retail", "hardware", "gas",
            "fast food", "fast_food",
        ]
        if any(signal in q for signal in complex_signals):
            return None

        start, end = resolved
        return ParsedQuery(
            search_text=query,
            date_range=DateRange(start=start, end=end),
        )

    @staticmethod
    def _post_parse_fixes(query: str, parsed: ParsedQuery) -> ParsedQuery:
        """Apply deterministic fallbacks when the LLM misses obvious signals."""
        resolved = resolve_dates(query, REFERENCE_DATE, dataset_end=DATASET_END_DATE)
        if resolved:
            start, end = resolved
            if parsed.date_range is None or not parsed.date_range.end:
                parsed.date_range = DateRange(start=start, end=end)
                logger.info("Date resolver: %s -> %s", start, end)

        if parsed.tip_filter is None:
            m = re.search(
                r"tip(?:ped)?\s+(?:over|above|more than|greater than|>)\s*(\d+)%?",
                query, re.IGNORECASE,
            )
            if m:
                parsed.tip_filter = {"operator": "gt", "value": float(m.group(1))}
                parsed.needs_all_results = True
                if "category" not in parsed.filters:
                    parsed.filters["category"] = "restaurant"
                parsed.chunk_type = "receipt"
                logger.info("Post-parse: extracted tip filter > %s%%", m.group(1))
            else:
                m = re.search(
                    r"tip(?:ped)?\s+(?:exactly\s+)?(\d+)%",
                    query, re.IGNORECASE,
                )
                if m:
                    parsed.tip_filter = {"operator": "eq", "value": float(m.group(1))}
                    parsed.needs_all_results = True
                    if "category" not in parsed.filters:
                        parsed.filters["category"] = "restaurant"
                    parsed.chunk_type = "receipt"
                    logger.info("Post-parse: extracted tip filter == %s%%", m.group(1))

        return parsed
