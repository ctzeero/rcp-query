from __future__ import annotations

import re
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.config import MAX_QUERY_CHARS, MAX_QUERY_SENTENCES

class ReceiptItem(BaseModel):
    name: str
    price: float
    sku: Optional[str] = None
    is_prescription: bool = False
    rx_number: Optional[str] = None

class Receipt(BaseModel):
    receipt_id: str
    merchant: str
    address: str
    city: str
    state: str
    date: date
    time: Optional[str] = None
    category: str
    items: list[ReceiptItem]
    subtotal: float
    tax: float
    tax_rate: Optional[float] = None
    total: float
    tip: Optional[float] = None
    tip_percentage: Optional[float] = None
    grand_total: Optional[float] = None
    payment_method: str
    card_last_four: Optional[str] = None
    has_warranty: bool = False
    has_prescription: bool = False
    has_loyalty_discount: bool = False
    items_count: int

class QueryRequest(BaseModel):
    query: str
    conversation_history: list[dict] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        sentences = [s.strip() for s in re.split(r"[.!?]+", v) if s.strip()]
        if len(sentences) > MAX_QUERY_SENTENCES:
            raise ValueError(
                f"Please keep your query to {MAX_QUERY_SENTENCES} sentences or fewer."
            )
        if len(v) > MAX_QUERY_CHARS:
            raise ValueError(
                f"Query is too long. Please keep it under {MAX_QUERY_CHARS} characters."
            )
        return v

class GroupDetail(BaseModel):
    """One row in a group-by breakdown (category, merchant, or time period)."""

    total: float
    count: int
    avg: float = 0.0

class AggregationResult(BaseModel):
    """Typed result from the aggregation layer."""

    type: str
    result: float | int | dict[str, GroupDetail] | None = None
    message: str = ""
    count: int | None = None
    total: float | None = None
    pre_tip_total: float | None = None
    tips_total: float | None = None
    receipt_id: str | None = None
    merchant: str | None = None

class QueryResponse(BaseModel):
    answer: str
    receipts: list[dict] = Field(default_factory=list)
    aggregation: Optional[AggregationResult] = None
    query_metadata: dict = Field(default_factory=dict)
