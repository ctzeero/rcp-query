from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.config import GEMINI_MODEL, LLM_TEMPERATURE_PARSING, RECEIPT_FILE_GLOB
from src.models import Receipt, ReceiptItem

logger = logging.getLogger(__name__)


class _LLMReceiptExtraction(BaseModel):
    """Schema for Gemini structured output extraction."""

    merchant: str
    address: str
    city: str
    state: str
    date: str  # MM/DD/YYYY
    time: Optional[str] = None
    items: list[_LLMItemExtraction]
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


class _LLMItemExtraction(BaseModel):
    name: str
    price: float
    sku: Optional[str] = None
    is_prescription: bool = False
    rx_number: Optional[str] = None


_LLMReceiptExtraction.model_rebuild()

_EXTRACTION_PROMPT = """\
Extract all structured data from this receipt text. Follow these rules:
- date: extract exactly as shown (MM/DD/YYYY format)
- time: extract if present, otherwise null
- tax_rate: extract the percentage number if shown (e.g. 8.8 for "8.8%"), otherwise null
- tip_percentage: extract from "TIP (XX%)" line if present (e.g. 20 for "TIP (20%)"), otherwise null
- tip: the dollar amount of the tip if present
- grand_total: the GRAND TOTAL line if present, otherwise null
- total: the TOTAL line (before tip)
- payment_method: VISA, MASTERCARD, DEBIT CARD, AMEX, etc.
- card_last_four: the last 4 digits after **** if present
- has_warranty: true if EXTENDED WARRANTY appears
- has_prescription: true if PHARMACY PICKUP or RX# appears
- has_loyalty_discount: true if RedCard, MEMBER SAVINGS, LOYALTY, or similar discount appears
- items: list all purchased items with name, price, sku (if present), is_prescription, rx_number
- For prescription items under "PHARMACY PICKUP", set is_prescription=true and extract RX# as rx_number

Receipt text:
{receipt_text}"""

def _extract_from_filename(filename: str) -> tuple[str, str]:
    """Extract receipt_id and category from filename.

    Example: receipt_001_grocery_20231107.txt -> ("receipt_001", "grocery")
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    receipt_id = f"{parts[0]}_{parts[1]}"
    category = "_".join(parts[2:-1])
    return receipt_id, category

def _parse_date(date_str: str) -> datetime:
    """Parse date string in MM/DD/YYYY format."""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_str}")

def _fallback_card_last_four(text: str) -> Optional[str]:
    """Regex fallback to extract card last four digits."""
    match = re.search(r"\*{3,4}(\d{4})", text)
    return match.group(1) if match else None

def _fallback_tax_rate(text: str) -> Optional[float]:
    """Regex fallback to extract tax rate percentage."""
    match = re.search(r"TAX\s*\((\d+\.?\d*)%\)", text)
    return float(match.group(1)) if match else None

def _fallback_tip_percentage(text: str) -> Optional[float]:
    """Regex fallback to extract tip percentage."""
    match = re.search(r"TIP\s*\((\d+\.?\d*)%\)", text)
    return float(match.group(1)) if match else None

class ReceiptParser:
    """Parses raw receipt .txt files into Receipt objects using Gemini."""

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None):
        if llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                temperature=LLM_TEMPERATURE_PARSING,
            )
        else:
            self._llm = llm

        self._structured_llm = self._llm.with_structured_output(
            _LLMReceiptExtraction
        )

    def parse_file(self, filepath: Path) -> Receipt:
        """Parse a single receipt file into a Receipt object."""
        text = filepath.read_text(encoding="utf-8")
        receipt_id, category = _extract_from_filename(filepath.name)

        extraction: _LLMReceiptExtraction = self._structured_llm.invoke(
            _EXTRACTION_PROMPT.format(receipt_text=text)
        )

        parsed_date = _parse_date(extraction.date)

        card_last_four = extraction.card_last_four or _fallback_card_last_four(text)
        tax_rate = extraction.tax_rate or _fallback_tax_rate(text)
        tip_percentage = extraction.tip_percentage or _fallback_tip_percentage(text)

        items = [
            ReceiptItem(
                name=item.name,
                price=item.price,
                sku=item.sku,
                is_prescription=item.is_prescription,
                rx_number=item.rx_number,
            )
            for item in extraction.items
        ]

        return Receipt(
            receipt_id=receipt_id,
            merchant=extraction.merchant,
            address=extraction.address,
            city=extraction.city,
            state=extraction.state,
            date=parsed_date.date(),
            time=extraction.time,
            category=category,
            items=items,
            subtotal=extraction.subtotal,
            tax=extraction.tax,
            tax_rate=tax_rate,
            total=extraction.total,
            tip=extraction.tip,
            tip_percentage=tip_percentage,
            grand_total=extraction.grand_total,
            payment_method=extraction.payment_method,
            card_last_four=card_last_four,
            has_warranty=extraction.has_warranty,
            has_prescription=extraction.has_prescription,
            has_loyalty_discount=extraction.has_loyalty_discount,
            items_count=len(items),
        )

    def parse_directory(
        self, directory: Path
    ) -> tuple[list[Receipt], list[tuple[str, str]]]:
        """Parse all .txt receipt files in a directory.

        Returns (successes, failures) where failures is a list of (filename, error_msg).
        """
        receipts: list[Receipt] = []
        failures: list[tuple[str, str]] = []

        txt_files = sorted(directory.glob(RECEIPT_FILE_GLOB))
        total = len(txt_files)

        for i, filepath in enumerate(txt_files, 1):
            try:
                logger.info("Parsing %d/%d: %s", i, total, filepath.name)
                receipt = self.parse_file(filepath)
                receipts.append(receipt)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", filepath.name, e)
                failures.append((filepath.name, str(e)))

        logger.info(
            "Parsed %d/%d receipts successfully. %d skipped.",
            len(receipts),
            total,
            len(failures),
        )
        return receipts, failures
