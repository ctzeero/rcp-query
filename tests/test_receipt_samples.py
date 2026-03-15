"""
Verification tests for the receipt_samples/ dataset.

Validates receipt count, category distribution, total spending,
date range, merchant presence, and structural integrity of each file.

Run with: python -m pytest tests/test_receipt_samples.py -v -s 2>&1
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "receipt_samples"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOTAL_RE = re.compile(r"^TOTAL:?\s+\$\s*([\d,]+\.\d{2})", re.MULTILINE)
_SUBTOTAL_RE = re.compile(r"^SUBTOTAL:?\s+\$\s*([\d,]+\.\d{2})", re.MULTILINE)
_TAX_RE = re.compile(r"^(?:SALES )?TAX.*\$\s*([\d,]+\.\d{2})", re.MULTILINE)
_DATE_RE = re.compile(
    r"(?:Date:\s*)?(\d{2}/\d{2}/\d{4})"
)
_FILENAME_RE = re.compile(
    r"^receipt_(\d{3})_([a-z_]+)_(\d{8})\.txt$"
)

EXPECTED_CATEGORIES = {
    "grocery": 25,
    "restaurant": 20,
    "coffee": 15,
    "fast_food": 15,
    "electronics": 8,
    "pharmacy": 7,
    "retail": 5,
    "hardware": 3,
    "gas": 2,
}

EXPECTED_MERCHANTS = {
    "grocery": ["Whole Foods", "Trader Joe", "Safeway", "Costco", "Target", "Walmart"],
    "restaurant": ["Thai", "Sushi", "Pizza", "Mediterranean", "Burger", "Pho"],
    "coffee": ["Blue Bottle", "Starbucks", "Peet", "Philz", "Ritual"],
    "fast_food": ["Chipotle", "McDonald", "Subway", "Panera", "Taco Bell"],
    "electronics": ["Best Buy", "Apple", "Micro Center", "B&H"],
    "pharmacy": ["CVS", "Walgreens", "Rite Aid"],
    "hardware": ["Lowe", "Ace Hardware"],
    "gas": ["Chevron", "76"],
}


def _receipt_files() -> list[Path]:
    """Return all receipt .txt files sorted by name."""
    return sorted(SAMPLES_DIR.glob("receipt_*.txt"))


def _parse_amount(text: str) -> float:
    return float(text.replace(",", ""))


def _parse_receipt(path: Path) -> dict:
    """Extract key fields from a receipt file."""
    content = path.read_text()
    total_m = _TOTAL_RE.search(content)
    subtotal_m = _SUBTOTAL_RE.search(content)
    tax_m = _TAX_RE.search(content)
    date_m = _DATE_RE.search(content)

    fname_m = _FILENAME_RE.match(path.name)
    category = fname_m.group(2) if fname_m else None
    receipt_num = int(fname_m.group(1)) if fname_m else None
    file_date = fname_m.group(3) if fname_m else None

    return {
        "path": path,
        "content": content,
        "total": _parse_amount(total_m.group(1)) if total_m else None,
        "subtotal": _parse_amount(subtotal_m.group(1)) if subtotal_m else None,
        "tax": _parse_amount(tax_m.group(1)) if tax_m else None,
        "date_str": date_m.group(1) if date_m else None,
        "category": category,
        "receipt_num": receipt_num,
        "file_date": file_date,
        "merchant": content.splitlines()[0].strip(),
    }


# Shared fixture — parsed once
@pytest.fixture(scope="module")
def receipts() -> list[dict]:
    files = _receipt_files()
    assert len(files) > 0, "No receipt files found"
    return [_parse_receipt(f) for f in files]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReceiptCount:
    def test_total_receipt_count(self, receipts):
        assert len(receipts) == 100

    def test_receipt_numbers_are_001_to_100(self, receipts):
        nums = sorted(r["receipt_num"] for r in receipts)
        assert nums == list(range(1, 101))


class TestCategoryDistribution:
    def test_all_expected_categories_present(self, receipts):
        categories = {r["category"] for r in receipts}
        assert categories == set(EXPECTED_CATEGORIES.keys())

    @pytest.mark.parametrize(
        "category,expected_count", list(EXPECTED_CATEGORIES.items())
    )
    def test_category_count(self, receipts, category, expected_count):
        actual = sum(1 for r in receipts if r["category"] == category)
        assert actual == expected_count, (
            f"{category}: expected {expected_count}, got {actual}"
        )


class TestMerchantPresence:
    @pytest.mark.parametrize(
        "category,keywords", list(EXPECTED_MERCHANTS.items())
    )
    def test_merchants_appear_in_category(self, receipts, category, keywords):
        """At least one receipt per category should mention each expected merchant."""
        cat_receipts = [r for r in receipts if r["category"] == category]
        all_text = " ".join(r["content"] for r in cat_receipts)
        for kw in keywords:
            assert kw.lower() in all_text.lower(), (
                f"Merchant keyword '{kw}' not found in any {category} receipt"
            )


class TestDateRange:
    def test_dates_within_expected_range(self, receipts):
        """All receipts should be between Nov 1 2023 and Jan 31 2024."""
        from datetime import datetime

        dates = []
        for r in receipts:
            assert r["date_str"] is not None, f"No date found in {r['path'].name}"
            dt = datetime.strptime(r["date_str"], "%m/%d/%Y")
            assert datetime(2023, 11, 1) <= dt <= datetime(2024, 1, 31), (
                f"{r['path'].name}: date {r['date_str']} out of range"
            )
            dates.append(dt)
        earliest = min(dates).strftime("%B %d, %Y")
        latest = max(dates).strftime("%B %d, %Y")
        print(f"\n  Date Range: {earliest} – {latest}")

    def test_file_date_matches_content_date(self, receipts):
        """The YYYYMMDD in the filename should match the date inside."""
        from datetime import datetime

        for r in receipts:
            if r["date_str"] is None or r["file_date"] is None:
                continue
            content_dt = datetime.strptime(r["date_str"], "%m/%d/%Y")
            file_dt = datetime.strptime(r["file_date"], "%Y%m%d")
            assert content_dt.date() == file_dt.date(), (
                f"{r['path'].name}: filename date {r['file_date']} "
                f"!= content date {r['date_str']}"
            )


class TestTotalSpending:
    def test_grand_total_matches(self, receipts):
        """Sum of all TOTAL lines should equal $7,556.94 (computed earlier)."""
        total = sum(r["total"] for r in receipts if r["total"] is not None)
        print(f"\n  Total Spending: ${total:,.2f}")
        assert abs(total - 7556.94) < 0.01, f"Total spending: ${total:.2f}"

    def test_total_in_expected_ballpark(self, receipts):
        """Sanity check: total should be roughly $7,000–$10,000."""
        total = sum(r["total"] for r in receipts if r["total"] is not None)
        print(f"\n  Total Spending: ${total:,.2f} (expected $7,000–$10,000)")
        assert 7000 <= total <= 10000, f"Total ${total:.2f} outside expected range"


class TestReceiptStructure:
    def test_every_receipt_has_total(self, receipts):
        missing = [r["path"].name for r in receipts if r["total"] is None]
        assert not missing, f"Receipts missing TOTAL: {missing}"

    def test_every_receipt_has_subtotal(self, receipts):
        missing = [r["path"].name for r in receipts if r["subtotal"] is None]
        assert not missing, f"Receipts missing SUBTOTAL: {missing}"

    def test_every_receipt_has_tax(self, receipts):
        missing = [r["path"].name for r in receipts if r["tax"] is None]
        assert not missing, f"Receipts missing TAX: {missing}"

    def test_total_equals_subtotal_plus_tax(self, receipts):
        """TOTAL should equal SUBTOTAL + TAX (within rounding tolerance)."""
        mismatches = []
        for r in receipts:
            if r["total"] and r["subtotal"] and r["tax"]:
                expected = r["subtotal"] + r["tax"]
                if abs(r["total"] - expected) > 0.02:
                    mismatches.append(
                        f"{r['path'].name}: {r['subtotal']} + {r['tax']} = "
                        f"{expected:.2f} != {r['total']}"
                    )
        assert not mismatches, f"TOTAL != SUBTOTAL + TAX:\n" + "\n".join(mismatches)

    def test_no_empty_receipts(self, receipts):
        for r in receipts:
            assert len(r["content"].strip()) > 50, (
                f"{r['path'].name} appears empty or too short"
            )

    def test_merchant_name_on_first_line(self, receipts):
        for r in receipts:
            assert len(r["merchant"]) > 0, (
                f"{r['path'].name} has no merchant on first line"
            )
