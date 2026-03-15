"""Deterministic date resolver for natural-language temporal expressions.

Runs *before* the LLM and uses regex + holidays + dateutil so that
expressions like "Q4 2023", "Thanksgiving week", or "11/07/2023"
always produce correct date ranges regardless of LLM quality.
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import date, datetime, timedelta
from typing import Optional

import holidays

logger = logging.getLogger(__name__)

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_HOLIDAY_ALIASES: dict[str, str] = {
    "thanksgiving": "Thanksgiving",
    "christmas": "Christmas",
    "new year": "New Year",
    "new years": "New Year",
    "labor day": "Labor Day",
    "memorial day": "Memorial Day",
    "independence day": "Independence Day",
    "july 4th": "Independence Day",
    "4th of july": "Independence Day",
    "veterans day": "Veterans Day",
    "black friday": "Black Friday",
    "mlk": "Martin Luther King",
    "martin luther king": "Martin Luther King",
}


def _iso(d: date) -> str:
    return d.isoformat()


def _month_range(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return _iso(date(year, month, 1)), _iso(date(year, month, last_day))


def _week_of(d: date) -> tuple[str, str]:
    """Return Monday-Sunday of the week containing *d*."""
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return _iso(monday), _iso(sunday)


def _get_holiday_date(name_key: str, year: int) -> date | None:
    """Look up a US holiday by alias for a given year."""
    if name_key == "Black Friday":
        us = holidays.US(years=year)
        dates = us.get_named("Thanksgiving")
        if dates:
            return dates[0] + timedelta(days=1)
        return None

    us = holidays.US(years=year)
    dates = us.get_named(name_key)
    observed = [d for d in dates if "observed" not in us.get(d, "").lower()]
    return observed[0] if observed else (dates[0] if dates else None)


def _infer_year(ref: date, month: int | None = None) -> int:
    """Pick the most likely year given the reference date and dataset range."""
    if month is not None:
        if month >= 10:
            return ref.year - 1 if ref.month <= 6 else ref.year
        if month <= 2:
            return ref.year
    return ref.year


def resolve_dates(
    query: str,
    reference_date: str,
    dataset_end: str | None = None,
) -> tuple[str, str] | None:
    """Try to extract a (start, end) date range from *query* deterministically.

    *reference_date* is "today" -- used for relative expressions like
    "last week", "recently".
    *dataset_end* anchors year inference for bare month names and holidays
    (e.g. "December" -> Dec 2023, not Dec 2025).  Falls back to
    *reference_date* if not provided.

    Returns ``None`` when no temporal expression is recognised.
    """
    q = query.lower().strip()
    ref = datetime.strptime(reference_date, "%Y-%m-%d").date()
    ds_end = datetime.strptime(dataset_end, "%Y-%m-%d").date() if dataset_end else ref

    # --- Explicit MM/DD/YYYY ---
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", query)
    if m:
        dt = datetime.strptime(m.group(0), "%m/%d/%Y").date()
        return _iso(dt), _iso(dt)

    # --- Quarter: Q1-Q4 YYYY ---
    m = re.search(r"\bq([1-4])\s*(\d{4})\b", q)
    if m:
        quarter, year = int(m.group(1)), int(m.group(2))
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3
        last_day = calendar.monthrange(year, end_month)[1]
        return _iso(date(year, start_month, 1)), _iso(date(year, end_month, last_day))

    # --- Holiday week / day ---
    for alias, canonical in _HOLIDAY_ALIASES.items():
        if alias not in q:
            continue
        # Use dataset_end to anchor year: pick the most recent holiday
        # on or before the dataset end date.
        hol_date = None
        for y in (ds_end.year, ds_end.year - 1):
            candidate = _get_holiday_date(canonical, y)
            if candidate and candidate <= ds_end:
                hol_date = candidate
                break
        if hol_date is None:
            hol_date = _get_holiday_date(canonical, ds_end.year)
        if not hol_date:
            continue

        if "week before" in q:
            end = hol_date - timedelta(days=1)
            start = end - timedelta(days=6)
            return _iso(start), _iso(end)
        if "week" in q:
            return _week_of(hol_date)
        return _iso(hol_date), _iso(hol_date)

    # --- "first week of Month" / "last week of Month" ---
    m = re.search(r"(first|last)\s+week\s+of\s+(\w+)", q)
    if m:
        pos, month_str = m.group(1), m.group(2)
        month_num = _MONTH_NAMES.get(month_str)
        if month_num:
            year = _infer_year(ds_end, month_num)
            if pos == "first":
                start = date(year, month_num, 1)
                end = start + timedelta(days=6)
            else:
                last_day = calendar.monthrange(year, month_num)[1]
                end = date(year, month_num, last_day)
                start = end - timedelta(days=6)
            return _iso(start), _iso(end)

    # --- "Month YYYY" or "Month" alone ---
    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"(?:\s+(\d{4}))?\b",
        q,
    )
    if m:
        month_num = _MONTH_NAMES[m.group(1)]
        year = int(m.group(2)) if m.group(2) else _infer_year(ds_end, month_num)
        return _month_range(year, month_num)

    # --- Relative: "past/last N days" ---
    m = re.search(r"\b(?:past|last|previous)\s+(\d+)\s+days?\b", q)
    if m:
        n = int(m.group(1))
        start = ref - timedelta(days=n)
        return _iso(start), _iso(ref)

    # --- Relative: "last week", "this week" ---
    if re.search(r"\blast\s+week\b", q):
        end = ref - timedelta(days=ref.weekday() + 1)
        start = end - timedelta(days=6)
        return _iso(start), _iso(end)
    if re.search(r"\bthis\s+week\b", q):
        return _week_of(ref)

    # --- Relative: "last month", "this month" ---
    if re.search(r"\blast\s+month\b", q):
        first_this = date(ref.year, ref.month, 1)
        last_prev = first_this - timedelta(days=1)
        first_prev = date(last_prev.year, last_prev.month, 1)
        return _iso(first_prev), _iso(last_prev)
    if re.search(r"\bthis\s+month\b", q):
        return _month_range(ref.year, ref.month)

    # --- "recently" = last 2 weeks ---
    if re.search(r"\brecently\b", q):
        start = ref - timedelta(days=14)
        return _iso(start), _iso(ref)

    # --- Year alone: "in 2023" ---
    m = re.search(r"\bin\s+(\d{4})\b", q)
    if m:
        year = int(m.group(1))
        return _iso(date(year, 1, 1)), _iso(date(year, 12, 31))

    return None


def dates_outside_dataset(
    date_range,
    dataset_start: str,
    dataset_end: str,
) -> bool:
    """Return True when *date_range* is entirely outside the dataset window."""
    if date_range is None or (not date_range.start and not date_range.end):
        return False
    if date_range.start and date_range.start > dataset_end:
        return True
    if date_range.end and date_range.end < dataset_start:
        return True
    return False
