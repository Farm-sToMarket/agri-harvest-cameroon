"""
Date and time utilities for agricultural data processing
"""

from datetime import datetime, date, timezone, timedelta
from typing import Optional, Tuple


def get_utc_now() -> datetime:
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


def get_agricultural_season(date_obj: date, latitude: Optional[float] = None) -> str:
    """
    Determine agricultural season based on Cameroon's climate patterns.

    Cameroon has two distinct rainfall regimes:
    - Bimodal (South, latitude < 6N): two rainy seasons separated by a short dry spell
    - Monomodal (North, latitude >= 6N): single rainy season with a long dry season

    Args:
        date_obj: Date to classify
        latitude: Latitude in decimal degrees. If None, defaults to bimodal (southern pattern).

    Returns:
        Season name
    """
    month = date_obj.month
    is_monomodal = latitude is not None and latitude >= 6.0

    if is_monomodal:
        # Northern Cameroon: single rainy season (May-October)
        if month in (11, 12, 1, 2, 3):
            return "dry_season"
        elif month == 4:
            return "early_rainy"
        elif month in (5, 6, 7, 8, 9):
            return "rainy_season"
        elif month == 10:
            return "late_rainy"
    else:
        # Southern Cameroon: bimodal rainfall
        if month in (12, 1, 2):
            return "grand_dry_season"
        elif month in (3, 4, 5):
            return "first_rainy_season"
        elif month in (6, 7):
            return "petit_dry_season"
        elif month in (8, 9, 10):
            return "second_rainy_season"
        elif month == 11:
            return "transition_to_dry"

    return "unknown"


def get_day_of_year(date_obj: date) -> int:
    """Get day of year (1-366)"""
    return date_obj.timetuple().tm_yday


def calculate_growing_degree_days(
    temp_min: float,
    temp_max: float,
    base_temp: float = 10.0
) -> float:
    """
    Calculate growing degree days using the standard formula

    Args:
        temp_min: Daily minimum temperature (C)
        temp_max: Daily maximum temperature (C)
        base_temp: Base temperature for crop growth (C)

    Returns:
        Growing degree days value
    """
    temp_avg = (temp_min + temp_max) / 2
    return max(0, temp_avg - base_temp)


def get_date_range(
    start_date: date,
    end_date: date
) -> list[date]:
    """
    Generate list of dates between start and end dates (inclusive)

    Args:
        start_date: Starting date
        end_date: Ending date

    Returns:
        List of dates
    """
    dates = []
    current_date = start_date

    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    return dates


def validate_date_range(
    start_date: Optional[date],
    end_date: Optional[date]
) -> Tuple[bool, str]:
    """
    Validate date range for data queries

    Args:
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        Tuple of (is_valid, error_message)
    """
    if start_date and end_date:
        if start_date > end_date:
            return False, "Start date must be before end date"

        # Check if date range is too large (more than 10 years)
        if (end_date - start_date).days > 3650:
            return False, "Date range cannot exceed 10 years"

    # Check if dates are in the future (for historical data)
    today = date.today()
    if start_date and start_date > today:
        return False, "Start date cannot be in the future"
    if end_date and end_date > today:
        return False, "End date cannot be in the future"

    return True, ""


def format_date_for_mongodb(date_obj: datetime) -> datetime:
    """
    Ensure datetime is UTC and properly formatted for MongoDB

    Args:
        date_obj: DateTime object to format

    Returns:
        UTC datetime object
    """
    if date_obj.tzinfo is None:
        # Assume naive datetime is UTC
        return date_obj.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC
        return date_obj.astimezone(timezone.utc)


def parse_date_string(date_str: str) -> Optional[date]:
    """
    Parse date string in various common formats

    Args:
        date_str: Date string to parse

    Returns:
        Parsed date object or None if parsing fails
    """
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m-%d-%Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    return None
