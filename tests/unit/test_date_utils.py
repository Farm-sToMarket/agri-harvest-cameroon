"""
Tests for date utilities
"""

import pytest
from datetime import date, datetime, timezone, timedelta
from utils.date_utils import (
    get_utc_now,
    get_agricultural_season,
    get_day_of_year,
    calculate_growing_degree_days,
    get_date_range,
    validate_date_range,
    format_date_for_mongodb,
    parse_date_string,
)


class TestGetAgriculturalSeason:
    """Tests for bimodal/monomodal season classification"""

    # --- Southern Cameroon (bimodal, lat < 6 or lat=None) ---

    @pytest.mark.parametrize("month", [12, 1, 2])
    def test_bimodal_grand_dry_season(self, month):
        d = date(2024, month, 15)
        assert get_agricultural_season(d) == "grand_dry_season"
        assert get_agricultural_season(d, latitude=3.5) == "grand_dry_season"

    @pytest.mark.parametrize("month", [3, 4, 5])
    def test_bimodal_first_rainy_season(self, month):
        d = date(2024, month, 15)
        assert get_agricultural_season(d) == "first_rainy_season"

    @pytest.mark.parametrize("month", [6, 7])
    def test_bimodal_petit_dry_season(self, month):
        d = date(2024, month, 15)
        assert get_agricultural_season(d) == "petit_dry_season"

    @pytest.mark.parametrize("month", [8, 9, 10])
    def test_bimodal_second_rainy_season(self, month):
        d = date(2024, month, 15)
        assert get_agricultural_season(d) == "second_rainy_season"

    def test_bimodal_transition_to_dry(self):
        d = date(2024, 11, 15)
        assert get_agricultural_season(d) == "transition_to_dry"

    # --- Northern Cameroon (monomodal, lat >= 6) ---

    @pytest.mark.parametrize("month", [11, 12, 1, 2, 3])
    def test_monomodal_dry_season(self, month):
        d = date(2024, month, 15)
        assert get_agricultural_season(d, latitude=10.0) == "dry_season"

    def test_monomodal_early_rainy(self):
        d = date(2024, 4, 15)
        assert get_agricultural_season(d, latitude=8.0) == "early_rainy"

    @pytest.mark.parametrize("month", [5, 6, 7, 8, 9])
    def test_monomodal_rainy_season(self, month):
        d = date(2024, month, 15)
        assert get_agricultural_season(d, latitude=10.5) == "rainy_season"

    def test_monomodal_late_rainy(self):
        d = date(2024, 10, 15)
        assert get_agricultural_season(d, latitude=7.0) == "late_rainy"

    # --- Boundary tests ---

    def test_latitude_boundary_6_is_monomodal(self):
        """Latitude exactly 6.0 should be classified as monomodal"""
        d = date(2024, 7, 15)
        assert get_agricultural_season(d, latitude=6.0) == "rainy_season"

    def test_latitude_just_below_6_is_bimodal(self):
        """Latitude 5.99 should be bimodal"""
        d = date(2024, 7, 15)
        assert get_agricultural_season(d, latitude=5.99) == "petit_dry_season"

    def test_no_latitude_defaults_to_bimodal(self):
        d = date(2024, 7, 15)
        assert get_agricultural_season(d) == "petit_dry_season"


class TestGetDayOfYear:
    def test_jan_first(self):
        assert get_day_of_year(date(2024, 1, 1)) == 1

    def test_dec_31_non_leap(self):
        assert get_day_of_year(date(2023, 12, 31)) == 365

    def test_dec_31_leap(self):
        assert get_day_of_year(date(2024, 12, 31)) == 366


class TestCalculateGrowingDegreeDays:
    def test_basic_gdd(self):
        gdd = calculate_growing_degree_days(20.0, 30.0, base_temp=10.0)
        assert gdd == 15.0

    def test_gdd_below_base(self):
        gdd = calculate_growing_degree_days(5.0, 10.0, base_temp=10.0)
        assert gdd == 0.0

    def test_gdd_custom_base(self):
        gdd = calculate_growing_degree_days(20.0, 30.0, base_temp=15.0)
        assert gdd == 10.0


class TestGetDateRange:
    def test_same_day(self):
        d = date(2024, 1, 1)
        assert get_date_range(d, d) == [d]

    def test_three_days(self):
        start = date(2024, 1, 1)
        end = date(2024, 1, 3)
        result = get_date_range(start, end)
        assert len(result) == 3
        assert result[0] == start
        assert result[-1] == end

    def test_empty_range(self):
        start = date(2024, 1, 3)
        end = date(2024, 1, 1)
        assert get_date_range(start, end) == []


class TestValidateDateRange:
    def test_valid_range(self):
        valid, msg = validate_date_range(date(2020, 1, 1), date(2020, 12, 31))
        assert valid is True
        assert msg == ""

    def test_start_after_end(self):
        valid, msg = validate_date_range(date(2020, 12, 31), date(2020, 1, 1))
        assert valid is False
        assert "before" in msg.lower()

    def test_range_too_large(self):
        valid, msg = validate_date_range(date(2000, 1, 1), date(2020, 1, 1))
        assert valid is False
        assert "10 years" in msg

    def test_future_start(self):
        future = date.today() + timedelta(days=30)
        valid, msg = validate_date_range(future, None)
        assert valid is False
        assert "future" in msg.lower()

    def test_none_values(self):
        valid, msg = validate_date_range(None, None)
        assert valid is True


class TestFormatDateForMongodb:
    def test_naive_datetime(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = format_date_for_mongodb(dt)
        assert result.tzinfo == timezone.utc

    def test_utc_datetime_unchanged(self):
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = format_date_for_mongodb(dt)
        assert result == dt


class TestParseDateString:
    def test_iso_format(self):
        assert parse_date_string("2024-01-15") == date(2024, 1, 15)

    def test_european_format(self):
        assert parse_date_string("15/01/2024") == date(2024, 1, 15)

    def test_invalid_format(self):
        assert parse_date_string("not-a-date") is None

    def test_empty_string(self):
        assert parse_date_string("") is None
