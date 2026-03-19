"""
Tests for geospatial utilities
"""

import pytest
import math
from utils.geospatial_utils import (
    validate_coordinates,
    calculate_distance,
    create_geojson_point,
    get_bounding_box,
    determine_agroecological_zone,
    calculate_slope,
    convert_coordinates_to_utm,
)


class TestValidateCoordinates:
    def test_valid_yaounde(self):
        valid, msg = validate_coordinates(3.8667, 11.5167)
        assert valid is True

    def test_valid_maroua(self):
        valid, msg = validate_coordinates(10.5833, 14.3167)
        assert valid is True

    def test_latitude_too_low(self):
        valid, msg = validate_coordinates(1.0, 11.0)
        assert valid is False
        assert "Latitude" in msg

    def test_latitude_too_high(self):
        valid, msg = validate_coordinates(14.0, 11.0)
        assert valid is False

    def test_longitude_too_low(self):
        valid, msg = validate_coordinates(5.0, 7.0)
        assert valid is False
        assert "Longitude" in msg

    def test_longitude_too_high(self):
        valid, msg = validate_coordinates(5.0, 17.0)
        assert valid is False

    def test_boundary_values(self):
        valid, _ = validate_coordinates(1.6, 8.3)
        assert valid is True
        valid, _ = validate_coordinates(13.1, 16.2)
        assert valid is True


class TestCalculateDistance:
    def test_same_point(self):
        dist = calculate_distance(3.8667, 11.5167, 3.8667, 11.5167)
        assert dist == 0.0

    def test_yaounde_to_douala(self):
        # Approximately 210 km
        dist = calculate_distance(3.8667, 11.5167, 4.0511, 9.7679)
        assert 190 < dist < 230

    def test_symmetry(self):
        d1 = calculate_distance(3.8667, 11.5167, 10.5833, 14.3167)
        d2 = calculate_distance(10.5833, 14.3167, 3.8667, 11.5167)
        assert abs(d1 - d2) < 0.001


class TestCreateGeojsonPoint:
    def test_basic(self):
        point = create_geojson_point(11.5167, 3.8667)
        assert point["type"] == "Point"
        assert point["coordinates"] == [11.5167, 3.8667]

    def test_geojson_order(self):
        """GeoJSON uses [longitude, latitude] order"""
        point = create_geojson_point(longitude=11.0, latitude=4.0)
        assert point["coordinates"][0] == 11.0  # longitude first
        assert point["coordinates"][1] == 4.0   # latitude second


class TestGetBoundingBox:
    def test_symmetric(self):
        bbox = get_bounding_box(5.0, 10.0, 10.0)
        assert bbox["north"] > 5.0
        assert bbox["south"] < 5.0
        assert bbox["east"] > 10.0
        assert bbox["west"] < 10.0

    def test_symmetric_lat_offsets(self):
        bbox = get_bounding_box(5.0, 10.0, 10.0)
        lat_diff_north = bbox["north"] - 5.0
        lat_diff_south = 5.0 - bbox["south"]
        assert abs(lat_diff_north - lat_diff_south) < 0.001


class TestDetermineAgroecologicalZone:
    """Tests for improved agroecological zone classification"""

    # --- Northern zones (latitude-driven) ---

    def test_sahel_savanna_far_north(self):
        """Maroua region, far north"""
        assert determine_agroecological_zone(10.5, 14.3, 420) == "sahel_savanna"

    def test_sudan_savanna_north(self):
        """Garoua region"""
        assert determine_agroecological_zone(9.3, 13.4, 213) == "sudan_savanna"

    def test_guinea_savanna_adamawa(self):
        """Ngaoundere region"""
        assert determine_agroecological_zone(7.3, 13.6, 1100) == "guinea_savanna"

    # --- Western highlands ---

    def test_western_highlands_bamenda(self):
        """Bamenda at high elevation"""
        assert determine_agroecological_zone(5.95, 10.15, 1600) == "western_highlands"

    def test_western_highlands_not_low_elevation(self):
        """Same lat/lon as highlands but low elevation"""
        assert determine_agroecological_zone(5.95, 10.15, 500) != "western_highlands"

    def test_western_highlands_not_eastern_cameroon(self):
        """High elevation but in eastern Cameroon"""
        assert determine_agroecological_zone(6.5, 12.0, 1500) == "guinea_savanna"

    def test_western_highlands_latitude_bounds(self):
        """Check that western highlands requires lat between 4.5 and 7.5"""
        assert determine_agroecological_zone(4.4, 10.0, 1500) != "western_highlands"

    # --- Transition zone ---

    def test_forest_savanna_transition(self):
        """Bertoua region, transition belt"""
        assert determine_agroecological_zone(5.5, 13.5, 600) == "forest_savanna_transition"

    # --- Southern zones ---

    def test_humid_forest_coast_douala(self):
        """Douala - coastal, low elevation"""
        assert determine_agroecological_zone(4.05, 9.77, 13) == "humid_forest_coast"

    def test_humid_forest_coast_limbe(self):
        """Limbe - coastal"""
        assert determine_agroecological_zone(4.0, 9.2, 50) == "humid_forest_coast"

    def test_humid_forest_inland_yaounde(self):
        """Yaounde - interior forest"""
        assert determine_agroecological_zone(3.87, 11.52, 650) == "humid_forest_inland"

    def test_humid_forest_inland_ebolowa(self):
        """Ebolowa - deep south interior"""
        assert determine_agroecological_zone(2.9, 11.15, 600) == "humid_forest_inland"

    def test_mont_cameroun_volcanic(self):
        """Mont Cameroun high-altitude volcanic zone"""
        assert determine_agroecological_zone(4.2, 9.17, 3000) == "mont_cameroun_volcanic"

    def test_mont_cameroun_low_elevation_not_volcanic(self):
        """Same lat/lon but below 2500m should not be mont_cameroun_volcanic"""
        assert determine_agroecological_zone(4.2, 9.17, 500) != "mont_cameroun_volcanic"


class TestCalculateSlope:
    def test_insufficient_data(self):
        assert calculate_slope([(3.0, 11.0, 600), (3.1, 11.1, 700)]) is None

    def test_flat_terrain(self):
        data = [(3.0, 11.0, 600), (3.1, 11.0, 600), (3.0, 11.1, 600)]
        slope = calculate_slope(data)
        assert slope == 0.0

    def test_positive_slope(self):
        data = [(3.0, 11.0, 600), (3.01, 11.0, 800), (3.02, 11.0, 1000)]
        slope = calculate_slope(data)
        assert slope is not None
        assert slope > 0


class TestConvertCoordinatesToUtm:
    def test_zone_32(self):
        result = convert_coordinates_to_utm(4.0, 10.0)
        assert result["zone"] == "32N"
        assert result["datum"] == "WGS84"

    def test_zone_33(self):
        result = convert_coordinates_to_utm(4.0, 13.0)
        assert result["zone"] == "33N"

    def test_boundary(self):
        result = convert_coordinates_to_utm(4.0, 12.0)
        assert result["zone"] == "33N"
