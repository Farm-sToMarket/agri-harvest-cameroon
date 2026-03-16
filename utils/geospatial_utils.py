"""
Geospatial utilities for agricultural data processing
"""

import math
from typing import Tuple, Dict, Any, List, Optional
from utils.constants import CAMEROON_BOUNDS


def validate_coordinates(latitude: float, longitude: float) -> Tuple[bool, str]:
    """
    Validate coordinates are within Cameroon bounds

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (CAMEROON_BOUNDS["south"] <= latitude <= CAMEROON_BOUNDS["north"]):
        return False, f"Latitude {latitude} is outside Cameroon bounds"

    if not (CAMEROON_BOUNDS["west"] <= longitude <= CAMEROON_BOUNDS["east"]):
        return False, f"Longitude {longitude} is outside Cameroon bounds"

    return True, ""


def calculate_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate distance between two points using Haversine formula

    Args:
        lat1, lon1: First point coordinates (decimal degrees)
        lat2, lon2: Second point coordinates (decimal degrees)

    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)

    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Earth's radius in kilometers

    return c * r


def create_geojson_point(longitude: float, latitude: float) -> Dict[str, Any]:
    """
    Create GeoJSON Point geometry

    Args:
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees

    Returns:
        GeoJSON Point geometry
    """
    return {
        "type": "Point",
        "coordinates": [longitude, latitude]
    }


def create_mongodb_spatial_query(
    latitude: float,
    longitude: float,
    radius_km: float = 10.0
) -> Dict[str, Any]:
    """
    Create MongoDB spatial query for finding nearby points

    Args:
        latitude: Center point latitude
        longitude: Center point longitude
        radius_km: Search radius in kilometers

    Returns:
        MongoDB spatial query
    """
    return {
        "coordinates": {
            "$near": {
                "$geometry": create_geojson_point(longitude, latitude),
                "$maxDistance": radius_km * 1000  # Convert to meters
            }
        }
    }


def get_bounding_box(
    latitude: float,
    longitude: float,
    radius_km: float
) -> Dict[str, float]:
    """
    Calculate bounding box around a point

    Args:
        latitude: Center point latitude
        longitude: Center point longitude
        radius_km: Radius in kilometers

    Returns:
        Dictionary with north, south, east, west bounds
    """
    # Approximate degrees per kilometer
    lat_deg_per_km = 1 / 111.0
    lon_deg_per_km = 1 / (111.0 * math.cos(math.radians(latitude)))

    lat_offset = radius_km * lat_deg_per_km
    lon_offset = radius_km * lon_deg_per_km

    return {
        "north": latitude + lat_offset,
        "south": latitude - lat_offset,
        "east": longitude + lon_offset,
        "west": longitude - lon_offset
    }


def determine_agroecological_zone(
    latitude: float,
    longitude: float,
    elevation: float
) -> str:
    """
    Determine agroecological zone based on coordinates and elevation.

    Uses a refined classification that accounts for Cameroon's north-south
    gradient, the western highlands massif, and the coastal zone.

    Reference zones (IRAD/FAO classification):
    - sahel_savanna: Far North (lat > 10), low elevation, < 600 mm rainfall
    - sudan_savanna: North (lat 8-10), intermediate elevation
    - guinea_savanna: Adamawa Plateau (lat 6-8)
    - western_highlands: Bamenda-Bamboutos massif (elevation > 1200m, lon < 11.5)
    - forest_savanna_transition: Transition belt (lat 5-6, not highlands)
    - humid_forest_coast: Coastal lowlands (lat < 5, lon < 10, elev < 500)
    - humid_forest_inland: Congo basin forest (lat < 5, interior)

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        elevation: Elevation in meters

    Returns:
        Agroecological zone name
    """
    # Mont Cameroun volcanic zone: high-altitude volcanic slopes
    if (4.0 <= latitude <= 4.35 and 9.0 <= longitude <= 9.35 and elevation > 2500):
        return "mont_cameroun_volcanic"

    # Western highlands: high-elevation massif in western Cameroon
    # Covers Bamenda, Bamboutos, and parts of the volcanic chain
    if elevation > 1200 and longitude < 11.5 and 4.5 < latitude < 7.5:
        return "western_highlands"

    # Northern zones (latitude-driven, north to south)
    if latitude > 10.0:
        return "sahel_savanna"

    if latitude > 8.0:
        return "sudan_savanna"

    if latitude > 6.0:
        return "guinea_savanna"

    # Transition belt
    if latitude > 5.0:
        return "forest_savanna_transition"

    # Southern zones: coastal vs inland distinction
    # Coastal zone: western, low-elevation, high-rainfall belt
    # Includes Douala basin, Limbe, Kribi corridor
    if longitude < 10.0 and elevation < 500:
        return "humid_forest_coast"

    # Default: interior forest (Centre, South, East regions)
    return "humid_forest_inland"


def calculate_slope(elevation_data: List[Tuple[float, float, float]]) -> Optional[float]:
    """
    Calculate slope from elevation data points

    Args:
        elevation_data: List of (lat, lon, elevation) tuples

    Returns:
        Slope in degrees or None if insufficient data
    """
    if len(elevation_data) < 3:
        return None

    # Simplified slope calculation using min and max elevations
    elevations = [point[2] for point in elevation_data]
    min_elev = min(elevations)
    max_elev = max(elevations)

    # Calculate horizontal distance between extreme points
    min_point = next(p for p in elevation_data if p[2] == min_elev)
    max_point = next(p for p in elevation_data if p[2] == max_elev)

    horizontal_distance = calculate_distance(
        min_point[0], min_point[1],
        max_point[0], max_point[1]
    ) * 1000  # Convert to meters

    if horizontal_distance == 0:
        return 0.0

    # Calculate slope in degrees
    elevation_diff = max_elev - min_elev
    slope_radians = math.atan(elevation_diff / horizontal_distance)
    slope_degrees = math.degrees(slope_radians)

    return slope_degrees


def convert_coordinates_to_utm(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Convert WGS84 coordinates to UTM (simplified for Cameroon)

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees

    Returns:
        Dictionary with UTM zone, easting, and northing
    """
    # Cameroon spans UTM zones 32N and 33N
    # This is a simplified conversion - in production use pyproj
    utm_zone = 32 if longitude < 12.0 else 33

    central_meridian = (utm_zone - 1) * 6 - 180 + 3

    return {
        "zone": f"{utm_zone}N",
        "easting": (longitude - central_meridian) * 111320,
        "northing": latitude * 111320,
        "datum": "WGS84"
    }
