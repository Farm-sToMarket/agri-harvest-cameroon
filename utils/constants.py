"""
Constants for the Cameroon Agricultural Data Management System
Loaded from YAML configuration files.
"""

from config.yaml_loader import load_geography, load_agriculture

_geo = load_geography()
_agri = load_agriculture()

# Geographic bounds for Cameroon
CAMEROON_BOUNDS = _geo["cameroon_bounds"]

# Elevation ranges
ELEVATION_RANGE = _geo["elevation_range"]

# Agroecological zones
AGROECOLOGICAL_ZONES = _geo["agroecological_zones"]

# Main crops in Cameroon
MAIN_CROPS = _agri["main_crops"]

# Weather station types
WEATHER_STATION_TYPES = _agri["weather_station_types"]

# Soil texture classes (USDA classification)
SOIL_TEXTURE_CLASSES = _agri["soil_texture_classes"]

# Data quality levels
DATA_QUALITY_LEVELS = _agri["data_quality_levels"]

# IRAD research centers
IRAD_CENTERS = _agri["irad_centers"]

# Default field validation ranges
VALIDATION_RANGES = _geo["validation_ranges"]
