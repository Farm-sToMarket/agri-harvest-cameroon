"""
Constants for the Cameroon Agricultural Data Management System
"""

from typing import Dict, List

# Geographic bounds for Cameroon
CAMEROON_BOUNDS = {
    "north": 13.1,
    "south": 1.6,
    "east": 16.2,
    "west": 8.3
}

# Elevation ranges
ELEVATION_RANGE = {
    "min": 0,      # Sea level at coast
    "max": 4095    # Mount Cameroon
}

# Agroecological zones
AGROECOLOGICAL_ZONES = [
    "sahel_savanna",
    "sudan_savanna",
    "guinea_savanna",
    "forest_savanna_transition",
    "humid_forest_inland",
    "humid_forest_coast",
    "western_highlands",
    "mont_cameroun_volcanic"
]

# Main crops in Cameroon
MAIN_CROPS = [
    "maize", "rice", "cassava", "potato", "tomato", "cocoa",
    "groundnut", "sorghum", "millet", "cowpea", "plantain_banana", "yam", "cotton"
]

# Weather station types
WEATHER_STATION_TYPES = [
    "automatic",
    "manual",
    "hybrid",
    "satellite_derived"
]

# Soil texture classes (USDA classification)
SOIL_TEXTURE_CLASSES = [
    "sand", "loamy_sand", "sandy_loam", "loam", "silt_loam", "silt",
    "sandy_clay_loam", "clay_loam", "silty_clay_loam", "sandy_clay",
    "silty_clay", "clay"
]

# Data quality levels
DATA_QUALITY_LEVELS = [
    "very_low",
    "low",
    "medium",
    "high",
    "very_high"
]

# IRAD research centers
IRAD_CENTERS = {
    "centre_sud": {
        "location": "Nkolbisson, Yaounde",
        "coordinates": {"lat": 3.8667, "lon": 11.5167},
        "elevation": 650,
        "agroecological_zone": "humid_forest_inland",
        "research_focus": ["maize", "groundnut", "cassava", "plantain_banana"],
        "data_quality": "high",
    },
    "west_highlands": {
        "location": "Bambili, Bamenda",
        "coordinates": {"lat": 5.9833, "lon": 10.2500},
        "elevation": 2000,
        "agroecological_zone": "western_highlands",
        "research_focus": ["potato", "maize", "cowpea"],
        "data_quality": "high",
    },
    "littoral": {
        "location": "Ekona, Buea",
        "coordinates": {"lat": 4.2000, "lon": 9.3500},
        "elevation": 450,
        "agroecological_zone": "humid_forest_coast",
        "research_focus": ["cocoa", "plantain_banana", "cassava"],
        "data_quality": "medium",
    },
    "far_north": {
        "location": "Maroua",
        "coordinates": {"lat": 10.5833, "lon": 14.3167},
        "elevation": 420,
        "agroecological_zone": "sahel_savanna",
        "research_focus": ["sorghum", "millet", "cotton", "groundnut"],
        "data_quality": "medium",
    },
    "south_west": {
        "location": "Kumba",
        "coordinates": {"lat": 4.6333, "lon": 9.4500},
        "elevation": 100,
        "agroecological_zone": "humid_forest_coast",
        "research_focus": ["cocoa", "plantain_banana", "cassava"],
        "data_quality": "medium",
    },
}

# Default field validation ranges
VALIDATION_RANGES = {
    "temperature": {"min": -5, "max": 50},
    "precipitation": {"min": 0, "max": 500},
    "humidity": {"min": 0, "max": 100},
    "ph": {"min": 3.5, "max": 9.5},
    "organic_carbon": {"min": 0.1, "max": 10.0}
}