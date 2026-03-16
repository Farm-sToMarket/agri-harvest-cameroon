"""
Configuration module for Cameroon Agricultural Data Management System
"""

from .settings import Settings, get_settings
from .database import DatabaseConfig, get_database
from .schema import (
    SoilSchema,
    SoilDataModel,
    WeatherSchema,
    WeatherDataModel,
    CropSchema,
    CropDataModel,
)

__all__ = [
    "Settings",
    "get_settings",
    "DatabaseConfig",
    "get_database",
    "SoilSchema",
    "SoilDataModel",
    "WeatherSchema",
    "WeatherDataModel",
    "CropSchema",
    "CropDataModel",
]

__version__ = "2.0.0"
