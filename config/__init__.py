"""
Configuration module for Cameroon Agricultural Data Management System
"""

from .settings import Settings, get_settings
from .schema import (
    SoilDataModel,
    WeatherDataModel,
    CropDataModel,
)

__all__ = [
    "Settings",
    "get_settings",
    "SoilDataModel",
    "WeatherDataModel",
    "CropDataModel",
]

__version__ = "2.0.0"
