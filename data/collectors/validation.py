"""Validation of collected climate datasets against physical bounds."""

import logging
from typing import Any

import numpy as np
import xarray as xr

from utils.constants import VALIDATION_RANGES

logger = logging.getLogger(__name__)

# Physical bounds per variable
# Covers all TerraClimate direct, derived, and CHIRPS-derived variables.
CLIMATE_BOUNDS: dict[str, dict[str, float]] = {
    # ── Direct TerraClimate variables ──
    "temperature_min": {"min": -5, "max": 50},
    "temperature_max": {"min": -5, "max": 50},
    "precipitation_monthly": {"min": 0, "max": 2000},
    "vapor_pressure": {"min": 0, "max": 10},
    "vapor_pressure_deficit": {"min": 0, "max": 15},
    "climate_water_deficit": {"min": 0, "max": 500},
    "soil_moisture": {"min": 0, "max": 1000},
    "actual_evapotranspiration": {"min": 0, "max": 500},
    "potential_evapotranspiration": {"min": 0, "max": 500},
    "wind_speed": {"min": 0, "max": 30},
    "solar_radiation_raw": {"min": 0, "max": 500},

    # ── Basic derived variables ──
    "temperature_mean": {"min": -5, "max": 50},
    "relative_humidity": {"min": 0, "max": 100},
    "solar_radiation": {"min": 0, "max": 40},
    "diurnal_range": {"min": 0, "max": 35},

    # ── Water-balance derived ──
    "aridity_index": {"min": 0, "max": 30},
    "water_balance": {"min": -300, "max": 1500},
    "crop_water_stress": {"min": 0, "max": 1},

    # ── Thermal indices ──
    "gdd_base10": {"min": 0, "max": 40},
    "gdd_base15": {"min": 0, "max": 35},

    # ── CHIRPS daily / aggregated ──
    "precipitation_daily": {"min": 0, "max": 500},
    "heavy_rain_days": {"min": 0, "max": 31},
    "wet_days": {"min": 0, "max": 31},
    "max_dry_spell": {"min": 0, "max": 31},
    "rain_intensity_max": {"min": 0, "max": 500},
    "precipitation_std": {"min": 0, "max": 200},
}


def validate_dataset(
    ds: xr.Dataset,
    bounds: dict[str, dict[str, float]] | None = None,
    clamp: bool = True,
) -> dict[str, Any]:
    """Validate dataset values against physical bounds.

    Args:
        ds: xarray Dataset to validate.
        bounds: Per-variable min/max bounds.  Defaults to :data:`CLIMATE_BOUNDS`.
        clamp: If True, clamp out-of-range values instead of just reporting.

    Returns:
        Report dict with per-variable violation counts.
    """
    if bounds is None:
        bounds = CLIMATE_BOUNDS

    report: dict[str, Any] = {}
    for var in ds.data_vars:
        var_name = str(var)
        if var_name not in bounds:
            continue
        lo = bounds[var_name]["min"]
        hi = bounds[var_name]["max"]
        arr = ds[var_name]
        below = int((arr < lo).sum().values)
        above = int((arr > hi).sum().values)
        total = int(arr.notnull().sum().values)

        report[var_name] = {
            "total_valid": total,
            "below_min": below,
            "above_max": above,
        }

        if below or above:
            logger.warning(
                "Variable %s: %d below %.1f, %d above %.1f (total valid: %d)",
                var_name, below, lo, above, hi, total,
            )
            if clamp:
                ds[var_name] = arr.clip(lo, hi)
                logger.info("Clamped %s to [%.1f, %.1f]", var_name, lo, hi)

    return report


def validate_spatial_coverage(
    ds: xr.Dataset,
    min_coverage_pct: float = 80.0,
) -> bool:
    """Check that enough grid cells contain data.

    Args:
        ds: xarray Dataset.
        min_coverage_pct: Minimum percentage of non-NaN cells required.

    Returns:
        True if coverage meets the threshold.
    """
    for var in ds.data_vars:
        arr = ds[str(var)]
        total_cells = int(arr.size)
        valid_cells = int(arr.notnull().sum().values)
        if total_cells == 0:
            continue
        coverage = 100.0 * valid_cells / total_cells
        if coverage < min_coverage_pct:
            logger.warning(
                "Variable %s spatial coverage %.1f%% < %.1f%% threshold",
                var, coverage, min_coverage_pct,
            )
            return False
    return True
