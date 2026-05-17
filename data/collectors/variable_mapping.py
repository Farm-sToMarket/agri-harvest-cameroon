"""Variable mapping, scale factors, and derived-variable computation.

Handles:
- TerraClimate raw -> project name mapping with scale factors
- Basic derivations: temperature_mean, relative_humidity, solar_radiation, diurnal_range
- Water-balance derivations: aridity_index, water_balance, crop_water_stress
- Thermal indices: gdd_base10, gdd_base15
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from config.yaml_loader import load_climate_sources

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariableSpec:
    """Specification for a single climate variable."""

    source_name: str
    project_name: str
    scale_factor: float
    units: str


def get_terraclimate_specs() -> dict[str, VariableSpec]:
    """Build the TerraClimate variable mapping from YAML config.

    Returns:
        Dict mapping source variable name to VariableSpec.
    """
    cfg = load_climate_sources()["terraclimate"]["variables"]
    specs: dict[str, VariableSpec] = {}
    for key, info in cfg.items():
        specs[info["source_name"]] = VariableSpec(
            source_name=info["source_name"],
            project_name=info["project_name"],
            scale_factor=info["scale_factor"],
            units=info["units"],
        )
    return specs


def apply_scale_factors(ds: xr.Dataset, specs: dict[str, VariableSpec]) -> xr.Dataset:
    """Apply scale factors and rename variables to project names.

    Args:
        ds: Raw xarray Dataset from TerraClimate.
        specs: Variable specifications from :func:`get_terraclimate_specs`.

    Returns:
        Dataset with scaled values and renamed variables.
    """
    rename_map: dict[str, str] = {}
    for src_name, spec in specs.items():
        if src_name in ds.data_vars:
            if spec.scale_factor != 1.0:
                ds[src_name] = ds[src_name] * spec.scale_factor
            rename_map[src_name] = spec.project_name
    ds = ds.rename(rename_map)
    return ds


# ---------------------------------------------------------------------------
# Basic derivations
# ---------------------------------------------------------------------------

def derive_temperature_mean(ds: xr.Dataset) -> xr.Dataset:
    """Derive temperature_mean = (temperature_min + temperature_max) / 2."""
    if "temperature_min" in ds and "temperature_max" in ds:
        ds["temperature_mean"] = (ds["temperature_min"] + ds["temperature_max"]) / 2.0
        ds["temperature_mean"].attrs["units"] = "degC"
        logger.info("Derived temperature_mean")
    return ds


def derive_diurnal_range(ds: xr.Dataset) -> xr.Dataset:
    """Derive diurnal_range = temperature_max - temperature_min."""
    if "temperature_min" in ds and "temperature_max" in ds:
        ds["diurnal_range"] = ds["temperature_max"] - ds["temperature_min"]
        ds["diurnal_range"].attrs["units"] = "degC"
        logger.info("Derived diurnal_range")
    return ds


def derive_relative_humidity(ds: xr.Dataset) -> xr.Dataset:
    """Derive relative humidity from vapor pressure and temperature_mean.

    RH = 100 * vap / (0.6108 * exp(17.27 * T / (T + 237.3)))
    where T is temperature_mean in degC and vap is vapor_pressure in kPa.
    """
    if "vapor_pressure" in ds and "temperature_mean" in ds:
        t = ds["temperature_mean"]
        vap = ds["vapor_pressure"]
        sat_vp = 0.6108 * np.exp(17.27 * t / (t + 237.3))
        rh = 100.0 * vap / sat_vp
        ds["relative_humidity"] = rh.clip(0, 100)
        ds["relative_humidity"].attrs["units"] = "%"
        logger.info("Derived relative_humidity")
    return ds


def convert_srad_to_mj(ds: xr.Dataset) -> xr.Dataset:
    """Convert solar_radiation_raw from W/m2 to MJ/m2/day.

    MJ/m2/day = W/m2 * 0.0864
    """
    if "solar_radiation_raw" in ds:
        ds["solar_radiation"] = ds["solar_radiation_raw"] * 0.0864
        ds["solar_radiation"].attrs["units"] = "MJ/m2/day"
        ds = ds.drop_vars("solar_radiation_raw")
        logger.info("Converted solar_radiation to MJ/m2/day")
    return ds


# ---------------------------------------------------------------------------
# Water-balance & drought derivations (uses AET, PET, soil, deficit)
# ---------------------------------------------------------------------------

def derive_aridity_index(ds: xr.Dataset) -> xr.Dataset:
    """Derive aridity_index = precipitation_monthly / PET.

    UNEP classification: <0.2 arid, 0.2-0.5 semi-arid, >0.65 humid.
    Division by zero protected: where PET==0, aridity_index is NaN.
    """
    if "precipitation_monthly" in ds and "potential_evapotranspiration" in ds:
        pet = ds["potential_evapotranspiration"]
        ds["aridity_index"] = ds["precipitation_monthly"] / pet.where(pet > 0)
        ds["aridity_index"].attrs["units"] = "dimensionless"
        logger.info("Derived aridity_index (P/PET)")
    return ds


def derive_water_balance(ds: xr.Dataset) -> xr.Dataset:
    """Derive water_balance = precipitation_monthly - AET.

    Positive = moisture surplus, negative = deficit.
    """
    if "precipitation_monthly" in ds and "actual_evapotranspiration" in ds:
        ds["water_balance"] = (
            ds["precipitation_monthly"] - ds["actual_evapotranspiration"]
        )
        ds["water_balance"].attrs["units"] = "mm"
        logger.info("Derived water_balance (P - AET)")
    return ds


def derive_crop_water_stress(ds: xr.Dataset) -> xr.Dataset:
    """Derive crop_water_stress = 1 - AET / PET.

    0 = no stress (AET meets full demand), 1 = maximum stress (no ET).
    Clamped to [0, 1].
    """
    if "actual_evapotranspiration" in ds and "potential_evapotranspiration" in ds:
        pet = ds["potential_evapotranspiration"]
        aet = ds["actual_evapotranspiration"]
        stress = 1.0 - aet / pet.where(pet > 0)
        ds["crop_water_stress"] = stress.clip(0, 1)
        ds["crop_water_stress"].attrs["units"] = "dimensionless"
        logger.info("Derived crop_water_stress (1 - AET/PET)")
    return ds


# ---------------------------------------------------------------------------
# Thermal indices (growing degree-days)
# ---------------------------------------------------------------------------

def derive_gdd(ds: xr.Dataset) -> xr.Dataset:
    """Derive growing degree-days at base 10 C and 15 C."""
    if "temperature_mean" in ds:
        t = ds["temperature_mean"]
        ds["gdd_base10"] = (t - 10.0).clip(min=0)
        ds["gdd_base10"].attrs["units"] = "degC-day"
        ds["gdd_base15"] = (t - 15.0).clip(min=0)
        ds["gdd_base15"].attrs["units"] = "degC-day"
        logger.info("Derived gdd_base10 and gdd_base15")
    return ds


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def apply_all_derivations(ds: xr.Dataset) -> xr.Dataset:
    """Chain all derived-variable computations.

    Order matters: temperature_mean must exist before RH, aridity_index, GDD.
    """
    # Basic
    ds = derive_temperature_mean(ds)
    ds = derive_diurnal_range(ds)
    ds = derive_relative_humidity(ds)
    ds = convert_srad_to_mj(ds)
    # Water-balance
    ds = derive_aridity_index(ds)
    ds = derive_water_balance(ds)
    ds = derive_crop_water_stress(ds)
    # Thermal indices
    ds = derive_gdd(ds)
    return ds
