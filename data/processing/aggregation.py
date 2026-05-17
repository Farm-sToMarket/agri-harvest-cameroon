"""Temporal and spatial aggregation of climate grids.

Includes:
- daily_to_monthly: resample daily -> monthly (sum for precip, mean for rest)
- compute_chirps_daily_stats: heavy_rain_days, wet_days, max_dry_spell,
  rain_intensity_max, precipitation_std per month from daily CHIRPS data
- extract_point_timeseries: single point extraction
- grid_to_zonal_means: spatial mean per agroecological zone bounding box
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Daily -> Monthly aggregation
# ---------------------------------------------------------------------------

def daily_to_monthly(
    ds: xr.Dataset,
    precip_var: str = "precipitation_daily",
) -> xr.Dataset:
    """Resample daily data to monthly.

    Precipitation is summed; all other variables are averaged.

    Args:
        ds: Daily-resolution xarray Dataset.
        precip_var: Name of the precipitation variable to sum.

    Returns:
        Monthly-resolution Dataset.
    """
    sum_vars = [v for v in ds.data_vars if v == precip_var]
    mean_vars = [v for v in ds.data_vars if v != precip_var]

    parts: list[xr.Dataset] = []
    if sum_vars:
        parts.append(ds[sum_vars].resample(time="MS").sum())
    if mean_vars:
        parts.append(ds[mean_vars].resample(time="MS").mean())

    monthly = xr.merge(parts)
    logger.info("Aggregated daily -> monthly: %s", dict(monthly.dims))
    return monthly


# ---------------------------------------------------------------------------
# CHIRPS daily precipitation statistics
# ---------------------------------------------------------------------------

def _max_dry_spell_1d(precip: np.ndarray, threshold: float) -> int:
    """Longest consecutive run of dry days in a 1-D array."""
    dry = precip < threshold
    if not dry.any():
        return 0
    max_run = 0
    current = 0
    for d in dry:
        if d:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def compute_chirps_daily_stats(
    ds: xr.Dataset,
    precip_var: str = "precipitation_daily",
    heavy_threshold: float = 20.0,
    wet_threshold: float = 1.0,
) -> xr.Dataset:
    """Compute monthly precipitation statistics from daily CHIRPS data.

    For each month computes:
    - heavy_rain_days: days with precip > heavy_threshold
    - wet_days: days with precip >= wet_threshold
    - max_dry_spell: longest consecutive run with precip < wet_threshold
    - rain_intensity_max: maximum daily precipitation
    - precipitation_std: standard deviation of daily precipitation

    Args:
        ds: Daily xarray Dataset with a precipitation variable.
        precip_var: Name of the daily precipitation variable.
        heavy_threshold: Threshold in mm for heavy rain day (default 20).
        wet_threshold: Threshold in mm for wet day (default 1).

    Returns:
        Monthly Dataset with the original monthly precip sum plus the
        five statistical columns.
    """
    precip = ds[precip_var]
    resampler = precip.resample(time="MS")

    # Heavy rain days: count of days > threshold
    heavy_rain_days = (precip > heavy_threshold).resample(time="MS").sum()
    heavy_rain_days.name = "heavy_rain_days"

    # Wet days: count of days >= threshold
    wet_days = (precip >= wet_threshold).resample(time="MS").sum()
    wet_days.name = "wet_days"

    # Max daily intensity per month
    rain_intensity_max = resampler.max()
    rain_intensity_max.name = "rain_intensity_max"

    # Std of daily precip per month
    precipitation_std = resampler.std()
    precipitation_std.name = "precipitation_std"

    # Monthly sum (total precipitation)
    precip_monthly_sum = resampler.sum()
    precip_monthly_sum.name = "precipitation_daily"  # keep name for pipeline compat

    # Max dry spell - requires apply_ufunc because it's a running-count metric
    lat_dim = _detect_spatial_dim(ds, ["lat", "latitude"])
    lon_dim = _detect_spatial_dim(ds, ["lon", "longitude"])

    def _dry_spell_for_group(group):
        """Compute max dry spell for a single month chunk."""
        arr = group.values
        # arr shape: (time, lat, lon) or similar
        result = np.zeros(arr.shape[1:], dtype=np.int32)
        it = np.nditer(result, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            ts_slice = arr[(slice(None),) + idx]
            valid = ts_slice[~np.isnan(ts_slice)]
            result[idx] = _max_dry_spell_1d(valid, wet_threshold) if len(valid) > 0 else 0
            it.iternext()
        coords = {k: v for k, v in group.coords.items() if k != "time"}
        dims = [d for d in group.dims if d != "time"]
        return xr.DataArray(result, dims=dims, coords=coords)

    dry_spell_parts = []
    for label, group in precip.resample(time="MS"):
        ds_part = _dry_spell_for_group(group)
        ds_part = ds_part.expand_dims(time=[label])
        dry_spell_parts.append(ds_part)

    max_dry_spell = xr.concat(dry_spell_parts, dim="time")
    max_dry_spell.name = "max_dry_spell"

    stats = xr.merge([
        precip_monthly_sum,
        heavy_rain_days,
        wet_days,
        max_dry_spell,
        rain_intensity_max,
        precipitation_std,
    ])

    logger.info(
        "Computed CHIRPS daily stats: %s, shape=%s",
        list(stats.data_vars), dict(stats.dims),
    )
    return stats


# ---------------------------------------------------------------------------
# Point / zonal extraction
# ---------------------------------------------------------------------------

def extract_point_timeseries(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    method: str = "nearest",
) -> pd.DataFrame:
    """Extract a time series for a single lat/lon point.

    Args:
        ds: xarray Dataset with spatial and time dimensions.
        lat: Target latitude.
        lon: Target longitude.
        method: Interpolation method (default "nearest").

    Returns:
        pandas DataFrame indexed by time.
    """
    lat_dim = _detect_spatial_dim(ds, ["lat", "latitude"])
    lon_dim = _detect_spatial_dim(ds, ["lon", "longitude"])

    point = ds.sel({lat_dim: lat, lon_dim: lon}, method=method)
    df = point.to_dataframe().reset_index()
    logger.info("Extracted point timeseries at (%.2f, %.2f): %d rows", lat, lon, len(df))
    return df


def grid_to_zonal_means(
    ds: xr.Dataset,
    zones: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Compute spatial mean for each agroecological zone.

    Args:
        ds: xarray Dataset.
        zones: Dict mapping zone name to {"lat_min", "lat_max", "lon_min", "lon_max"}.

    Returns:
        DataFrame with zone, time, and mean variable columns.
    """
    lat_dim = _detect_spatial_dim(ds, ["lat", "latitude"])
    lon_dim = _detect_spatial_dim(ds, ["lon", "longitude"])

    frames: list[pd.DataFrame] = []
    for zone_name, bbox in zones.items():
        subset = ds.sel(
            {lat_dim: slice(bbox["lat_min"], bbox["lat_max"]),
             lon_dim: slice(bbox["lon_min"], bbox["lon_max"])},
        )
        zonal_mean = subset.mean(dim=[lat_dim, lon_dim])
        df = zonal_mean.to_dataframe().reset_index()
        df["agroecological_zone"] = zone_name
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    logger.info("Computed zonal means for %d zones", len(zones))
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_spatial_dim(ds: xr.Dataset, candidates: list[str]) -> str:
    """Find the first matching dimension name."""
    all_names = set(ds.dims) | set(ds.coords)
    for name in candidates:
        if name in all_names:
            return name
    raise ValueError(
        f"Cannot find spatial dimension among {candidates} in {list(ds.dims)}"
    )
