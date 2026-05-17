"""Export climate xarray datasets to Parquet via pandas."""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from data.processing.aggregation import compute_chirps_daily_stats, daily_to_monthly
from utils.date_utils import get_agricultural_season
from utils.file_utils import ensure_directory_exists
from utils.geospatial_utils import determine_agroecological_zone

logger = logging.getLogger(__name__)


def dataset_to_dataframe(
    ds: xr.Dataset,
    add_zone: bool = True,
    add_season: bool = True,
) -> pd.DataFrame:
    """Flatten an xarray Dataset into a pandas DataFrame.

    Steps:
        1. Convert to DataFrame and reset index.
        2. Rename lat/lon -> latitude/longitude.
        3. Drop rows where all climate variables are NaN.
        4. Optionally add agroecological_zone and season columns.

    Args:
        ds: xarray Dataset (gridded climate data).
        add_zone: Assign agroecological_zone from lat/lon.
        add_season: Assign season from time and latitude.

    Returns:
        Flat DataFrame suitable for Parquet export.
    """
    df = ds.to_dataframe().reset_index()

    # Normalise coordinate names
    rename = {}
    if "lat" in df.columns:
        rename["lat"] = "latitude"
    if "lon" in df.columns:
        rename["lon"] = "longitude"
    if rename:
        df = df.rename(columns=rename)

    # Identify climate-value columns (everything except coords)
    coord_cols = {"latitude", "longitude", "time"}
    value_cols = [c for c in df.columns if c not in coord_cols]
    df = df.dropna(subset=value_cols, how="all")

    if add_zone and "latitude" in df.columns and "longitude" in df.columns:
        df["agroecological_zone"] = df.apply(
            lambda r: determine_agroecological_zone(
                r["latitude"], r["longitude"], elevation=0.0,
            ),
            axis=1,
        )

    if add_season and "time" in df.columns and "latitude" in df.columns:
        df["season"] = df.apply(
            lambda r: get_agricultural_season(
                r["time"].date() if hasattr(r["time"], "date") else r["time"],
                latitude=r["latitude"],
            ),
            axis=1,
        )

    logger.info("DataFrame shape: %s, columns: %s", df.shape, list(df.columns))
    return df


def save_parquet(
    df: pd.DataFrame,
    path: Path | str,
    compression: str = "zstd",
) -> Path:
    """Save a DataFrame as a Parquet file.

    Args:
        df: pandas DataFrame.
        path: Output file path.
        compression: Parquet compression codec.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    ensure_directory_exists(path.parent)
    df.to_parquet(path, engine="pyarrow", compression=compression, index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Saved Parquet %s (%.1f MB, %d rows)", path, size_mb, len(df))
    return path


def export_terraclimate_parquet(
    nc_paths: list[Path],
    output_dir: Path | str,
) -> Path:
    """Concatenate annual TerraClimate NetCDFs and export as a single Parquet.

    The output contains all 11 direct TerraClimate variables (after scale
    factors) plus all derived variables (temperature_mean, diurnal_range,
    relative_humidity, solar_radiation, aridity_index, water_balance,
    crop_water_stress, gdd_base10, gdd_base15).

    Args:
        nc_paths: List of annual NetCDF file paths.
        output_dir: Directory for the output Parquet.

    Returns:
        Path to the written Parquet file.
    """
    output_dir = Path(output_dir)
    datasets = [xr.open_dataset(p) for p in sorted(nc_paths)]
    combined = xr.concat(datasets, dim="time")
    df = dataset_to_dataframe(combined)

    years = sorted({t.year for t in combined.time.values.astype("datetime64[us]").astype("object")})
    filename = f"terraclimate_cameroon_{years[0]}_{years[-1]}.parquet"
    out_path = save_parquet(df, output_dir / filename)

    for ds in datasets:
        ds.close()

    return out_path


def export_chirps_parquet(
    nc_paths: list[Path],
    output_dir: Path | str,
    aggregate_monthly: bool = False,
    compute_daily_stats: bool = True,
) -> Path:
    """Concatenate annual CHIRPS NetCDFs and export as Parquet.

    When ``compute_daily_stats`` is True (default), the output includes
    monthly precipitation statistics derived from daily data:
    heavy_rain_days, wet_days, max_dry_spell, rain_intensity_max,
    precipitation_std.

    Args:
        nc_paths: List of annual NetCDF file paths.
        output_dir: Directory for the output Parquet.
        aggregate_monthly: If True, aggregate daily data to monthly sums.
        compute_daily_stats: If True, compute monthly precipitation statistics
            from the daily data before export. Implies monthly output.

    Returns:
        Path to the written Parquet file.
    """
    output_dir = Path(output_dir)
    datasets = [xr.open_dataset(p) for p in sorted(nc_paths)]
    combined = xr.concat(datasets, dim="time")

    if compute_daily_stats:
        combined = compute_chirps_daily_stats(combined)
    elif aggregate_monthly:
        combined = daily_to_monthly(combined)

    df = dataset_to_dataframe(combined)

    years = sorted({t.year for t in combined.time.values.astype("datetime64[us]").astype("object")})
    if compute_daily_stats or aggregate_monthly:
        freq_tag = "monthly_stats"
    else:
        freq_tag = "daily"
    filename = f"chirps_cameroon_{freq_tag}_{years[0]}_{years[-1]}.parquet"
    out_path = save_parquet(df, output_dir / filename)

    for ds in datasets:
        ds.close()

    return out_path
