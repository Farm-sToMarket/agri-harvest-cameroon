"""Climate data processing: aggregation and Parquet export."""

from data.processing.aggregation import (
    compute_chirps_daily_stats,
    daily_to_monthly,
    extract_point_timeseries,
    grid_to_zonal_means,
)
from data.processing.export import (
    dataset_to_dataframe,
    save_parquet,
    export_terraclimate_parquet,
    export_chirps_parquet,
)

__all__ = [
    "compute_chirps_daily_stats",
    "daily_to_monthly",
    "extract_point_timeseries",
    "grid_to_zonal_means",
    "dataset_to_dataframe",
    "save_parquet",
    "export_terraclimate_parquet",
    "export_chirps_parquet",
]
