"""CHIRPS daily precipitation collector (~5 km, NetCDF via HTTP)."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from data.collectors.base import BaseClimateCollector, NetworkError
from utils.file_utils import ensure_directory_exists

logger = logging.getLogger(__name__)


class CHIRPSCollector(BaseClimateCollector):
    """Collect daily precipitation grids from CHIRPS-2.0.

    Each year is a single NetCDF file downloaded over HTTP and opened with
    xarray.  Data is subset to Cameroon, fill values replaced by NaN,
    negatives clipped, and saved as annual NetCDF files.
    """

    def __init__(self) -> None:
        super().__init__()
        self.chirps_config = self.config["chirps"]
        self.base_url = self.chirps_config["base_url"]
        self.file_pattern = self.chirps_config["file_pattern"]
        self.fill_value = self.chirps_config["variable"]["fill_value"]
        self.output_dir = Path(self.config["output"]["chirps_dir"])
        ensure_directory_exists(self.output_dir)

    def _build_url(self, year: int) -> str:
        """Construct the download URL for a given year."""
        filename = self.file_pattern.format(year=year)
        return f"{self.base_url}/{filename}"

    def collect_year(self, year: int) -> xr.Dataset:
        """Download and pre-process one year of CHIRPS data.

        Args:
            year: Calendar year to collect.

        Returns:
            xarray Dataset with variable ``precipitation_daily``.
        """
        url = self._build_url(year)
        ds = self._open_opendap_with_retry(url)
        ds = self._subset_spatial(ds)

        src_name = self.chirps_config["variable"]["source_name"]
        proj_name = self.chirps_config["variable"]["project_name"]

        # Replace fill values and clip negatives
        ds[src_name] = ds[src_name].where(ds[src_name] != self.fill_value)
        ds[src_name] = ds[src_name].clip(min=0)

        # Rename to project convention
        ds = ds.rename({src_name: proj_name})

        ds = ds.load()
        logger.info("Collected CHIRPS year=%d shape=%s", year, dict(ds.dims))
        return ds

    def collect(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[Path]:
        """Collect CHIRPS daily precipitation for the requested years.

        Args:
            start_year: Override start year (default from config).
            end_year: Override end year (default from config).

        Returns:
            List of paths to saved annual NetCDF files.
        """
        start_year = start_year or self.start_year
        end_year = end_year or self.end_year

        output_paths: list[Path] = []
        for year in range(start_year, end_year + 1):
            out_path = self.output_dir / f"chirps_cameroon_{year}.nc"

            # Cache check: skip if file already exists
            if out_path.exists():
                logger.info("Cache hit, skipping: %s", out_path)
                output_paths.append(out_path)
                continue

            try:
                ds = self.collect_year(year)
                self._save_netcdf(ds, out_path)
                output_paths.append(out_path)
            except (NetworkError, Exception) as exc:
                logger.error("Failed to collect CHIRPS year=%d: %s", year, exc)
                continue

        logger.info(
            "CHIRPS collection complete: %d files written", len(output_paths),
        )
        return output_paths
