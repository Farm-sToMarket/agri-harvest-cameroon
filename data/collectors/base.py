"""Abstract base class for climate data collectors."""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import xarray as xr

from config.yaml_loader import load_climate_sources
from utils.constants import CAMEROON_BOUNDS
from utils.file_utils import ensure_directory_exists

logger = logging.getLogger(__name__)


class ClimateCollectorError(Exception):
    """Base exception for climate data collection errors."""


class NetworkError(ClimateCollectorError):
    """Raised when a network request fails after all retries."""


class ValidationError(ClimateCollectorError):
    """Raised when collected data fails validation."""


class BaseClimateCollector(ABC):
    """Abstract base class for climate data collectors.

    Loads configuration from climate_sources.yaml, provides spatial/temporal
    subsetting, OpenDAP retry logic, and intermediate NetCDF saving.
    """

    def __init__(self) -> None:
        self.config = load_climate_sources()
        bounds = self.config["spatial"]
        buffer = bounds["buffer_deg"]
        self.lat_min = bounds["lat_min"] - buffer
        self.lat_max = bounds["lat_max"] + buffer
        self.lon_min = bounds["lon_min"] - buffer
        self.lon_max = bounds["lon_max"] + buffer

        net = self.config["network"]
        self.max_retries = net["max_retries"]
        self.retry_delay = net["retry_delay_seconds"]
        self.timeout = net["timeout_seconds"]

        self.start_year = self.config["temporal"]["start_year"]
        self.end_year = self.config["temporal"]["end_year"]

    def _open_opendap_with_retry(self, url: str) -> xr.Dataset:
        """Open an OpenDAP dataset with exponential-backoff retry.

        Args:
            url: OpenDAP endpoint URL.

        Returns:
            xarray Dataset.

        Raises:
            NetworkError: If all retries are exhausted.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("Opening %s (attempt %d/%d)", url, attempt, self.max_retries)
                ds = xr.open_dataset(url, engine="netcdf4")
                return ds
            except Exception as exc:
                last_error = exc
                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d failed for %s: %s. Retrying in %ds...",
                    attempt, url, exc, delay,
                )
                time.sleep(delay)
        raise NetworkError(
            f"Failed to open {url} after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _detect_dim(ds: xr.Dataset, candidates: list[str]) -> str:
        """Detect dimension name from a list of candidates.

        Args:
            ds: xarray Dataset.
            candidates: Possible dimension names (e.g. ["lat", "latitude"]).

        Returns:
            The first matching dimension name found.

        Raises:
            ClimateCollectorError: If none of the candidates exist.
        """
        all_names = set(ds.dims) | set(ds.coords)
        for name in candidates:
            if name in all_names:
                return name
        raise ClimateCollectorError(
            f"Cannot find dimension among {candidates} in dataset with "
            f"dims={list(ds.dims)} coords={list(ds.coords)}"
        )

    def _subset_spatial(self, ds: xr.Dataset) -> xr.Dataset:
        """Subset dataset to Cameroon bounding box.

        Handles both ascending and descending latitude, and variable naming
        conventions (lat/lon vs latitude/longitude).
        """
        lat_dim = self._detect_dim(ds, ["lat", "latitude"])
        lon_dim = self._detect_dim(ds, ["lon", "longitude"])

        lat_vals = ds[lat_dim].values
        lat_ascending = lat_vals[0] < lat_vals[-1] if len(lat_vals) > 1 else True

        if lat_ascending:
            ds = ds.sel(
                {lat_dim: slice(self.lat_min, self.lat_max),
                 lon_dim: slice(self.lon_min, self.lon_max)},
            )
        else:
            ds = ds.sel(
                {lat_dim: slice(self.lat_max, self.lat_min),
                 lon_dim: slice(self.lon_min, self.lon_max)},
            )
        return ds

    def _subset_temporal(
        self, ds: xr.Dataset, start: str, end: str,
    ) -> xr.Dataset:
        """Subset dataset to a time range.

        Args:
            ds: xarray Dataset with a 'time' coordinate.
            start: Start date string (e.g. "2010-01-01").
            end: End date string (e.g. "2024-12-31").
        """
        time_dim = self._detect_dim(ds, ["time", "t"])
        return ds.sel({time_dim: slice(start, end)})

    def _save_netcdf(self, ds: xr.Dataset, path: Path) -> Path:
        """Save dataset to NetCDF with directory creation and size logging."""
        ensure_directory_exists(path.parent)
        ds.to_netcdf(path)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Saved %s (%.1f MB)", path, size_mb)
        return path

    @abstractmethod
    def collect(self, start_year: int, end_year: int, **kwargs) -> list[Path]:
        """Collect climate data for the given year range.

        Returns:
            List of output file paths.
        """
