"""TerraClimate data collector (monthly ~4 km via OpenDAP/THREDDS)."""

import logging
from pathlib import Path

import xarray as xr

from data.collectors.base import BaseClimateCollector, NetworkError
from data.collectors.variable_mapping import (
    apply_all_derivations,
    apply_scale_factors,
    get_terraclimate_specs,
)
from utils.file_utils import ensure_directory_exists

logger = logging.getLogger(__name__)


class TerraClimateCollector(BaseClimateCollector):
    """Collect monthly climate grids from TerraClimate THREDDS.

    Each variable is served as a separate OpenDAP aggregation endpoint.
    Data is spatially/temporally subset, merged, scale-factor-adjusted,
    and written to annual NetCDF files.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tc_config = self.config["terraclimate"]
        self.base_url = self.tc_config["base_url"]
        self.output_dir = Path(self.config["output"]["terraclimate_dir"])
        ensure_directory_exists(self.output_dir)

    def _build_url(self, variable: str) -> str:
        """Construct the THREDDS OpenDAP URL for a variable.

        Example:
            https://thredds.northwestknowledge.net/thredds/dodsC/agg_terraclimate_tmin_1958_CurrentYear_GLOBE.nc
        """
        return (
            f"{self.base_url}_{variable}_1958_CurrentYear_GLOBE.nc"
        )

    def collect_variable(
        self,
        variable: str,
        start_year: int,
        end_year: int,
    ) -> xr.Dataset:
        """Download a single TerraClimate variable, subset to Cameroon.

        Args:
            variable: TerraClimate variable name (e.g. "tmin").
            start_year: First year to collect.
            end_year: Last year to collect (inclusive).

        Returns:
            xarray Dataset containing the variable for the requested period.
        """
        url = self._build_url(variable)
        ds = self._open_opendap_with_retry(url)
        ds = self._subset_spatial(ds)
        ds = self._subset_temporal(
            ds,
            f"{start_year}-01-01",
            f"{end_year}-12-31",
        )
        # Load into memory to avoid holding the remote connection
        ds = ds.load()
        logger.info(
            "Collected variable=%s shape=%s", variable, dict(ds.dims),
        )
        return ds

    def collect(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
        variables: list[str] | None = None,
    ) -> list[Path]:
        """Collect all requested TerraClimate variables.

        Args:
            start_year: Override start year (default from config).
            end_year: Override end year (default from config).
            variables: Subset of TerraClimate variable names to collect.
                       Defaults to all 11 configured variables.

        Returns:
            List of paths to the saved annual NetCDF files.
        """
        start_year = start_year or self.start_year
        end_year = end_year or self.end_year

        if variables is None:
            variables = list(self.tc_config["variables"].keys())

        specs = get_terraclimate_specs()

        # Collect each variable individually
        datasets: list[xr.Dataset] = []
        for var in variables:
            src_name = self.tc_config["variables"][var]["source_name"]
            try:
                ds = self.collect_variable(src_name, start_year, end_year)
                datasets.append(ds)
            except NetworkError:
                logger.error("Skipping variable %s due to network error", var)

        if not datasets:
            logger.error("No variables collected; aborting")
            return []

        merged = xr.merge(datasets)
        merged = apply_scale_factors(merged, specs)
        merged = apply_all_derivations(merged)

        # Split into annual files
        output_paths: list[Path] = []
        for year in range(start_year, end_year + 1):
            yearly = merged.sel(time=str(year))
            if yearly.time.size == 0:
                continue
            out_path = self.output_dir / f"terraclimate_cameroon_{year}.nc"
            self._save_netcdf(yearly, out_path)
            output_paths.append(out_path)

        logger.info(
            "TerraClimate collection complete: %d files written", len(output_paths),
        )
        return output_paths
