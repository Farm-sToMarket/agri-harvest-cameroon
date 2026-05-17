"""CLI entry point for climate data collection.

Usage examples::

    python -m scripts.collect_climate --source all
    python -m scripts.collect_climate --source terraclimate --start-year 2015
    python -m scripts.collect_climate --source chirps --chirps-monthly --no-daily-stats
    python -m scripts.collect_climate --source terraclimate --variables tmin tmax ppt
"""

import argparse
import logging
import sys
from pathlib import Path

from config.yaml_loader import load_climate_sources


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = load_climate_sources()

    parser = argparse.ArgumentParser(
        description="Collect real climate data from TerraClimate and CHIRPS.",
    )
    parser.add_argument(
        "--source",
        choices=["terraclimate", "chirps", "all"],
        default="all",
        help="Data source to collect (default: all).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=cfg["temporal"]["start_year"],
        help=f"Start year (default: {cfg['temporal']['start_year']}).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=cfg["temporal"]["end_year"],
        help=f"End year (default: {cfg['temporal']['end_year']}).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="TerraClimate variables to collect (default: all 11).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Keep intermediate NetCDF files only; skip Parquet export.",
    )
    parser.add_argument(
        "--chirps-monthly",
        action="store_true",
        help="Aggregate CHIRPS daily data to monthly sums (simple, no stats).",
    )
    parser.add_argument(
        "--no-daily-stats",
        action="store_true",
        help="Skip computing CHIRPS daily precipitation statistics "
             "(heavy_rain_days, wet_days, max_dry_spell, etc.).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    args = _parse_args(argv)
    logger = logging.getLogger(__name__)

    cfg = load_climate_sources()
    merged_dir = Path(cfg["output"]["merged_dir"])

    # ── TerraClimate ─────────────────────────────────────────────────────────
    tc_paths: list[Path] = []
    if args.source in ("terraclimate", "all"):
        from data.collectors.terraclimate import TerraClimateCollector
        from data.collectors.validation import validate_dataset, validate_spatial_coverage

        logger.info("=== Collecting TerraClimate data ===")
        collector = TerraClimateCollector()
        tc_paths = collector.collect(
            start_year=args.start_year,
            end_year=args.end_year,
            variables=args.variables,
        )

        import xarray as xr
        for p in tc_paths:
            ds = xr.open_dataset(p)
            validate_dataset(ds, clamp=True)
            validate_spatial_coverage(ds)
            ds.close()

        if tc_paths and not args.skip_export:
            from data.processing.export import export_terraclimate_parquet
            pq = export_terraclimate_parquet(tc_paths, merged_dir)
            logger.info("TerraClimate Parquet: %s", pq)

    # ── CHIRPS ───────────────────────────────────────────────────────────────
    chirps_paths: list[Path] = []
    if args.source in ("chirps", "all"):
        from data.collectors.chirps import CHIRPSCollector
        from data.collectors.validation import validate_dataset, validate_spatial_coverage

        logger.info("=== Collecting CHIRPS data ===")
        collector = CHIRPSCollector()
        chirps_paths = collector.collect(
            start_year=args.start_year,
            end_year=args.end_year,
        )

        import xarray as xr
        for p in chirps_paths:
            ds = xr.open_dataset(p)
            validate_dataset(ds, clamp=True)
            validate_spatial_coverage(ds)
            ds.close()

        if chirps_paths and not args.skip_export:
            from data.processing.export import export_chirps_parquet
            compute_stats = not args.no_daily_stats
            pq = export_chirps_parquet(
                chirps_paths,
                merged_dir,
                aggregate_monthly=args.chirps_monthly,
                compute_daily_stats=compute_stats,
            )
            logger.info("CHIRPS Parquet: %s", pq)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=== Collection complete ===")
    logger.info(
        "TerraClimate files: %d, CHIRPS files: %d",
        len(tc_paths), len(chirps_paths),
    )

    if tc_paths:
        import xarray as xr
        sample = xr.open_dataset(tc_paths[0])
        logger.info(
            "TerraClimate variables in output: %s", list(sample.data_vars),
        )
        sample.close()


if __name__ == "__main__":
    main()
