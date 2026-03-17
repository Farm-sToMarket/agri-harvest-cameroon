"""Convert CSV dataset to Parquet for 5x faster loading and 50% less memory.

Usage:
    python -m models.v1.convert_parquet data/cameroon_agricultural_features.csv
"""

import sys
from pathlib import Path

import polars as pl

from models.v1.config import STRATIFY_COLUMN


def convert_csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path | None = None,
    compression: str = "zstd",
) -> Path:
    """Read CSV with Polars, optimize dtypes, write Parquet.

    Parameters
    ----------
    csv_path : path to source CSV
    parquet_path : output path (default: same name with .parquet suffix)
    compression : 'zstd' (best ratio) or 'snappy' (fastest decompression)
    """
    csv_path = Path(csv_path)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    else:
        parquet_path = Path(parquet_path)

    print(f"Reading {csv_path} ...")
    df = pl.read_csv(csv_path, low_memory=True)
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Optimize dtypes
    float64_cols = [c for c in df.columns if df[c].dtype == pl.Float64]
    if float64_cols:
        df = df.with_columns(pl.col(float64_cols).cast(pl.Float32))

    _string_type = pl.String if hasattr(pl, "String") else pl.Utf8
    cat_candidates = [STRATIFY_COLUMN, "crop_type", "crop_name", "season", "crop_group"]
    for c in cat_candidates:
        if c in df.columns and df[c].dtype in (pl.Utf8, _string_type):
            df = df.with_columns(pl.col(c).cast(pl.Categorical))

    df.write_parquet(parquet_path, compression=compression, use_pyarrow=True)

    size_mb = parquet_path.stat().st_size / (1024 ** 2)
    print(f"  Saved to {parquet_path} ({size_mb:.1f} MB, {compression})")
    return parquet_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m models.v1.convert_parquet <csv_path> [parquet_path]")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    convert_csv_to_parquet(src, dst)
