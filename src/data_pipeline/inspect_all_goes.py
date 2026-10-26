"""Lightweight inventory of every raw GOES/NetCDF file in data/raw."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def find_input_files(data_dir: Path) -> list[Path]:
    """Return all .nc and .nc4 files without opening their image data."""
    return sorted((*data_dir.glob("*.nc"), *data_dir.glob("*.nc4")))


def identify_channel(dataset_name: str, filename: str) -> str:
    match = re.search(r"C(\d{2})", f"{dataset_name} {filename}".upper())
    return f"C{match.group(1)}" if match else "unknown"


def inspect_file(path: Path) -> None:
    with xr.open_dataset(path) as ds:
        dataset_name = str(ds.attrs.get("dataset_name", "not present"))
        platform = str(ds.attrs.get("platform_ID", "unknown"))
        channel = identify_channel(dataset_name, path.name)
        print(f"\nFILE: {path.name}")
        print(f"  dataset_name: {dataset_name}")
        print(f"  platform: {platform}")
        print(f"  channel: {channel}")
        print(f"  dimensions: {dict(ds.sizes)}")
        print(f"  Rad present: {'yes' if 'Rad' in ds else 'no'}")
        if channel != "C13":
            print(f"  CHANNEL FLAG: NOT Channel 13 ({channel})")
        if "Rad" not in ds:
            print("  Rad shape: unavailable; finite count: unavailable; non-finite count: unavailable; radiance range: unavailable")
            return
        rad = ds["Rad"]
        values = np.asarray(rad.values)
        finite = np.isfinite(values)
        finite_values = values[finite]
        print(f"  Rad shape: {values.shape}")
        print(f"  Rad finite count: {finite_values.size:,}")
        print(f"  Rad non-finite count: {(~finite).sum():,}")
        if finite_values.size:
            print(f"  Radiance min/max: {finite_values.min():.6g} / {finite_values.max():.6g} {rad.attrs.get('units', '')}")
        else:
            print("  Radiance min/max: no finite radiance values")


def main() -> None:
    files = find_input_files(DEFAULT_DATA_DIR)
    print(f"Files found in {DEFAULT_DATA_DIR}: {len(files)}")
    if not files:
        print("No .nc or .nc4 files found.")
        return
    failed = 0
    for path in files:
        try:
            inspect_file(path)
        except Exception as error:
            failed += 1
            print(f"\nERROR inspecting {path.name}: {error}")
    print(f"\nInventory summary: {len(files) - failed} succeeded; {failed} failed.")


if __name__ == "__main__":
    main()
