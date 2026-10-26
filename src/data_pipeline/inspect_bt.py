"""Inspect one or all calibrated brightness-temperature NetCDF files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
VARIABLE = "brightness_temperature"


def find_input_files(data_dir: Path) -> list[Path]:
    """Find BT outputs without matching *_BT_prepared.nc files."""
    return sorted(
        path
        for pattern in ("*_BT.nc", "*_BT.nc4")
        for path in data_dir.glob(pattern)
        if not path.stem.endswith("_prepared")
    )


def inspect_file(path: Path, plot: bool) -> None:
    with xr.open_dataset(path) as ds:
        print(f"\nFILE: {path.name}")
        if VARIABLE not in ds:
            raise KeyError(f"missing {VARIABLE!r}; found: {', '.join(ds.data_vars)}")
        data = ds[VARIABLE]
        if data.ndim < 2:
            raise ValueError(f"{VARIABLE!r} has {data.ndim} dimension(s); expected an image")
        extras = [dim for dim in data.dims if dim not in set(data.dims[-2:])]
        if extras:
            selection = {dim: 0 for dim in extras}
            data = data.isel(selection)
            print(f"Selected index 0 along extra dimensions: {selection}")
        image = np.asarray(data.values)
        finite = np.isfinite(image)
        values = image[finite]
        print(f"Variable attributes: {data.attrs}")
        print(f"Shape: {image.shape}")
        print(f"Units: {data.attrs.get('units', 'not specified')}")
        print(f"Finite pixels: {values.size:,}")
        print(f"Non-finite pixels: {(~finite).sum():,}")
        if values.size:
            print(f"Minimum: {values.min():.4f}")
            print(f"Maximum: {values.max():.4f}")
            print(f"Mean: {values.mean():.4f}")
        else:
            print("Minimum/maximum/mean: no finite values")
        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            plt.imshow(image, origin="upper", cmap="inferno")
            plt.colorbar(label=data.attrs.get("units", "Stored value"))
            plt.title(f"{path.name}\n{VARIABLE}")
            plt.tight_layout()
            plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Inspect one BT file; defaults to all *_BT.nc files")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory searched without --input")
    parser.add_argument("--plot", action="store_true", help="Plot each file explicitly selected for inspection")
    args = parser.parse_args()
    if args.input:
        files = [args.input if args.input.is_absolute() else PROJECT_ROOT / args.input]
    else:
        directory = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
        files = find_input_files(directory)
    print(f"BT files found: {len(files)}")
    if not files:
        print("No *_BT.nc or *_BT.nc4 files found.")
        return
    failed = 0
    for path in files:
        try:
            if not path.is_file():
                raise FileNotFoundError(f"file does not exist: {path}")
            inspect_file(path, args.plot)
        except Exception as error:
            failed += 1
            print(f"ERROR inspecting {path.name}: {error}")
    print(f"\nInspection summary: {len(files) - failed} succeeded; {failed} failed.")


if __name__ == "__main__":
    main()
