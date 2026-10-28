"""Inspect one or all raw GOES NetCDF files in data/raw."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def find_input_files(data_dir: Path) -> list[Path]:
    return sorted((*data_dir.glob("*.nc"), *data_dir.glob("*.nc4")))


def inspect_file(file_path: Path, variable_name: str | None, plot: bool) -> None:
    """Print one file's metadata and Rad/image statistics; optionally plot it."""
    with xr.open_dataset(file_path) as ds:
        print(f"\n{'=' * 78}\nFILE: {file_path.name}")
        print(f"DATASET_NAME: {ds.attrs.get('dataset_name', 'not present')}")
        print("\nDIMENSIONS:")
        print(ds.sizes)
        print("\nDATA VARIABLES:")
        print(list(ds.data_vars))
        print("\nCOORDINATES:")
        print(list(ds.coords))
        print("\nGLOBAL ATTRIBUTES:")
        print(ds.attrs)

        selected = variable_name or ("Rad" if "Rad" in ds else None)
        if selected is None:
            print("Rad variable information: Rad is absent; no image statistics available.")
            return
        if selected not in ds:
            raise KeyError(f"variable {selected!r} is absent; available: {', '.join(ds.data_vars)}")
        data = ds[selected]
        print(f"\nVARIABLE ({selected}) STRUCTURE:")
        print(data)
        print("VARIABLE ATTRIBUTES:")
        print(data.attrs)
        if data.ndim < 2:
            raise ValueError(f"variable {selected!r} has {data.ndim} dimension(s); expected an image")
        plane_dims = set(data.dims[-2:])
        extra = [dim for dim in data.dims if dim not in plane_dims]
        if extra:
            selection = {dim: 0 for dim in extra}
            data = data.isel(selection)
            print(f"Selected index 0 along extra dimensions: {selection}")
        image = np.asarray(data.values)
        finite = np.isfinite(image)
        finite_values = image[finite]
        print(f"SHAPE: {image.shape}")
        print(f"FINITE PIXELS: {finite_values.size:,}")
        print(f"MINIMUM: {finite_values.min() if finite_values.size else 'no finite values'}")
        print(f"MAXIMUM: {finite_values.max() if finite_values.size else 'no finite values'}")
        print(f"NON-FINITE PIXELS: {(~finite).sum():,}")
        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            plt.imshow(image, origin="upper")
            plt.colorbar(label=data.attrs.get("units", "Stored value"))
            plt.title(f"{file_path.name}\n{selected}")
            plt.tight_layout()
            plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="Inspect this raw file instead of all files")
    parser.add_argument("--input", "--file", dest="input", type=Path, help="Inspect this raw file instead of all files")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory searched when no file is specified")
    parser.add_argument("--variable", help="Image variable to inspect (defaults to Rad)")
    parser.add_argument("--plot", action="store_true", help="Plot each inspected file; off by default")
    args = parser.parse_args()
    if args.file and args.input:
        parser.error("specify a file either positionally or with --input, not both")
    selected = args.input or args.file
    if selected:
        files = [selected if selected.is_absolute() else PROJECT_ROOT / selected]
    else:
        directory = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
        files = find_input_files(directory)
    print(f"Files found: {len(files)}")
    if not files:
        print("No .nc or .nc4 files found.")
        return
    failed = 0
    for path in files:
        try:
            if not path.is_file():
                raise FileNotFoundError(f"file does not exist: {path}")
            inspect_file(path, args.variable, args.plot)
        except Exception as error:
            failed += 1
            print(f"ERROR inspecting {path.name}: {error}")
    print(f"\nInspection summary: {len(files) - failed} succeeded; {failed} failed.")


if __name__ == "__main__":
    main()
