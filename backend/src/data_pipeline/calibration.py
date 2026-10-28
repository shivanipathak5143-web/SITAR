"""Convert GOES-19 ABI Channel 13 radiance files to brightness temperature."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"
PLANCK_ATTRIBUTES = ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2")


def identify_channel(name: str) -> str | None:
    """Extract an ABI channel token such as C13 from a file/dataset name."""
    match = re.search(r"C(\d{2})", name.upper())
    return f"C{match.group(1)}" if match else None


def is_goes19_abi_c13(path: Path, dataset_name: str, platform: str, instrument: str) -> tuple[bool, str]:
    """Require positive evidence that the input is GOES-19 ABI Channel 13."""
    channel = identify_channel(dataset_name) or identify_channel(path.name)
    if channel != "C13":
        return False, f"{channel or 'unknown channel'} file is not Channel 13"
    platform_text = f"{platform} {path.name}".upper()
    if "G19" not in platform_text and "GOES-19" not in platform_text:
        return False, "file is not identified as GOES-19"
    if "ABI" not in f"{instrument} {dataset_name} {path.name}".upper():
        return False, "file is not identified as ABI data"
    return True, ""


def find_input_files(input_dir: Path) -> list[Path]:
    """List .nc and .nc4 files in one directory, sorted by filename."""
    return sorted((*input_dir.glob("*.nc"), *input_dir.glob("*.nc4")))


def radiance_to_brightness_temperature(radiance: xr.DataArray) -> xr.DataArray:
    """Apply the GOES ABI Planck inversion; invalid/non-positive radiance is NaN."""
    missing = [name for name in PLANCK_ATTRIBUTES if name not in radiance.attrs]
    if missing:
        raise ValueError(f"Radiance is missing Planck calibration attributes: {', '.join(missing)}")

    fk1, fk2, bc1, bc2 = (float(radiance.attrs[name]) for name in PLANCK_ATTRIBUTES)
    rad = radiance.astype("float64")
    valid = np.isfinite(rad) & (rad > 0)
    # Keep the existing scientifically correct ABI inverse-Planck equation.
    temperature = (fk2 / np.log1p(fk1 / rad) - bc1) / bc2
    temperature = temperature.where(valid)
    temperature.name = "brightness_temperature"
    temperature.attrs = {
        "long_name": "Channel 13 brightness temperature",
        "standard_name": "toa_brightness_temperature",
        "units": "K",
        "source_variable": radiance.name or "Rad",
        "calibration": "GOES ABI Planck inversion with fk1, fk2, bc1, and bc2",
    }
    return temperature


def process_file(input_path: Path, output_path: Path, overwrite: bool = False) -> tuple[float, float, int, int]:
    """Calibrate one input while it is open; return range and valid/invalid counts."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite to replace): {output_path}")
    with xr.open_dataset(input_path) as source:
        if "Rad" not in source:
            raise ValueError("missing required Rad variable")
        dataset_name = str(source.attrs.get("dataset_name", ""))
        platform = str(source.attrs.get("platform_ID", ""))
        instrument = str(source.attrs.get("instrument_type", ""))
        valid_input, reason = is_goes19_abi_c13(input_path, dataset_name, platform, instrument)
        if not valid_input:
            raise ValueError(reason)

        rad = source["Rad"].copy()
        # NOAA stores the Planck coefficients as scalar data variables.
        for name in PLANCK_ATTRIBUTES:
            if name not in source:
                raise ValueError(f"missing calibration variable {name!r}")
            rad.attrs[name] = float(source[name].values)
        temperature = radiance_to_brightness_temperature(rad)
        result = xr.Dataset({"brightness_temperature": temperature})
        for coordinate in ("x", "y", "t"):
            if coordinate in source.coords:
                result = result.assign_coords({coordinate: source[coordinate]})
                result[coordinate].encoding = {}
        result.attrs = {
            "title": "GOES-19 ABI Channel 13 brightness temperature",
            "source_file": input_path.name,
            "dataset_name": dataset_name or input_path.name,
            "platform": platform,
            "instrument": instrument,
        }

        values = np.asarray(temperature.values)
        finite = np.isfinite(values)
        valid_count = int(finite.sum())
        invalid_count = int(values.size - valid_count)
        if valid_count == 0:
            raise ValueError("calibration produced no finite brightness temperatures")
        minimum = float(values[finite].min())
        maximum = float(values[finite].max())
        result.to_netcdf(output_path)
        return minimum, maximum, valid_count, invalid_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--input", type=Path, help="Process one specific raw NetCDF file")
    choice.add_argument("--input-dir", type=Path, help="Process all .nc/.nc4 files in this directory (default: data/raw)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing brightness-temperature outputs")
    args = parser.parse_args()

    if args.input:
        input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
        files = [input_path]
    else:
        input_dir = args.input_dir or DEFAULT_INPUT
        if not input_dir.is_absolute():
            input_dir = PROJECT_ROOT / input_dir
        files = find_input_files(input_dir)
    print(f"Files found: {len(files)}")
    if not files:
        print("No .nc or .nc4 files found.")
        return

    processed = skipped = failed = 0
    for input_path in files:
        if not input_path.is_file():
            failed += 1
            print(f"FAILED: {input_path.name}: file does not exist")
            continue
        output_path = DEFAULT_OUTPUT / f"{input_path.stem}_BT.nc"
        try:
            # Inspect metadata first so wrong-channel/non-GOES inputs are skips.
            with xr.open_dataset(input_path) as source:
                ok, reason = is_goes19_abi_c13(
                    input_path,
                    str(source.attrs.get("dataset_name", "")),
                    str(source.attrs.get("platform_ID", "")),
                    str(source.attrs.get("instrument_type", "")),
                )
                has_rad = "Rad" in source
            if not ok:
                skipped += 1
                print(f"SKIPPED: {input_path.name}: {reason}")
                continue
            if not has_rad:
                failed += 1
                print(f"FAILED: {input_path.name}: missing required Rad variable")
                continue
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"SKIPPED: {input_path.name}: output exists ({output_path.name}); use --overwrite")
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            low, high, valid, invalid = process_file(input_path, output_path, overwrite=args.overwrite)
            processed += 1
            print(f"PROCESSED: {input_path.name}")
            print(f"  Output: {output_path.name}")
            print(f"  Temperature range: {low:.2f} to {high:.2f} K")
            print(f"  Valid pixels: {valid:,}; invalid pixels: {invalid:,}")
        except Exception as error:
            failed += 1
            print(f"FAILED: {input_path.name}: {error}")

    print("\nBatch summary:")
    print(f"  Files found: {len(files)}")
    print(f"  Files processed: {processed}")
    print(f"  Files skipped: {skipped}")
    print(f"  Files failed: {failed}")


if __name__ == "__main__":
    main()
