"""Normalize one or all calibrated BT frames for later optical-flow use.

Each output preserves brightness temperature, adds
``normalized_brightness_temperature`` in [0, 1], and retains a separate
``valid_mask`` (1 = finite observation, 0 = invalid). Invalid normalized
pixels are filled with 0.5; the mask identifies which values are real.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
VARIABLE = "brightness_temperature"
NORMALIZED_VARIABLE = "normalized_brightness_temperature"
MASK_VARIABLE = "valid_mask"
INVALID_FILL_VALUE = np.float32(0.5)


def find_input_files(data_dir: Path) -> list[Path]:
    """Find calibrated BT files only, never a prepared output."""
    return sorted(
        path
        for pattern in ("*_BT.nc", "*_BT.nc4")
        for path in data_dir.glob(pattern)
        if not path.stem.endswith("_prepared")
    )


def prepare_dataset(source: xr.Dataset) -> tuple[xr.Dataset, float, float, int, int]:
    """Create normalized BT and mask while retaining the original dataset."""
    if VARIABLE not in source:
        raise KeyError(f"Expected {VARIABLE!r}; found: {', '.join(source.data_vars)}")
    data = source[VARIABLE].astype("float64")
    if data.attrs.get("units") != "K":
        raise ValueError(f"Expected {VARIABLE!r} in Kelvin; got {data.attrs.get('units')!r}")

    valid = np.isfinite(data)
    valid_values = data.where(valid)
    minimum = float(valid_values.min(skipna=True).values)
    maximum = float(valid_values.max(skipna=True).values)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Brightness-temperature image has no finite observations")
    if maximum <= minimum:
        raise ValueError(f"Cannot normalize a constant or invalid range: {minimum} to {maximum} K")

    normalized = ((data - minimum) / (maximum - minimum)).where(valid, INVALID_FILL_VALUE).astype("float32")
    normalized.name = NORMALIZED_VARIABLE
    normalized.attrs = {
        "long_name": "Min-max normalized Channel 13 brightness temperature",
        "units": "1",
        "valid_range": np.array([0.0, 1.0], dtype=np.float32),
        "normalization": "(brightness_temperature - normalization_min_kelvin) / (normalization_max_kelvin - normalization_min_kelvin)",
        "invalid_pixel_fill_value": float(INVALID_FILL_VALUE),
        "invalid_pixel_handling": "0.5 is a fill value for invalid pixels; valid_mask identifies actual valid observations",
    }
    mask = valid.astype("uint8")
    mask.name = MASK_VARIABLE
    mask.attrs = {
        "long_name": "Valid brightness-temperature observation mask",
        "units": "1",
        "flag_values": np.array([0, 1], dtype=np.uint8),
        "flag_meanings": "invalid valid",
    }

    result = source.copy()
    result[NORMALIZED_VARIABLE] = normalized
    result[MASK_VARIABLE] = mask
    result.attrs = dict(source.attrs)
    result.attrs.update(
        {
            "normalization_method": "Per-frame min-max scaling over finite brightness temperatures",
            "normalization_min_kelvin": minimum,
            "normalization_max_kelvin": maximum,
            "invalid_normalized_fill_value": float(INVALID_FILL_VALUE),
            "valid_mask_variable": MASK_VARIABLE,
            "normalization_note": "The 0.5 value is only a fill for invalid pixels; valid_mask identifies actual observations.",
        }
    )
    values = np.asarray(data.values)
    valid_pixels = int(np.isfinite(values).sum())
    invalid_pixels = int(values.size - valid_pixels)
    return result, minimum, maximum, valid_pixels, invalid_pixels


def process_file(input_path: Path, overwrite: bool = False, output_path: Path | None = None) -> tuple[Path, float, float, int, int]:
    """Prepare one frame and write it before moving to the next frame."""
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    output_path = output_path or input_path.with_name(f"{input_path.stem}_prepared.nc")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path must differ from the calibrated input path")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite to replace): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(input_path) as source:
        prepared, minimum, maximum, valid, invalid = prepare_dataset(source)
        prepared.to_netcdf(output_path)
    return output_path, minimum, maximum, valid, invalid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--input", type=Path, help="Prepare one specific calibrated BT file")
    choice.add_argument("--input-dir", type=Path, help="Search this directory (default: data/processed)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing prepared outputs")
    args = parser.parse_args()

    if args.input:
        input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
        files = [input_path]
    else:
        input_dir = args.input_dir or DEFAULT_DATA_DIR
        if not input_dir.is_absolute():
            input_dir = PROJECT_ROOT / input_dir
        files = find_input_files(input_dir)
    print(f"BT files found: {len(files)}")
    if not files:
        print("No *_BT.nc or *_BT.nc4 files found.")
        return

    processed = skipped = failed = 0
    for input_path in files:
        if not input_path.is_file():
            failed += 1
            print(f"FAILED: {input_path.name}: file does not exist")
            continue
        output_path = input_path.with_name(f"{input_path.stem}_prepared.nc")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"SKIPPED: {input_path.name}: output exists ({output_path.name}); use --overwrite")
            continue
        try:
            output, low, high, valid, invalid = process_file(input_path, overwrite=args.overwrite, output_path=output_path)
            processed += 1
            print(f"PROCESSED: {input_path.name}")
            print(f"  Original temperature range: {low:.2f} to {high:.2f} K")
            print("  Normalized range: 0.0 to 1.0 (valid pixels)")
            print(f"  Valid pixels: {valid:,}; invalid pixels: {invalid:,}")
            print(f"  Output: {output}")
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
