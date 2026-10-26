"""Create small, validated image/mask inputs from prepared BT frames.

This prepares data for later optical-flow experiments. It does not calculate
optical flow. By default, every ``*_BT_prepared.nc`` file in data/processed is
processed one at a time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import xarray as xr

try:  # Supports both direct script execution and ``python -m`` execution.
    from .check_frame_pair import (
        MASK_VARIABLE,
        NORMALIZED_VARIABLE,
        PROJECT_ROOT,
        check_frame_pair,
        resolve_path,
    )
except ImportError:
    from check_frame_pair import (
        MASK_VARIABLE,
        NORMALIZED_VARIABLE,
        PROJECT_ROOT,
        check_frame_pair,
        resolve_path,
    )


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "flow_input"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed"
INVALID_FILL_VALUE = np.float32(0.5)


def find_prepared_files(input_dir: Path) -> list[Path]:
    """Find all prepared brightness-temperature frames in one directory."""
    return sorted(input_dir.glob("*_BT_prepared.nc"))


def flow_output_path(input_path: Path, output_dir: Path) -> Path:
    """Use the prepared frame name without the intermediate BT tag."""
    stem = input_path.stem
    if stem.endswith("_BT_prepared"):
        stem = stem[: -len("_BT_prepared")]
    elif stem.endswith("_prepared"):
        stem = stem[: -len("_prepared")]
    return output_dir / f"{stem}_flow_input.nc"


def resized_coordinates(values: np.ndarray, output_length: int) -> np.ndarray:
    """Map output pixel centers onto the source coordinate axes."""
    source_length = values.size
    source_positions = (np.arange(output_length, dtype=np.float64) + 0.5) * source_length / output_length - 0.5
    source_positions = np.clip(source_positions, 0, source_length - 1)
    return np.interp(source_positions, np.arange(source_length), values.astype(np.float64))


def prepare_flow_frame(input_path: Path, output_path: Path, size: int, overwrite: bool = False) -> dict:
    """Downsample one frame; invalid fill values never enter interpolation."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists (use --overwrite to replace): {output_path}")

    with xr.open_dataset(input_path) as source:
        for variable in (NORMALIZED_VARIABLE, MASK_VARIABLE):
            if variable not in source:
                raise KeyError(f"{input_path.name} is missing required variable {variable!r}")
        image_data = source[NORMALIZED_VARIABLE]
        mask_data = source[MASK_VARIABLE]
        if image_data.ndim != 2 or mask_data.ndim != 2:
            raise ValueError("normalized image and valid_mask must both be 2D")
        if image_data.dims != mask_data.dims or image_data.shape != mask_data.shape:
            raise ValueError("normalized image and valid_mask dimensions/shapes do not match")
        y_dim, x_dim = image_data.dims
        if x_dim not in source.coords or y_dim not in source.coords:
            raise ValueError("prepared frame must have 1D x and y coordinates")
        x_values = np.asarray(source[x_dim].values)
        y_values = np.asarray(source[y_dim].values)
        if x_values.ndim != 1 or y_values.ndim != 1:
            raise ValueError("x and y coordinates must be one-dimensional")
        original_image = np.asarray(image_data.values, dtype=np.float32)
        original_mask = np.asarray(mask_data.values)
        if size >= min(original_image.shape):
            raise ValueError(
                f"--size {size} must be smaller than both source dimensions "
                f"for downsampling; source shape is {original_image.shape}"
            )
        if not np.isfinite(original_mask).all() or not np.isin(original_mask, (0, 1)).all():
            raise ValueError("valid_mask must contain only finite 0 and 1 values")
        if not np.isfinite(original_image[original_mask == 1]).all():
            raise ValueError("valid pixels in the normalized image contain non-finite values")
        if not (original_mask == 1).any():
            raise ValueError("frame is entirely invalid")

        source_valid = (original_mask == 1) & np.isfinite(original_image)
        source_weight = source_valid.astype(np.float32)
        # Area resampling is suitable for continuous images when shrinking.
        # Resample only real observations, then divide by valid coverage so
        # the 0.5 invalid fill is never interpreted as image information.
        numerator = cv2.resize(
            np.where(source_valid, original_image, 0.0),
            (size, size),
            interpolation=cv2.INTER_AREA,
        )
        coverage = cv2.resize(source_weight, (size, size), interpolation=cv2.INTER_AREA)
        # Nearest-neighbor keeps the validity mask categorical/binary.
        resized_mask = cv2.resize(
            original_mask.astype(np.uint8),
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
        output_valid = (resized_mask == 1) & (coverage > 0)
        resized_image = np.full((size, size), INVALID_FILL_VALUE, dtype=np.float32)
        resized_image[output_valid] = numerator[output_valid] / coverage[output_valid]
        resized_image = np.clip(resized_image, 0.0, 1.0)
        resized_mask = output_valid.astype(np.uint8)

        coordinates = {
            x_dim: (x_dim, resized_coordinates(x_values, size), dict(source[x_dim].attrs)),
            y_dim: (y_dim, resized_coordinates(y_values, size), dict(source[y_dim].attrs)),
        }
        # Retain time coordinates and any scalar coordinates such as y_image.
        for name, coordinate in source.coords.items():
            if name in coordinates:
                continue
            if coordinate.ndim == 0 and name != x_dim and name != y_dim:
                coordinates[name] = coordinate

        normalized_attrs = dict(image_data.attrs)
        normalized_attrs.update(
            {
                "invalid_pixel_fill_value": float(INVALID_FILL_VALUE),
                "invalid_pixel_handling": "0.5 is a fill value for invalid pixels; valid_mask identifies actual valid observations",
                "resampling": "Area-weighted average over valid source pixels only",
            }
        )
        mask_attrs = dict(mask_data.attrs)
        mask_attrs.update(
            {
                "resampling": "Nearest-neighbor; mask values remain binary",
                "flag_values": np.array([0, 1], dtype=np.uint8),
                "flag_meanings": "invalid valid",
            }
        )
        output = xr.Dataset(
            {
                NORMALIZED_VARIABLE: ((y_dim, x_dim), resized_image, normalized_attrs),
                MASK_VARIABLE: ((y_dim, x_dim), resized_mask, mask_attrs),
            },
            coords=coordinates,
            attrs=dict(source.attrs),
        )
        output.attrs.update(
            {
                "original_shape": str(tuple(original_image.shape)),
                "new_shape": str(tuple(resized_image.shape)),
                "preprocessing_stage": "downsampled optical-flow input preparation; optical flow not computed",
                "source_file": input_path.name,
                "invalid_normalized_fill_value": float(INVALID_FILL_VALUE),
                "valid_mask_variable": MASK_VARIABLE,
            }
        )
        input_attrs = dict(source.attrs)
        input_timestamp = next(
            (str(source[name].values) for name in ("t", "time") if name in source.coords),
            "unavailable",
        )
        valid_pixels = int(resized_mask.sum())
        invalid_pixels = int(resized_mask.size - valid_pixels)
        original_shape = tuple(original_image.shape)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_netcdf(output_path)

    return {
        "input": input_path,
        "output": output_path,
        "original_shape": original_shape,
        "new_shape": (size, size),
        "valid_pixels": valid_pixels,
        "invalid_pixels": invalid_pixels,
        "timestamp": input_timestamp,
        "dataset_attrs": input_attrs,
    }


def validate_output(path: Path) -> dict:
    """Check that a saved flow input meets the expected shape and value rules."""
    with xr.open_dataset(path) as ds:
        missing = [name for name in (NORMALIZED_VARIABLE, MASK_VARIABLE) if name not in ds]
        if missing:
            raise ValueError(f"{path.name} is missing: {', '.join(missing)}")
        image = np.asarray(ds[NORMALIZED_VARIABLE].values)
        mask = np.asarray(ds[MASK_VARIABLE].values)
        if image.shape != mask.shape:
            raise ValueError("saved normalized image and mask shapes differ")
        if image.ndim != 2:
            raise ValueError(f"saved image must be 2D; got {image.shape}")
        if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError("saved valid_mask contains values other than 0 and 1")
        valid = mask == 1
        if not valid.any():
            raise ValueError("saved frame is entirely invalid")
        if not np.isfinite(image[valid]).all():
            raise ValueError("saved valid pixels contain non-finite normalized values")
        if ((image[valid] < -1e-6) | (image[valid] > 1.0 + 1e-6)).any():
            raise ValueError("valid normalized pixels are outside [0, 1]")
        invalid = ~valid
        if invalid.any() and not np.allclose(image[invalid], INVALID_FILL_VALUE, rtol=0, atol=1e-6):
            raise ValueError("invalid pixels are not consistently filled with 0.5")
        for name in ("x", "y"):
            if name not in ds.coords or ds[name].size != image.shape[1 if name == "x" else 0]:
                raise ValueError(f"saved {name} coordinate does not match the image shape")
        return {
            "shape": tuple(image.shape),
            "valid_pixels": int(valid.sum()),
            "invalid_pixels": int(invalid.sum()),
            "min": float(image[valid].min()),
            "max": float(image[valid].max()),
            "x": np.asarray(ds["x"].values),
            "y": np.asarray(ds["y"].values),
            "x_units": str(ds["x"].attrs.get("units", "")),
            "y_units": str(ds["y"].attrs.get("units", "")),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame1", type=Path, help="Optional first prepared frame for the previous two-frame workflow")
    parser.add_argument("--frame2", type=Path, help="Optional second prepared frame for the previous two-frame workflow")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory searched for *_BT_prepared.nc files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for new flow-input NetCDF files")
    parser.add_argument("--size", type=int, default=1024, help="Square output side length (default: 1024)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing flow-input outputs")
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be a positive integer")
    if bool(args.frame1) != bool(args.frame2):
        parser.error("--frame1 and --frame2 must be provided together")
    if args.frame1 and args.input_dir != DEFAULT_INPUT_DIR:
        parser.error("use either --frame1/--frame2 or --input-dir, not both")

    output_dir = resolve_path(args.output_dir)
    if args.frame1:
        files = [resolve_path(args.frame1), resolve_path(args.frame2)]
        print("Checking selected source pair compatibility before resizing...")
        compatible, _, _ = check_frame_pair(files[0], files[1])
        if not compatible:
            raise SystemExit("Cannot prepare flow inputs: source frames are not compatible.")
    else:
        input_dir = resolve_path(args.input_dir)
        files = find_prepared_files(input_dir)

    print(f"Prepared BT files found: {len(files)}")
    if not files:
        print("No *_BT_prepared.nc files found.")
        print("Batch summary: Files found: 0; Files processed: 0; Files skipped: 0; Files failed: 0")
        return

    processed = skipped = failed = 0
    for input_path in files:
        output_path = flow_output_path(input_path, output_dir)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"SKIPPED: {input_path.name}: output already exists ({output_path.name})")
            continue
        try:
            summary = prepare_flow_frame(input_path, output_path, args.size, overwrite=args.overwrite)
            validated = validate_output(output_path)
            summary.update(validated)
            processed += 1
            print(f"\nPROCESSED: {input_path.name}")
            print(f"  Source: {input_path.name}")
            print(f"  Shape: {summary['original_shape']} -> {summary['shape']}")
            print(f"  Valid pixels: {summary['valid_pixels']:,}; invalid pixels: {summary['invalid_pixels']:,}")
            print(f"  Valid normalized range: {summary['min']:.6f} to {summary['max']:.6f}")
            print(f"  Output: {output_path}")
        except Exception as error:
            failed += 1
            print(f"FAILED: {input_path.name}: {error}")

    print("\nBatch summary:")
    print(f"  Files found: {len(files)}")
    print(f"  Files processed: {processed}")
    print(f"  Files skipped: {skipped}")
    print(f"  Files failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print("No optical flow was calculated.")


if __name__ == "__main__":
    main()
