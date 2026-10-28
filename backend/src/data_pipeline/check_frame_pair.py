"""Check that two prepared GOES-19 ABI Channel 13 frames are compatible."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_VARIABLE = "normalized_brightness_temperature"
MASK_VARIABLE = "valid_mask"
COORDINATE_ATOL = 1e-7  # radians; comfortably below one ABI full-disk pixel


@dataclass
class FrameInfo:
    path: Path
    dataset_name: str
    timestamp: datetime | None
    image_shape: tuple[int, ...] | None
    valid_pixels: int
    total_pixels: int
    has_normalized: bool
    has_mask: bool
    channel_ok: bool
    x: np.ndarray | None
    y: np.ndarray | None
    x_units: str | None
    y_units: str | None
    mask_ok: bool
    image_has_valid_data: bool
    variable_attrs: dict
    mask_attrs: dict
    dataset_attrs: dict
    dimensions: dict
    error: str | None = None


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_timestamp(value: object) -> datetime | None:
    """Parse a real timestamp from a time coordinate or metadata value."""
    if value is None:
        return None
    try:
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                return None
            text = np.datetime_as_string(value, unit="us")
        elif isinstance(value, datetime):
            return value
        else:
            text = str(value).strip()
        if not text or text.lower() in {"nat", "none", "nan"}:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except (TypeError, ValueError, OverflowError):
        return None


def get_timestamp(ds: xr.Dataset) -> datetime | None:
    for coordinate in ("t", "time"):
        if coordinate in ds.coords:
            values = np.asarray(ds[coordinate].values)
            if values.size:
                timestamp = parse_timestamp(values.reshape(-1)[0])
                if timestamp:
                    return timestamp
    for attribute in ("time_coverage_start", "timestamp", "start_time"):
        timestamp = parse_timestamp(ds.attrs.get(attribute))
        if timestamp:
            return timestamp
    return None


def check_channel(path: Path, dataset_attrs: dict) -> tuple[bool, str]:
    """Require GOES-19, ABI, and C13 evidence in the file or metadata."""
    identity = " ".join(
        [path.name]
        + [str(dataset_attrs.get(key, "")) for key in ("dataset_name", "source_file", "title", "platform", "platform_ID", "instrument", "instrument_type")]
    ).upper()
    channel_match = re.search(r"C(\d{2})", identity)
    is_c13 = bool(channel_match and channel_match.group(1) == "13")
    is_goes19 = "G19" in identity or "GOES-19" in identity
    is_abi = "ABI" in identity
    reasons = []
    if not is_c13:
        reasons.append("not identified as C13")
    if not is_goes19:
        reasons.append("not identified as GOES-19")
    if not is_abi:
        reasons.append("not identified as ABI")
    return not reasons, ", ".join(reasons)


def load_frame(path: Path) -> FrameInfo:
    """Read one prepared frame at a time and retain only coordinate vectors."""
    try:
        if not path.is_file():
            raise FileNotFoundError(f"file does not exist: {path}")
        with xr.open_dataset(path) as ds:
            attrs = dict(ds.attrs)
            dataset_name = str(attrs.get("dataset_name", path.name))
            channel_ok, _ = check_channel(path, attrs)
            timestamp = get_timestamp(ds)
            dimensions = dict(ds.sizes)
            has_normalized = NORMALIZED_VARIABLE in ds
            has_mask = MASK_VARIABLE in ds
            image_shape = None
            valid_pixels = total_pixels = 0
            mask_ok = False
            image_has_valid_data = False
            variable_attrs = dict(ds[NORMALIZED_VARIABLE].attrs) if has_normalized else {}
            mask_attrs = dict(ds[MASK_VARIABLE].attrs) if has_mask else {}
            x = np.asarray(ds["x"].values) if "x" in ds.coords else None
            y = np.asarray(ds["y"].values) if "y" in ds.coords else None
            x_units = str(ds["x"].attrs.get("units", "")) if "x" in ds.coords else None
            y_units = str(ds["y"].attrs.get("units", "")) if "y" in ds.coords else None

            if has_normalized:
                image = np.asarray(ds[NORMALIZED_VARIABLE].values)
                if image.ndim != 2:
                    raise ValueError(f"{NORMALIZED_VARIABLE} must be a 2D image; got shape {image.shape}")
                image_shape = tuple(image.shape)
                if has_mask:
                    mask = np.asarray(ds[MASK_VARIABLE].values)
                    total_pixels = int(mask.size)
                    mask_finite = np.isfinite(mask)
                    binary = mask_finite & ((mask == 0) | (mask == 1))
                    mask_ok = bool(binary.all() and mask.shape == image.shape)
                    valid = mask == 1
                    valid_pixels = int(valid.sum())
                    image_has_valid_data = bool(valid_pixels and np.isfinite(image[valid]).any())
                else:
                    total_pixels = int(image.size)
                    image_has_valid_data = bool(np.isfinite(image).any())

        return FrameInfo(
            path, dataset_name, timestamp, image_shape, valid_pixels, total_pixels,
            has_normalized, has_mask, channel_ok, x, y, x_units, y_units, mask_ok,
            image_has_valid_data, variable_attrs, mask_attrs, attrs, dimensions,
        )
    except Exception as error:
        return FrameInfo(
            path, path.name, None, None, 0, 0, False, False, False,
            None, None, None, None, False, False, {}, {}, {}, {}, str(error),
        )


def coordinates_compatible(first: np.ndarray | None, second: np.ndarray | None) -> bool:
    if first is None or second is None or first.shape != second.shape:
        return False
    if not np.issubdtype(first.dtype, np.number) or not np.issubdtype(second.dtype, np.number):
        return np.array_equal(first, second)
    return bool(np.allclose(first, second, rtol=0.0, atol=COORDINATE_ATOL, equal_nan=False))


def coordinate_description(name: str, values: np.ndarray | None) -> str:
    if values is None:
        return f"{name}: unavailable"
    if values.size == 0:
        return f"{name}: empty (length 0)"
    return (
        f"{name}: length {values.size}, first/last {values[0]:.8g} / {values[-1]:.8g}, "
        f"range {values.min():.8g} to {values.max():.8g}"
    )


def print_frame(info: FrameInfo, label: str) -> None:
    print(f"\n{label}")
    print(f"Filename: {info.path.name}")
    if info.error:
        print(f"ERROR: {info.error}")
        return
    print(f"Dataset name: {info.dataset_name}")
    print(f"Timestamp: {info.timestamp.isoformat() if info.timestamp else 'unavailable'}")
    print(f"Dimensions: {info.dimensions}")
    print(f"Shape: {info.image_shape if info.image_shape is not None else 'unavailable'}")
    print(f"Valid pixels: {info.valid_pixels:,}")
    percentage = 100.0 * info.valid_pixels / info.total_pixels if info.total_pixels else 0.0
    print(f"Valid percentage: {percentage:.2f}%")
    print(f"Contains {NORMALIZED_VARIABLE}: {'yes' if info.has_normalized else 'no'}")
    print(f"Contains {MASK_VARIABLE}: {'yes' if info.has_mask else 'no'}")
    print(coordinate_description("x coordinate", info.x) + f" {info.x_units or ''}")
    print(coordinate_description("y coordinate", info.y) + f" {info.y_units or ''}")
    print(f"Normalized variable attributes: {info.variable_attrs}")
    print(f"Valid-mask attributes: {info.mask_attrs}")
    print(f"Dataset attributes: {info.dataset_attrs}")


def check_frame_pair(frame1_path: Path, frame2_path: Path, print_report: bool = True) -> tuple[bool, FrameInfo, FrameInfo]:
    """Return pair compatibility and per-frame summaries, reading sequentially."""
    frame1 = load_frame(resolve_path(frame1_path))
    frame2 = load_frame(resolve_path(frame2_path))
    same_shape = bool(frame1.image_shape and frame2.image_shape and frame1.image_shape == frame2.image_shape)
    same_x = coordinates_compatible(frame1.x, frame2.x)
    same_y = coordinates_compatible(frame1.y, frame2.y)
    same_x_count = bool(frame1.x is not None and frame2.x is not None and frame1.x.size == frame2.x.size)
    same_y_count = bool(frame1.y is not None and frame2.y is not None and frame1.y.size == frame2.y.size)
    spatial = same_x and same_y
    same_coordinate_units = bool(frame1.x_units == frame2.x_units and frame1.y_units == frame2.y_units)
    timestamp_available = bool(frame1.timestamp and frame2.timestamp)
    try:
        seconds = (frame2.timestamp - frame1.timestamp).total_seconds() if timestamp_available else None
    except TypeError:
        seconds = None
        timestamp_available = False
    checks = {
        "frame1_ok": frame1.error is None,
        "frame2_ok": frame2.error is None,
        "frame1_normalized": frame1.has_normalized,
        "frame2_normalized": frame2.has_normalized,
        "frame1_mask": frame1.has_mask,
        "frame2_mask": frame2.has_mask,
        "same_shape": same_shape,
        "same_x_count": same_x_count,
        "same_y_count": same_y_count,
        "same_x": same_x,
        "same_y": same_y,
        "same_coordinate_units": same_coordinate_units,
        "same_spatial_coverage": spatial and same_coordinate_units,
        "frame1_channel": frame1.channel_ok,
        "frame2_channel": frame2.channel_ok,
        "frame1_valid_data": frame1.image_has_valid_data and frame1.valid_pixels > 0,
        "frame2_valid_data": frame2.image_has_valid_data and frame2.valid_pixels > 0,
        "frame1_mask_binary": frame1.mask_ok,
        "frame2_mask_binary": frame2.mask_ok,
    }
    compatible = all(checks.values())

    if print_report:
        print("=" * 72)
        print("FRAME PAIR COMPATIBILITY CHECK")
        print("=" * 72)
        print_frame(frame1, "FRAME 1")
        print_frame(frame2, "FRAME 2")
        print("\nTEMPORAL INFORMATION")
        if seconds is None:
            print("Timestamp: unavailable")
            print("Time difference: unavailable")
        else:
            print(f"Time difference: {seconds:.3f} seconds ({seconds / 60.0:.3f} minutes; frame2 - frame1)")
        print("\nSPATIAL COMPATIBILITY")
        for label, key in (("Same shape", "same_shape"), ("Same x coordinate count", "same_x_count"), ("Same y coordinate count", "same_y_count"), ("Same x grid", "same_x"), ("Same y grid", "same_y"), ("Same coordinate units", "same_coordinate_units"), ("Same spatial coverage", "same_spatial_coverage")):
            print(f"{label}: {'PASS' if checks[key] else 'FAIL'}")
        print("\nDATA COMPATIBILITY")
        for label, key in (("Frame 1 contains normalized data", "frame1_normalized"), ("Frame 2 contains normalized data", "frame2_normalized"), ("Frame 1 contains valid_mask", "frame1_mask"), ("Frame 2 contains valid_mask", "frame2_mask"), ("Frame 1 has valid finite pixels", "frame1_valid_data"), ("Frame 2 has valid finite pixels", "frame2_valid_data"), ("Frame 1 mask is binary and image-sized", "frame1_mask_binary"), ("Frame 2 mask is binary and image-sized", "frame2_mask_binary"), ("Frame 1 is GOES-19 ABI C13", "frame1_channel"), ("Frame 2 is GOES-19 ABI C13", "frame2_channel")):
            print(f"{label}: {'PASS' if checks[key] else 'FAIL'}")
        if not frame1.channel_ok or not frame2.channel_ok:
            print("ERROR: Frames are not both Channel 13.")
        if not frame1.mask_ok or not frame2.mask_ok:
            print("ERROR: valid_mask must contain only 0 and 1.")
        if frame1.error or frame2.error:
            print(f"Frame read error: {frame1.error or frame2.error}")
        print("\nFINAL RESULT")
        print(f"FRAME PAIR SUITABLE FOR OPTICAL FLOW: {'YES' if compatible else 'NO'}")
        print("=" * 72)
    return compatible, frame1, frame2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame1", required=True, type=Path, help="First prepared frame")
    parser.add_argument("--frame2", required=True, type=Path, help="Second prepared frame")
    args = parser.parse_args()
    compatible, _, _ = check_frame_pair(args.frame1, args.frame2)
    if not compatible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
