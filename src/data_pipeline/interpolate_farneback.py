"""Generate a halfway frame using bidirectional Farneback optical flow.

This is a classical baseline, not a claim of physically correct cloud motion.
The prediction uses only the before and after frames; no ground-truth frame is
loaded by this script.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import xarray as xr

try:
    from . import optical_flow_farneback as farneback
    from .check_frame_pair import check_channel
except ImportError:
    import optical_flow_farneback as farneback
    from check_frame_pair import check_channel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "interpolation"
DEFAULT_FLOW_DIR = PROJECT_ROOT / "data" / "processed" / "optical_flow"
PREDICTED_VARIABLE = "predicted_frame"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_matching_flow(path: Path, before_path: Path, after_path: Path, expected_mask: np.ndarray, before: dict) -> np.ndarray:
    """Read dx/dy only when file metadata identifies this exact source pair."""
    with xr.open_dataset(path) as ds:
        if str(ds.attrs.get("frame1_filename", "")) != before_path.name or str(ds.attrs.get("frame2_filename", "")) != after_path.name:
            raise ValueError(f"flow file {path} is not for the requested before/after frames")
        for name in ("dx", "dy", "common_valid_mask"):
            if name not in ds:
                raise KeyError(f"flow file is missing {name!r}")
        dx = np.asarray(ds["dx"].values, dtype=np.float32)
        dy = np.asarray(ds["dy"].values, dtype=np.float32)
        mask = np.asarray(ds["common_valid_mask"].values)
        if dx.shape != before["image"].shape or dy.shape != dx.shape or mask.shape != dx.shape:
            raise ValueError("cached flow dimensions do not match input frames")
        if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError("cached common_valid_mask is not binary")
        if not np.array_equal(mask == 1, expected_mask):
            raise ValueError("cached flow common-valid mask does not match these source frames")
        if not np.isfinite(dx[expected_mask]).all() or not np.isfinite(dy[expected_mask]).all():
            raise ValueError("cached flow has non-finite vectors on common-valid pixels")
        if "x" in ds.coords and not np.allclose(ds["x"].values, before["x"], rtol=0, atol=1e-7):
            raise ValueError("cached flow x grid does not match input frames")
        if "y" in ds.coords and not np.allclose(ds["y"].values, before["y"], rtol=0, atol=1e-7):
            raise ValueError("cached flow y grid does not match input frames")
    return np.stack((dx, dy), axis=-1)


def find_matching_flow(before_path: Path, after_path: Path, explicit_flow: Path | None, expected_mask: np.ndarray, before: dict) -> tuple[np.ndarray | None, Path | None]:
    """Find a pair-specific saved flow; ignore generic flows for other pairs."""
    if explicit_flow is not None:
        path = resolve_path(explicit_flow)
        return read_matching_flow(path, before_path, after_path, expected_mask, before), path
    candidates = sorted(DEFAULT_FLOW_DIR.rglob("flow_frame1_to_frame2.nc")) if DEFAULT_FLOW_DIR.exists() else []
    for path in candidates:
        try:
            flow = read_matching_flow(path, before_path, after_path, expected_mask, before)
            return flow, path
        except (ValueError, KeyError, OSError):
            continue
    return None, None


def warp_halfway(image: np.ndarray, source_mask: np.ndarray, flow: np.ndarray, flow_valid: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Warp an image by fraction of its source-to-target displacement.

    Farneback stores forward displacement: target = source + flow(source).
    OpenCV remap wants destination-to-source coordinates, so fixed-point
    iteration solves source = target - fraction * flow(source). Image values
    are resampled as a validity-weighted average; invalid fill 0.5 contributes
    nothing to the warped observation.
    """
    height, width = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    finite_flow = np.isfinite(flow).all(axis=2)
    safe_flow = np.where(finite_flow[..., None], flow, 0.0).astype(np.float32)
    map_x = grid_x - fraction * safe_flow[..., 0]
    map_y = grid_y - fraction * safe_flow[..., 1]
    flow_valid = flow_valid & finite_flow

    for _ in range(6):
        sampled_dx = cv2.remap(safe_flow[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sampled_dy = cv2.remap(safe_flow[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        map_x = grid_x - fraction * sampled_dx
        map_y = grid_y - fraction * sampled_dy

    source_valid = (source_mask == 1) & np.isfinite(image)
    weights = cv2.remap(source_valid.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    numerator = cv2.remap(
        np.where(source_valid, image, 0.0).astype(np.float32), map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped = np.full((height, width), 0.5, dtype=np.float32)
    has_support = weights > 1e-6
    warped[has_support] = numerator[has_support] / weights[has_support]

    nearest_source_valid = cv2.remap(source_valid.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) == 1
    nearest_flow_valid = cv2.remap(flow_valid.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) == 1
    in_bounds = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    output_mask = nearest_source_valid & nearest_flow_valid & has_support & in_bounds & np.isfinite(warped)
    warped = np.clip(warped, 0.0, 1.0)
    warped[~output_mask] = 0.5
    return warped, output_mask


def midpoint_timestamp(first: datetime | None, second: datetime | None) -> datetime | None:
    if first is None or second is None:
        return None
    try:
        return first + (second - first) / 2
    except TypeError as error:
        raise ValueError("input timestamps use incompatible timezone information") from error


def interpolate_frames(before_path: Path, after_path: Path, output_path: Path, flow_path: Path | None = None) -> dict:
    """Interpolate the halfway frame from the two endpoint flow-input files."""
    before_path = resolve_path(before_path)
    after_path = resolve_path(after_path)
    output_path = resolve_path(output_path)
    before = farneback.load_flow_frame(before_path)
    after = farneback.load_flow_frame(after_path)
    farneback.validate_inputs(before, after)
    for path, frame in ((before_path, before), (after_path, after)):
        valid_c13, reason = check_channel(path, frame["dataset_attrs"])
        if not valid_c13:
            raise ValueError(f"{path.name} is not a GOES-19 ABI C13 flow-input frame: {reason}")
    if before["timestamp"] and after["timestamp"] and after["timestamp"] <= before["timestamp"]:
        raise ValueError("frame-after timestamp must be later than frame-before timestamp")

    common_mask = farneback.create_common_mask(before["mask"], after["mask"])
    if not common_mask.any():
        raise ValueError("source frames have no common-valid pixels")
    parameters = dict(farneback.DEFAULT_PARAMETERS)
    flow, cached_flow_path = find_matching_flow(before_path, after_path, flow_path, common_mask, before)
    flow_source = "existing pair-specific flow file" if flow is not None else "calculated with existing Farneback implementation"
    if flow is None:
        flow, _, _, flow_valid = farneback.calculate_farneback_flow(
            before["image"], after["image"], common_mask, parameters
        )
    else:
        flow_valid = common_mask & np.isfinite(flow).all(axis=2)
    # A second, reverse-direction field is needed to warp the later frame back
    # toward the halfway time. It is calculated with the same existing method.
    backward_flow, _, _, backward_valid = farneback.calculate_farneback_flow(
        after["image"], before["image"], common_mask, parameters
    )

    halfway_before, valid_before = warp_halfway(before["image"], before["mask"], flow, flow_valid, 0.5)
    halfway_after, valid_after = warp_halfway(after["image"], after["mask"], backward_flow, backward_valid, 0.5)
    predicted_mask = valid_before & valid_after
    if not predicted_mask.any():
        raise ValueError("bidirectional halfway warps have no common-valid pixels")
    predicted = np.full(before["image"].shape, 0.5, dtype=np.float32)
    predicted[predicted_mask] = 0.5 * (halfway_before[predicted_mask] + halfway_after[predicted_mask])

    middle_time = midpoint_timestamp(before["timestamp"], after["timestamp"])
    delta_seconds = (
        (after["timestamp"] - before["timestamp"]).total_seconds()
        if before["timestamp"] is not None and after["timestamp"] is not None
        else None
    )
    coordinates = {
        "x": ("x", before["x"], before["x_attrs"]),
        "y": ("y", before["y"], before["y_attrs"]),
    }
    if middle_time is not None:
        if middle_time.tzinfo is not None:
            middle_time = middle_time.astimezone(timezone.utc).replace(tzinfo=None)
        coordinates["t"] = np.datetime64(middle_time, "ns")
    output = xr.Dataset(
        {
            PREDICTED_VARIABLE: (
                ("y", "x"), predicted,
                {
                    "long_name": "Farneback optical-flow-based predicted intermediate normalized brightness temperature",
                    "units": "1",
                    "valid_range": np.array([0.0, 1.0], dtype=np.float32),
                    "invalid_pixel_fill_value": 0.5,
                    "invalid_pixel_handling": "Use valid_mask; 0.5 is a fill value, not an observation",
                },
            ),
            "valid_mask": (("y", "x"), predicted_mask.astype(np.uint8), {
                "long_name": "Valid predicted intermediate observation mask",
                "units": "1",
                "flag_values": np.array([0, 1], dtype=np.uint8),
                "flag_meanings": "invalid valid",
            }),
        },
        coords=coordinates,
        attrs={
            "title": "Farneback optical-flow-based baseline frame interpolation",
            "frame_before": before_path.name,
            "frame_after": after_path.name,
            "frame_before_timestamp": before["timestamp"].isoformat() if before["timestamp"] else "unavailable",
            "frame_after_timestamp": after["timestamp"].isoformat() if after["timestamp"] else "unavailable",
            "predicted_timestamp": middle_time.isoformat() if middle_time else "unavailable",
            "endpoint_time_difference_seconds": delta_seconds if delta_seconds is not None else "unavailable",
            "flow_source": f"{flow_source}: {cached_flow_path}" if cached_flow_path else flow_source,
            "farneback_parameters_json": json.dumps(parameters),
            "interpolation_method": "Bidirectional half-flow warp and average of the two halfway images",
            "normalization_method": before["dataset_attrs"].get("normalization_method", "inherited from normalized flow inputs"),
            "normalization_min_kelvin_frame_before": before["dataset_attrs"].get("normalization_min_kelvin", "unavailable"),
            "normalization_max_kelvin_frame_before": before["dataset_attrs"].get("normalization_max_kelvin", "unavailable"),
            "normalization_min_kelvin_frame_after": after["dataset_attrs"].get("normalization_min_kelvin", "unavailable"),
            "normalization_max_kelvin_frame_after": after["dataset_attrs"].get("normalization_max_kelvin", "unavailable"),
            "scientific_caveat": "Baseline interpolation only; metric improvement does not prove physically correct cloud motion.",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(output_path)
    return {
        "output": output_path,
        "timestamp": middle_time,
        "time_difference_seconds": delta_seconds,
        "valid_pixels": int(predicted_mask.sum()),
        "flow_source": flow_source,
        "parameters": parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-before", required=True, type=Path, help="Earlier flow-input NetCDF")
    parser.add_argument("--frame-after", required=True, type=Path, help="Later flow-input NetCDF")
    parser.add_argument("--output", type=Path, help="Predicted NetCDF path (optional)")
    parser.add_argument("--flow", type=Path, help="Optional matching source-to-target flow NetCDF")
    args = parser.parse_args()
    before = resolve_path(args.frame_before)
    after = resolve_path(args.frame_after)
    output = resolve_path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"predicted_{before.stem}_to_{after.stem}.nc"
    summary = interpolate_frames(before, after, output, args.flow)
    print("Farneback optical-flow-based baseline frame interpolation complete.")
    print(f"Frame before: {before.name}")
    print(f"Frame after: {after.name}")
    print(f"Predicted midpoint timestamp: {summary['timestamp'].isoformat() if summary['timestamp'] else 'unavailable'}")
    print(f"Valid predicted pixels: {summary['valid_pixels']:,}")
    print(f"Flow: {summary['flow_source']}")
    print(f"Output: {summary['output']}")


if __name__ == "__main__":
    main()
