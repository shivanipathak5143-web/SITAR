"""Run a baseline OpenCV Farneback flow calculation on a prepared frame pair.

This is an exploratory baseline only. It does not establish scientific
accuracy, and it does not interpolate frames.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_VARIABLE = "normalized_brightness_temperature"
MASK_VARIABLE = "valid_mask"
EXPECTED_SHAPE = (1024, 1024)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "optical_flow"

# Conservative starting values for a baseline dense optical-flow run.
DEFAULT_PARAMETERS = {
    "pyr_scale": 0.5,  # Scale between pyramid levels.
    "levels": 3,  # Number of image pyramid levels.
    "winsize": 21,  # Averaging window; larger is smoother and less local.
    "iterations": 3,  # Solver iterations at each pyramid level.
    "poly_n": 5,  # Neighborhood size for the local polynomial expansion.
    "poly_sigma": 1.2,  # Gaussian standard deviation for that expansion.
    "flags": 0,  # No optional Farneback flags enabled.
}
QUIVER_DISPLAY_SCALE = 8.0  # Visualization only; saved flow vectors are unscaled.


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_timestamp(value: object) -> datetime | None:
    """Convert an actual dataset timestamp to datetime; return None if absent."""
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
    for name in ("t", "time"):
        if name in ds.coords:
            values = np.asarray(ds[name].values)
            if values.size:
                timestamp = parse_timestamp(values.reshape(-1)[0])
                if timestamp is not None:
                    return timestamp
    for name in ("time_coverage_start", "timestamp", "start_time"):
        timestamp = parse_timestamp(ds.attrs.get(name))
        if timestamp is not None:
            return timestamp
    return None


def load_flow_frame(path: Path) -> dict:
    """Load only the normalized image, its mask, coordinates, and metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    with xr.open_dataset(path) as ds:
        missing = [name for name in (NORMALIZED_VARIABLE, MASK_VARIABLE) if name not in ds]
        if missing:
            raise KeyError(f"{path.name} is missing required variables: {', '.join(missing)}")
        image_data = ds[NORMALIZED_VARIABLE]
        mask_data = ds[MASK_VARIABLE]
        if image_data.ndim != 2 or mask_data.ndim != 2:
            raise ValueError("normalized image and valid_mask must both be 2D")
        if image_data.shape != mask_data.shape or image_data.dims != mask_data.dims:
            raise ValueError("normalized image and valid_mask shapes/dimensions do not match")
        image = np.asarray(image_data.load().values, dtype=np.float32)
        mask = np.asarray(mask_data.load().values)
        x_name = image_data.dims[-1]
        y_name = image_data.dims[-2]
        if x_name not in ds.coords or y_name not in ds.coords:
            raise ValueError("input flow frame must contain 1D x and y coordinates")
        x = np.asarray(ds[x_name].values)
        y = np.asarray(ds[y_name].values)
        if x.ndim != 1 or y.ndim != 1 or x.size != image.shape[1] or y.size != image.shape[0]:
            raise ValueError("x/y coordinates do not match image dimensions")
        return {
            "path": path,
            "image": image,
            "mask": mask,
            "x": x,
            "y": y,
            "x_attrs": dict(ds[x_name].attrs),
            "y_attrs": dict(ds[y_name].attrs),
            "image_attrs": dict(image_data.attrs),
            "mask_attrs": dict(mask_data.attrs),
            "dataset_attrs": dict(ds.attrs),
            "timestamp": get_timestamp(ds),
        }


def validate_inputs(frame1: dict, frame2: dict) -> None:
    """Require expected raster size, binary masks, valid data, and same grid."""
    if frame1["image"].shape != frame2["image"].shape:
        raise ValueError(f"image shapes differ: {frame1['image'].shape} vs {frame2['image'].shape}")
    if frame1["image"].shape != EXPECTED_SHAPE:
        raise ValueError(f"expected input shape {EXPECTED_SHAPE}, got {frame1['image'].shape}")
    for label, frame in (("frame1", frame1), ("frame2", frame2)):
        image, mask = frame["image"], frame["mask"]
        if mask.shape != image.shape:
            raise ValueError(f"{label} mask shape does not match its image")
        if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{label} valid_mask must contain only 0 and 1")
        if not (mask == 1).any():
            raise ValueError(f"{label} is entirely invalid")
        valid = mask == 1
        if not np.isfinite(image[valid]).all():
            raise ValueError(f"{label} has non-finite normalized values marked as valid")
        if ((image[valid] < -1e-6) | (image[valid] > 1.0 + 1e-6)).any():
            raise ValueError(f"{label} valid normalized values fall outside [0, 1]")
    for coord in ("x", "y"):
        values1, values2 = frame1[coord], frame2[coord]
        if values1.shape != values2.shape or not np.allclose(values1, values2, rtol=0.0, atol=1e-7):
            raise ValueError(f"{coord} coordinates differ; frames are not on the same spatial grid")
        attrs_name = f"{coord}_attrs"
        if frame1[attrs_name].get("units", "") != frame2[attrs_name].get("units", ""):
            raise ValueError(f"{coord} coordinate units differ")


def create_common_mask(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Pixels are observations only when both input masks say they are valid."""
    return (mask1 == 1) & (mask2 == 1)


def calculate_farneback_flow(
    image1: np.ndarray,
    image2: np.ndarray,
    common_valid_mask: np.ndarray,
    parameters: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate dense flow and set non-common pixels to NaN in saved products."""
    # Farneback has no validity-mask argument. Give it a shared neutral value
    # outside the common region, then exclude those estimates from all results.
    # OpenCV accepts CV_32FC1; scale normalized intensities to the familiar
    # 0-255 image range so small [0, 1] gradients are not numerically tiny.
    # The neutral invalid fill is scaled with the image (0.5 -> 127.5).
    cv_image1 = (np.where(common_valid_mask, image1, 0.5) * 255.0).astype(np.float32)
    cv_image2 = (np.where(common_valid_mask, image2, 0.5) * 255.0).astype(np.float32)
    raw_flow = cv2.calcOpticalFlowFarneback(
        cv_image1,
        cv_image2,
        None,
        parameters["pyr_scale"],
        parameters["levels"],
        parameters["winsize"],
        parameters["iterations"],
        parameters["poly_n"],
        parameters["poly_sigma"],
        parameters["flags"],
    )
    if raw_flow.shape != (*image1.shape, 2):
        raise ValueError(f"OpenCV returned unexpected flow shape {raw_flow.shape}")
    finite_flow = np.isfinite(raw_flow).all(axis=2)
    valid_flow = common_valid_mask & finite_flow
    if not valid_flow.any():
        raise ValueError("no finite flow vectors were produced in common-valid pixels")

    flow = raw_flow.astype(np.float32, copy=True)
    flow[~valid_flow] = np.nan
    magnitude = np.hypot(flow[..., 0], flow[..., 1]).astype(np.float32)
    angle = np.arctan2(flow[..., 1], flow[..., 0]).astype(np.float32)
    magnitude[~valid_flow] = np.nan
    angle[~valid_flow] = np.nan
    return flow, magnitude, angle, valid_flow


def calculate_flow_statistics(flow: np.ndarray, magnitude: np.ndarray, valid_mask: np.ndarray) -> dict:
    """Summarize only finite flow vectors at common-valid pixels."""
    valid = valid_mask & np.isfinite(flow).all(axis=2) & np.isfinite(magnitude)
    if not valid.any():
        raise ValueError("no valid flow vectors available for statistics")
    dx = flow[..., 0][valid]
    dy = flow[..., 1][valid]
    mag = magnitude[valid]
    return {
        "common_valid_pixels": int(valid_mask.sum()),
        "finite_flow_pixels": int(valid.sum()),
        "dx_mean": float(dx.mean()),
        "dx_median": float(np.median(dx)),
        "dx_min": float(dx.min()),
        "dx_max": float(dx.max()),
        "dy_mean": float(dy.mean()),
        "dy_median": float(np.median(dy)),
        "dy_min": float(dy.min()),
        "dy_max": float(dy.max()),
        "magnitude_mean": float(mag.mean()),
        "magnitude_median": float(np.median(mag)),
        "magnitude_max": float(mag.max()),
    }


def make_metadata(frame1: dict, frame2: dict, parameters: dict) -> dict:
    timestamp1, timestamp2 = frame1["timestamp"], frame2["timestamp"]
    try:
        delta_seconds = (timestamp2 - timestamp1).total_seconds() if timestamp1 and timestamp2 else None
    except TypeError:
        delta_seconds = None
    return {
        "frame1_filename": frame1["path"].name,
        "frame2_filename": frame2["path"].name,
        "frame1_timestamp": timestamp1.isoformat() if timestamp1 else None,
        "frame2_timestamp": timestamp2.isoformat() if timestamp2 else None,
        "time_difference_seconds": delta_seconds,
        "time_difference_minutes": delta_seconds / 60.0 if delta_seconds is not None else None,
        "farneback_parameters": parameters,
        "flow_units": "pixel displacement between input frames",
        "algorithm": "OpenCV Farneback dense optical flow; baseline experiment only",
    }


def save_flow(
    output_dir: Path,
    flow: np.ndarray,
    magnitude: np.ndarray,
    angle: np.ndarray,
    common_valid_mask: np.ndarray,
    frame1: dict,
    metadata: dict,
) -> tuple[Path, Path]:
    """Save required compressed NPZ plus a readable NetCDF representation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "flow_frame1_to_frame2.npz"
    nc_path = output_dir / "flow_frame1_to_frame2.nc"
    np.savez_compressed(
        npz_path,
        flow=flow,
        magnitude=magnitude,
        angle=angle,
        common_valid_mask=common_valid_mask.astype(np.uint8),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    dataset = xr.Dataset(
        {
            "dx": (("y", "x"), flow[..., 0], {"long_name": "Eastward image displacement", "units": "pixels"}),
            "dy": (("y", "x"), flow[..., 1], {"long_name": "Downward image displacement", "units": "pixels"}),
            "magnitude": (("y", "x"), magnitude, {"long_name": "Flow vector magnitude", "units": "pixels"}),
            "angle": (("y", "x"), angle, {"long_name": "Flow vector angle from positive x axis", "units": "radians"}),
            "common_valid_mask": (("y", "x"), common_valid_mask.astype(np.uint8), {"flag_values": np.array([0, 1], dtype=np.uint8), "flag_meanings": "invalid valid"}),
        },
        coords={
            "x": ("x", frame1["x"], frame1["x_attrs"]),
            "y": ("y", frame1["y"], frame1["y_attrs"]),
        },
        attrs={
            "title": "Baseline Farneback optical flow from frame 1 to frame 2",
            "frame1_filename": metadata["frame1_filename"],
            "frame2_filename": metadata["frame2_filename"],
            "frame1_timestamp": metadata["frame1_timestamp"] or "unavailable",
            "frame2_timestamp": metadata["frame2_timestamp"] or "unavailable",
            "time_difference_seconds": metadata["time_difference_seconds"] if metadata["time_difference_seconds"] is not None else "unavailable",
            "farneback_parameters_json": json.dumps(metadata["farneback_parameters"]),
            "note": "Baseline experiment only; flow vectors are not asserted to be scientifically accurate.",
        },
    )
    dataset.to_netcdf(nc_path)
    return npz_path, nc_path


def plot_magnitude(path: Path, magnitude: np.ndarray, mask: np.ndarray) -> None:
    shown = np.ma.masked_where(~mask, magnitude)
    fig, axis = plt.subplots(figsize=(9, 8))
    axis.set_facecolor("black")
    image = axis.imshow(shown, origin="upper", cmap="magma")
    fig.colorbar(image, ax=axis, label="Flow magnitude (pixels between frames)")
    axis.set_title("Farneback flow magnitude")
    axis.set_xlabel("Column (pixels)")
    axis.set_ylabel("Row (pixels)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_direction(path: Path, angle: np.ndarray, mask: np.ndarray) -> None:
    shown = np.ma.masked_where(~mask, angle)
    fig, axis = plt.subplots(figsize=(9, 8))
    axis.set_facecolor("black")
    image = axis.imshow(shown, origin="upper", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    colorbar = fig.colorbar(image, ax=axis, label="Flow direction (radians)")
    colorbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axis.set_title("Farneback flow direction")
    axis.set_xlabel("Column (pixels)")
    axis.set_ylabel("Row (pixels)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_quiver(
    path: Path,
    background: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    step: int,
    title: str,
    background_label: str,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(np.ma.masked_where(~mask, background), origin="upper", cmap="gray", vmin=0, vmax=1)
    fig.colorbar(image, ax=axis, label=background_label)
    rows, columns = np.mgrid[0 : mask.shape[0] : step, 0 : mask.shape[1] : step]
    arrows_valid = mask[::step, ::step]
    # The estimated flow is subpixel in many places. Enlarge vectors only in
    # the plots so their direction is visible; the saved arrays remain true
    # pixel displacements.
    dx = flow[::step, ::step, 0] * QUIVER_DISPLAY_SCALE
    dy = flow[::step, ::step, 1] * QUIVER_DISPLAY_SCALE
    axis.quiver(
        columns[arrows_valid], rows[arrows_valid], dx[arrows_valid], dy[arrows_valid],
        color="cyan", angles="xy", scale_units="xy", scale=1, width=0.0022,
    )
    axis.set_title(f"{title} (one arrow every {step} pixels; vectors shown at {QUIVER_DISPLAY_SCALE:g}x scale)")
    axis.set_xlabel("Column (pixels)")
    axis.set_ylabel("Row (pixels)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame1", required=True, type=Path, help="First 1024x1024 prepared flow-input NetCDF")
    parser.add_argument("--frame2", required=True, type=Path, help="Second 1024x1024 prepared flow-input NetCDF")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for flow arrays and figures")
    parser.add_argument("--pyr-scale", type=float, default=DEFAULT_PARAMETERS["pyr_scale"], help="Pyramid scale between 0 and 1")
    parser.add_argument("--levels", type=int, default=DEFAULT_PARAMETERS["levels"], help="Number of pyramid levels")
    parser.add_argument("--winsize", type=int, default=DEFAULT_PARAMETERS["winsize"], help="Averaging window size")
    parser.add_argument("--iterations", type=int, default=DEFAULT_PARAMETERS["iterations"], help="Iterations per pyramid level")
    parser.add_argument("--poly-n", type=int, default=DEFAULT_PARAMETERS["poly_n"], help="Polynomial expansion neighborhood size")
    parser.add_argument("--poly-sigma", type=float, default=DEFAULT_PARAMETERS["poly_sigma"], help="Gaussian sigma for polynomial expansion")
    parser.add_argument("--flags", type=int, default=DEFAULT_PARAMETERS["flags"], help="OpenCV Farneback flags")
    parser.add_argument("--quiver-step", type=int, default=25, help="Arrow spacing in pixels for both vector plots")
    args = parser.parse_args()

    if not 0 < args.pyr_scale < 1 or args.levels < 1 or args.winsize < 1 or args.iterations < 1 or args.poly_n < 5 or args.poly_sigma <= 0:
        parser.error("Farneback parameters must be positive; pyr-scale must be between 0 and 1 and poly-n must be at least 5")
    if args.quiver_step < 1:
        parser.error("--quiver-step must be at least 1")

    frame1_path = resolve_path(args.frame1)
    frame2_path = resolve_path(args.frame2)
    output_dir = resolve_path(args.output_dir)
    parameters = {
        "pyr_scale": args.pyr_scale,
        "levels": args.levels,
        "winsize": args.winsize,
        "iterations": args.iterations,
        "poly_n": args.poly_n,
        "poly_sigma": args.poly_sigma,
        "flags": args.flags,
    }

    print("Loading the two requested NetCDF frames...")
    frame1 = load_flow_frame(frame1_path)
    frame2 = load_flow_frame(frame2_path)
    validate_inputs(frame1, frame2)
    common_mask = create_common_mask(frame1["mask"], frame2["mask"])
    total_pixels = int(common_mask.size)
    common_count = int(common_mask.sum())
    print(f"Frame 1 valid pixels: {int((frame1['mask'] == 1).sum()):,}")
    print(f"Frame 2 valid pixels: {int((frame2['mask'] == 1).sum()):,}")
    print(f"Common valid pixels: {common_count:,}")
    print(f"Common valid percentage: {100.0 * common_count / total_pixels:.2f}%")
    if common_count == 0:
        raise SystemExit("The two frames have no common valid pixels; flow cannot be summarized.")

    print("Calculating baseline Farneback dense optical flow...")
    flow, magnitude, angle, flow_mask = calculate_farneback_flow(
        frame1["image"], frame2["image"], common_mask, parameters
    )
    stats = calculate_flow_statistics(flow, magnitude, flow_mask)
    time_metadata = make_metadata(frame1, frame2, parameters)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path, nc_path = save_flow(output_dir, flow, magnitude, angle, common_mask, frame1, time_metadata)

    magnitude_path = output_dir / "flow_magnitude.png"
    direction_path = output_dir / "flow_direction.png"
    quiver_path = output_dir / "flow_quiver.png"
    overlay_path = output_dir / "flow_overlay.png"
    plot_magnitude(magnitude_path, magnitude, flow_mask)
    plot_direction(direction_path, angle, flow_mask)
    plot_quiver(
        quiver_path, magnitude, flow, flow_mask, args.quiver_step,
        "Flow vectors on magnitude", "Flow magnitude (pixels)",
    )
    plot_quiver(
        overlay_path, np.where(common_mask, frame1["image"], np.nan), flow,
        flow_mask, args.quiver_step, "Flow overlay on frame 1",
        "Frame 1 normalized brightness temperature",
    )

    expected_flow_shape = (*EXPECTED_SHAPE, 2)
    if flow.shape != expected_flow_shape or magnitude.shape != EXPECTED_SHAPE or angle.shape != EXPECTED_SHAPE or flow_mask.shape != EXPECTED_SHAPE:
        raise SystemExit("Output validation failed: unexpected flow, magnitude, angle, or mask shape")
    if not np.isfinite(flow[flow_mask]).all() or not np.isfinite(magnitude[flow_mask]).all() or not np.isfinite(angle[flow_mask]).all():
        raise SystemExit("Output validation failed: valid flow contains NaN or infinity")
    output_paths = [npz_path, nc_path, magnitude_path, direction_path, quiver_path, overlay_path]
    missing_outputs = [path for path in output_paths if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise SystemExit("Output validation failed; missing/empty: " + ", ".join(path.name for path in missing_outputs))

    print("\nFLOW SUMMARY")
    print(f"Flow shape: {flow.shape}")
    print(f"Common valid pixel count: {stats['common_valid_pixels']:,}")
    print("Vector units: pixel displacement between input frames")
    print(f"dx: mean {stats['dx_mean']:.6f}, median {stats['dx_median']:.6f}, min {stats['dx_min']:.6f}, max {stats['dx_max']:.6f}")
    print(f"dy: mean {stats['dy_mean']:.6f}, median {stats['dy_median']:.6f}, min {stats['dy_min']:.6f}, max {stats['dy_max']:.6f}")
    print(f"Magnitude: mean {stats['magnitude_mean']:.6f}, median {stats['magnitude_median']:.6f}, max {stats['magnitude_max']:.6f} pixels")
    print("\nFARNEBACK OPTICAL FLOW COMPLETE")
    print("=" * 48)
    print(f"Frame 1: {frame1_path.name}")
    print(f"Frame 2: {frame2_path.name}")
    print(f"Image shape: {EXPECTED_SHAPE[0]} x {EXPECTED_SHAPE[1]}")
    minutes = time_metadata["time_difference_minutes"]
    print(f"Time difference: {minutes:.3f} minutes" if minutes is not None else "Time difference: unavailable")
    print(f"Common valid pixels: {common_count:,} ({100.0 * common_count / total_pixels:.2f}%)")
    print(f"Mean dx: {stats['dx_mean']:.6f} pixels")
    print(f"Mean dy: {stats['dy_mean']:.6f} pixels")
    print(f"Mean magnitude: {stats['magnitude_mean']:.6f} pixels")
    print(f"Maximum magnitude: {stats['magnitude_max']:.6f} pixels")
    print("Outputs:")
    for path in output_paths:
        print(f"  {path}")
    print("Baseline results require visual and scientific validation; code execution alone does not establish accuracy.")


if __name__ == "__main__":
    main()
