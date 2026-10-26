"""Measure whether a saved flow field improves frame-to-frame alignment.

This evaluates a baseline flow field, not the physical accuracy of cloud
motion. Satellite imagery can change through cloud evolution, parallax,
illumination, and atmospheric effects.
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
from skimage.metrics import structural_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_VARIABLE = "normalized_brightness_temperature"
MASK_VARIABLE = "valid_mask"
EXPECTED_SHAPE = (1024, 1024)
SSIM_WINDOW = 7
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "optical_flow" / "validation"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_timestamp(value: object) -> datetime | None:
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


def dataset_timestamp(ds: xr.Dataset) -> datetime | None:
    for name in ("t", "time"):
        if name in ds.coords:
            values = np.asarray(ds[name].values)
            if values.size:
                result = parse_timestamp(values.reshape(-1)[0])
                if result is not None:
                    return result
    for name in ("frame1_timestamp", "frame2_timestamp", "time_coverage_start", "timestamp"):
        result = parse_timestamp(ds.attrs.get(name))
        if result is not None:
            return result
    return None


def load_frame(path: Path) -> dict:
    """Read just the normalized image and validity mask from one frame."""
    if not path.is_file():
        raise FileNotFoundError(f"frame file does not exist: {path}")
    with xr.open_dataset(path) as ds:
        missing = [name for name in (NORMALIZED_VARIABLE, MASK_VARIABLE) if name not in ds]
        if missing:
            raise KeyError(f"{path.name} is missing: {', '.join(missing)}")
        image_data = ds[NORMALIZED_VARIABLE]
        mask_data = ds[MASK_VARIABLE]
        if image_data.ndim != 2 or image_data.dims != ("y", "x"):
            raise ValueError(f"{NORMALIZED_VARIABLE} must have dimensions ('y', 'x'); got {image_data.dims}")
        if mask_data.dims != image_data.dims or mask_data.shape != image_data.shape:
            raise ValueError("valid_mask shape/dimensions do not match normalized image")
        return {
            "path": path,
            "image": np.asarray(image_data.load().values, dtype=np.float32),
            "mask": np.asarray(mask_data.load().values),
            "x": np.asarray(ds["x"].values) if "x" in ds.coords else None,
            "y": np.asarray(ds["y"].values) if "y" in ds.coords else None,
            "x_attrs": dict(ds["x"].attrs) if "x" in ds.coords else {},
            "y_attrs": dict(ds["y"].attrs) if "y" in ds.coords else {},
            "timestamp": dataset_timestamp(ds),
        }


def load_flow(path: Path) -> dict:
    """Read flow components, magnitude, and their common-valid mask."""
    if not path.is_file():
        raise FileNotFoundError(f"flow file does not exist: {path}")
    variables = ("dx", "dy", "magnitude", "common_valid_mask")
    with xr.open_dataset(path) as ds:
        missing = [name for name in variables if name not in ds]
        if missing:
            raise KeyError(f"{path.name} is missing: {', '.join(missing)}")
        arrays = {name: np.asarray(ds[name].load().values) for name in variables}
        shapes = {name: value.shape for name, value in arrays.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"flow arrays have incompatible shapes: {shapes}")
        if arrays["dx"].ndim != 2:
            raise ValueError(f"flow components must be 2D rasters; got {shapes['dx']}")
        mask = arrays["common_valid_mask"]
        if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError("flow common_valid_mask must contain only 0 and 1")
        return {
            **arrays,
            "x": np.asarray(ds["x"].values) if "x" in ds.coords else None,
            "y": np.asarray(ds["y"].values) if "y" in ds.coords else None,
            "x_attrs": dict(ds["x"].attrs) if "x" in ds.coords else {},
            "y_attrs": dict(ds["y"].attrs) if "y" in ds.coords else {},
            "attrs": dict(ds.attrs),
        }


def validate_inputs(frame1: dict, frame2: dict, flow: dict) -> None:
    """Check dimensions, binary masks, common pixels, and shared grids."""
    if frame1["image"].shape != frame2["image"].shape:
        raise ValueError("frame image shapes do not match")
    if frame1["image"].shape != EXPECTED_SHAPE:
        raise ValueError(f"expected image shape {EXPECTED_SHAPE}; got {frame1['image'].shape}")
    if frame1["mask"].shape != frame1["image"].shape or frame2["mask"].shape != frame2["image"].shape:
        raise ValueError("a valid_mask shape does not match its image")
    if flow["dx"].shape != frame1["image"].shape:
        raise ValueError("flow arrays do not match frame image shapes")
    for label, frame in (("frame1", frame1), ("frame2", frame2)):
        mask = frame["mask"]
        if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{label} valid_mask must contain only 0 and 1")
        valid = mask == 1
        if not valid.any():
            raise ValueError(f"{label} is entirely invalid")
        if not np.isfinite(frame["image"][valid]).all():
            raise ValueError(f"{label} contains non-finite values marked as valid")
    if frame1["x"] is None or frame1["y"] is None:
        raise ValueError("frame 1 is missing x/y coordinates")
    for coord in ("x", "y"):
        for pair_name, first, second in (
            ("frame x/y grids", frame1[coord], frame2[coord]),
            ("flow/frame grids", frame1[coord], flow[coord]),
        ):
            if first is None or second is None or first.shape != second.shape:
                raise ValueError(f"{pair_name} have incompatible {coord} coordinates")
            if not np.allclose(first, second, rtol=0.0, atol=1e-7):
                raise ValueError(f"{pair_name} differ in the {coord} coordinate")
        attrs_name = f"{coord}_attrs"
        if frame1[attrs_name].get("units", "") != frame2[attrs_name].get("units", ""):
            raise ValueError(f"frame {coord} coordinate units do not match")
    flow_mask = flow["common_valid_mask"] == 1
    expected_common = (frame1["mask"] == 1) & (frame2["mask"] == 1)
    if not np.array_equal(flow_mask, expected_common):
        raise ValueError("flow common_valid_mask does not match the intersection of the input frame masks")
    if not flow_mask.any():
        raise ValueError("flow common_valid_mask has no valid pixels")
    finite_flow = np.isfinite(flow["dx"]) & np.isfinite(flow["dy"])
    if not finite_flow[flow_mask].all():
        raise ValueError("flow contains non-finite dx/dy values in common-valid pixels")


def warp_frame(image: np.ndarray, dx: np.ndarray, dy: np.ndarray, source_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-warp Frame 1 using its source-to-target dx/dy field.

    Farneback's stored field at source pixel (x, y) points to target
    (x + dx, y + dy). cv2.remap requires a destination-to-source map. We
    therefore numerically invert that forward field: initialize the source
    location as target - flow(target), then iterate source = target - flow(source).
    This keeps the dx/dy direction explicit and avoids silently swapping axes.
    """
    height, width = image.shape
    target_x, target_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    finite = np.isfinite(dx) & np.isfinite(dy)
    dx_clean = np.where(finite, dx, 0.0).astype(np.float32)
    dy_clean = np.where(finite, dy, 0.0).astype(np.float32)
    map_x = target_x - dx_clean
    map_y = target_y - dy_clean

    # Fixed-point inversion of the source-to-target displacement field.
    for _ in range(5):
        sampled_dx = cv2.remap(dx_clean, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sampled_dy = cv2.remap(dy_clean, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        map_x = target_x - sampled_dx
        map_y = target_y - sampled_dy

    warped = cv2.remap(
        image.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.5,
    )
    warped_mask = cv2.remap(
        source_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) == 1
    in_bounds = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return warped, warped_mask & in_bounds


def create_evaluation_mask(
    frame1: dict,
    frame2: dict,
    flow: dict,
    warped: np.ndarray,
    warped_source_valid: np.ndarray,
) -> np.ndarray:
    """Keep common observations and fully valid 7x7 SSIM neighborhoods."""
    common = flow["common_valid_mask"] == 1
    evaluation = (
        common
        & (frame1["mask"] == 1)
        & (frame2["mask"] == 1)
        & warped_source_valid
        & np.isfinite(warped)
    )
    # SSIM uses a 7x7 local window. Eroding by its radius makes every metric
    # use pixels whose SSIM neighborhood contains only evaluation pixels.
    kernel = np.ones((SSIM_WINDOW, SSIM_WINDOW), dtype=np.uint8)
    evaluation = cv2.erode(evaluation.astype(np.uint8), kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0) == 1
    return evaluation


def calculate_metrics(reference: np.ndarray, comparison: np.ndarray, evaluation_mask: np.ndarray) -> dict:
    """Compute masked MSE, MAE, PSNR, and SSIM for images in [0, 1]."""
    if not evaluation_mask.any():
        raise ValueError("evaluation mask is empty")
    ref = reference[evaluation_mask].astype(np.float64)
    comp = comparison[evaluation_mask].astype(np.float64)
    if not np.isfinite(ref).all() or not np.isfinite(comp).all():
        raise ValueError("evaluation region contains non-finite image values")
    difference = ref - comp
    mse = float(np.mean(difference * difference))
    mae = float(np.mean(np.abs(difference)))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))

    # Obtain the local SSIM map, then average only at evaluation pixels.
    _, ssim_map = structural_similarity(
        reference.astype(np.float32), comparison.astype(np.float32),
        data_range=1.0, win_size=SSIM_WINDOW, full=True,
    )
    ssim = float(np.mean(ssim_map[evaluation_mask]))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def save_warped_frame(path: Path, warped: np.ndarray, evaluation_mask: np.ndarray) -> None:
    shown = np.ma.masked_where(~evaluation_mask, warped)
    fig, axis = plt.subplots(figsize=(9, 8))
    axis.set_facecolor("black")
    image = axis.imshow(shown, origin="upper", cmap="gray", vmin=0, vmax=1)
    fig.colorbar(image, ax=axis, label="Normalized brightness temperature")
    axis.set_title("Frame 1 forward-warped toward Frame 2")
    axis.set_xlabel("Column (pixels)")
    axis.set_ylabel("Row (pixels)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_comparison_plot(
    path: Path,
    frame1: np.ndarray,
    frame2: np.ndarray,
    warped: np.ndarray,
    evaluation_mask: np.ndarray,
) -> None:
    """Save a side-by-side comparison restricted to the evaluation region."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (frame1, frame2, warped),
        ("Original Frame 1", "Actual Frame 2", "Warped Frame 1"),
    ):
        shown = np.ma.masked_where(~evaluation_mask, values)
        axis.set_facecolor("black")
        image = axis.imshow(shown, origin="upper", cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xlabel("Column (pixels)")
        axis.set_ylabel("Row (pixels)")
    fig.colorbar(image, ax=axes, label="Normalized brightness temperature", shrink=0.85)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_warped_netcdf(path: Path, warped: np.ndarray, evaluation_mask: np.ndarray, frame1: dict, frame2: dict, flow_path: Path) -> None:
    output = xr.Dataset(
        {
            "warped_normalized_brightness_temperature": (("y", "x"), warped.astype(np.float32), {"units": "1", "invalid_fill_value": 0.5}),
            "evaluation_mask": (("y", "x"), evaluation_mask.astype(np.uint8), {"flag_values": np.array([0, 1], dtype=np.uint8), "flag_meanings": "excluded evaluated"}),
        },
        coords={
            "x": ("x", frame1["x"], frame1["x_attrs"]),
            "y": ("y", frame1["y"], frame1["y_attrs"]),
        },
        attrs={
            "frame1_filename": frame1["path"].name,
            "frame2_filename": frame2["path"].name,
            "flow_file": flow_path.name,
            "warp_convention": "Flow is source-to-target; inverse coordinates are solved for cv2.remap.",
        },
    )
    output.to_netcdf(path)


def json_number(value: float) -> float | str:
    return value if np.isfinite(value) else "Infinity"


def save_results(path: Path, results: dict) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2, allow_nan=False)
        output.write("\n")


def print_metric_comparison(original: dict, warped: dict) -> dict:
    """Print metric values and say whether warping improved each metric."""
    higher_is_better = {"mse": False, "mae": False, "psnr": True, "ssim": True}
    labels = {"mse": "MSE", "mae": "MAE", "psnr": "PSNR", "ssim": "SSIM"}
    improved = {}
    print("\nMETRIC COMPARISON")
    print(f"{'Metric':<10} {'Original Frame 1 -> Frame 2':>30} {'Warped Frame 1 -> Frame 2':>30}")
    for key, label in labels.items():
        before, after = original[key], warped[key]
        relation = after > before if higher_is_better[key] else after < before
        improved[key] = bool(relation)
        before_text = "Infinity" if np.isinf(before) else f"{before:.8f}"
        after_text = "Infinity" if np.isinf(after) else f"{after:.8f}"
        print(f"{label:<10} {before_text:>30} {after_text:>30}")
        print(f"  {'Improved' if relation else 'Did not improve'} ({'higher' if higher_is_better[key] else 'lower'} is better)")
    return improved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame1", required=True, type=Path, help="First prepared flow-input NetCDF")
    parser.add_argument("--frame2", required=True, type=Path, help="Second prepared flow-input NetCDF")
    parser.add_argument("--flow", required=True, type=Path, help="Farneback flow NetCDF")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for validation files")
    args = parser.parse_args()

    frame1_path = resolve_path(args.frame1)
    frame2_path = resolve_path(args.frame2)
    flow_path = resolve_path(args.flow)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading requested frames and flow arrays...")
    frame1 = load_frame(frame1_path)
    frame2 = load_frame(frame2_path)
    flow = load_flow(flow_path)
    validate_inputs(frame1, frame2, flow)

    # Stored dx/dy is forward flow at source pixels: target = source + flow.
    # warp_frame explicitly inverts it to the map direction cv2.remap expects.
    warped, warped_source_valid = warp_frame(
        frame1["image"], flow["dx"], flow["dy"], frame1["mask"]
    )
    evaluation_mask = create_evaluation_mask(frame1, frame2, flow, warped, warped_source_valid)
    evaluation_count = int(evaluation_mask.sum())
    evaluation_percentage = 100.0 * evaluation_count / evaluation_mask.size
    if evaluation_count == 0:
        raise SystemExit("No valid comparison pixels remain after flow, source-mask, and SSIM-window checks.")
    print(f"Evaluation pixels: {evaluation_count:,}")
    print(f"Evaluation percentage: {evaluation_percentage:.2f}%")

    original_metrics = calculate_metrics(frame1["image"], frame2["image"], evaluation_mask)
    warped_metrics = calculate_metrics(warped, frame2["image"], evaluation_mask)
    improved = print_metric_comparison(original_metrics, warped_metrics)

    warped_for_save = np.where(evaluation_mask, warped, 0.5).astype(np.float32)
    warped_png = output_dir / "warped_frame1.png"
    comparison_png = output_dir / "flow_warp_comparison.png"
    warped_nc = output_dir / "warped_frame1.nc"
    metrics_json = output_dir / "validation_metrics.json"
    save_warped_frame(warped_png, warped_for_save, evaluation_mask)
    create_comparison_plot(comparison_png, frame1["image"], frame2["image"], warped_for_save, evaluation_mask)
    save_warped_netcdf(warped_nc, warped_for_save, evaluation_mask, frame1, frame2, flow_path)

    timestamp1, timestamp2 = frame1["timestamp"], frame2["timestamp"]
    try:
        time_difference_seconds = (timestamp2 - timestamp1).total_seconds() if timestamp1 and timestamp2 else None
    except TypeError:
        time_difference_seconds = None
    results = {
        "frame1": frame1_path.name,
        "frame2": frame2_path.name,
        "flow_file": flow_path.name,
        "evaluation_pixel_count": evaluation_count,
        "evaluation_percentage": evaluation_percentage,
        "original_mse": json_number(original_metrics["mse"]),
        "warped_mse": json_number(warped_metrics["mse"]),
        "original_mae": json_number(original_metrics["mae"]),
        "warped_mae": json_number(warped_metrics["mae"]),
        "original_psnr": json_number(original_metrics["psnr"]),
        "warped_psnr": json_number(warped_metrics["psnr"]),
        "original_ssim": json_number(original_metrics["ssim"]),
        "warped_ssim": json_number(warped_metrics["ssim"]),
        "time_difference_seconds": time_difference_seconds,
        "time_difference_minutes": time_difference_seconds / 60.0 if time_difference_seconds is not None else None,
        "flow_warp_convention": "Forward displacement: target=(source_x+dx, source_y+dy); inverted for cv2.remap destination-to-source coordinates.",
        "ssim_evaluation_note": f"Evaluation mask eroded by {SSIM_WINDOW // 2} pixels so each {SSIM_WINDOW}x{SSIM_WINDOW} SSIM window contains evaluated pixels.",
        "metric_improved": improved,
        "scientific_caveat": "Improved image metrics do not prove physically correct cloud motion; cloud evolution, parallax, illumination, and atmospheric effects remain.",
    }
    save_results(metrics_json, results)

    output_paths = (warped_png, comparison_png, warped_nc, metrics_json)
    if any(not path.is_file() or path.stat().st_size == 0 for path in output_paths):
        raise SystemExit("Validation output check failed: at least one output is missing or empty")
    print("\nOPTICAL FLOW VALIDATION COMPLETE")
    print("=" * 40)
    print(f"Original Frame 1 -> Frame 2: MSE {original_metrics['mse']:.8f}, MAE {original_metrics['mae']:.8f}, PSNR {original_metrics['psnr']:.4f}, SSIM {original_metrics['ssim']:.6f}")
    print(f"Warped Frame 1 -> Frame 2:   MSE {warped_metrics['mse']:.8f}, MAE {warped_metrics['mae']:.8f}, PSNR {warped_metrics['psnr']:.4f}, SSIM {warped_metrics['ssim']:.6f}")
    print(f"Evaluation pixels: {evaluation_count:,} ({evaluation_percentage:.2f}%)")
    print("Metric interpretation: " + "; ".join(f"{key.upper()} {'improved' if value else 'did not improve'}" for key, value in improved.items()))
    print("Outputs:")
    for path in output_paths:
        print(f"  {path}")
    print("Metric improvements indicate better image alignment on this evaluation region, not physical accuracy of cloud motion.")


if __name__ == "__main__":
    main()
