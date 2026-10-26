"""Evaluate the existing Farneback baseline on consecutive flow-input frames."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:  # Work both as a script and as a module.
    from . import optical_flow_farneback as farneback
    from . import validate_optical_flow as validation
    from .check_frame_pair import COORDINATE_ATOL, check_channel
except ImportError:
    import optical_flow_farneback as farneback
    import validate_optical_flow as validation
    from check_frame_pair import COORDINATE_ATOL, check_channel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "flow_input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "optical_flow"
CSV_COLUMNS = (
    "pair", "frame1", "frame2", "timestamp1", "timestamp2", "time_difference_minutes",
    "evaluation_pixels", "evaluation_percentage", "mse_before", "mse_after",
    "mae_before", "mae_after", "psnr_before", "psnr_after", "ssim_before", "ssim_after",
    "mean_dx", "mean_dy", "mean_flow_magnitude", "max_flow_magnitude", "status", "error",
)
METRIC_NAMES = ("mse", "mae", "psnr", "ssim")


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


def read_frame_metadata(path: Path) -> dict:
    """Read only metadata and small coordinate vectors during discovery."""
    try:
        with xr.open_dataset(path) as ds:
            timestamp = None
            for name in ("t", "time"):
                if name in ds.coords:
                    values = np.asarray(ds[name].values)
                    if values.size:
                        timestamp = parse_timestamp(values.reshape(-1)[0])
                        if timestamp is not None:
                            break
            if timestamp is None:
                for name in ("time_coverage_start", "timestamp", "start_time"):
                    timestamp = parse_timestamp(ds.attrs.get(name))
                    if timestamp is not None:
                        break
            attrs = dict(ds.attrs)
            is_c13, channel_reason = check_channel(path, attrs)
            image_shape = tuple(ds[farneback.NORMALIZED_VARIABLE].shape) if farneback.NORMALIZED_VARIABLE in ds else None
            has_mask = farneback.MASK_VARIABLE in ds
            x = np.asarray(ds["x"].values) if "x" in ds.coords else None
            y = np.asarray(ds["y"].values) if "y" in ds.coords else None
            x_units = str(ds["x"].attrs.get("units", "")) if "x" in ds.coords else None
            y_units = str(ds["y"].attrs.get("units", "")) if "y" in ds.coords else None
            error = None
            if image_shape is None:
                error = f"missing {farneback.NORMALIZED_VARIABLE}"
            elif not has_mask:
                error = f"missing {farneback.MASK_VARIABLE}"
            return {
                "path": path,
                "timestamp": timestamp,
                "image_shape": image_shape,
                "has_mask": has_mask,
                "x": x,
                "y": y,
                "x_units": x_units,
                "y_units": y_units,
                "attrs": attrs,
                "channel_ok": is_c13,
                "channel_reason": channel_reason,
                "error": error,
            }
    except Exception as error:
        return {
            "path": path, "timestamp": None, "image_shape": None, "has_mask": False,
            "x": None, "y": None, "x_units": None, "y_units": None, "attrs": {},
            "channel_ok": False, "channel_reason": "metadata could not be read", "error": str(error),
        }


def discover_frames(input_dir: Path) -> list[dict]:
    """Find every flow input and sort by data timestamp, then filename."""
    paths = sorted(input_dir.glob("*_flow_input.nc"))
    frames = [read_frame_metadata(path) for path in paths]
    # Valid timestamps determine chronological order. Untimed files follow in
    # deterministic filename order so no timestamp is fabricated.
    def chronological_key(frame: dict) -> tuple:
        timestamp = frame["timestamp"]
        if timestamp is None:
            return (1, 0.0, frame["path"].name.lower())
        # GOES coordinates commonly carry UTC time without an explicit zone.
        # Treat such naive timestamps as UTC only for sorting their chronology.
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return (0, timestamp.timestamp(), frame["path"].name.lower())

    frames.sort(key=chronological_key)
    return frames


def timestamp_text(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "unavailable"


def time_difference_minutes(first: datetime | None, second: datetime | None) -> float | None:
    if first is None or second is None:
        return None
    try:
        return (second - first).total_seconds() / 60.0
    except TypeError:
        return None


def same_grid(frame1: dict, frame2: dict) -> tuple[bool, str]:
    if frame1["image_shape"] is None or frame2["image_shape"] is None:
        return False, "one or both image shapes are unavailable"
    if frame1["image_shape"] != frame2["image_shape"]:
        return False, f"image dimensions differ: {frame1['image_shape']} vs {frame2['image_shape']}"
    for coord in ("x", "y"):
        first, second = frame1[coord], frame2[coord]
        if first is None or second is None or first.shape != second.shape:
            return False, f"{coord} coordinate dimensions differ or are unavailable"
        if not np.allclose(first, second, rtol=0.0, atol=COORDINATE_ATOL):
            return False, f"{coord} spatial coordinate values differ"
        if frame1[f"{coord}_units"] != frame2[f"{coord}_units"]:
            return False, f"{coord} coordinate units differ"
    return True, "same image dimensions and spatial grid"


def print_frame_inventory(frames: list[dict]) -> None:
    print(f"Frames found: {len(frames)}")
    for index, frame in enumerate(frames, start=1):
        identity = "PASS" if frame["channel_ok"] else f"FAIL ({frame['channel_reason']})"
        print(
            f"  Frame {index}: {frame['path'].name} | "
            f"timestamp: {timestamp_text(frame['timestamp'])} | "
            f"shape: {frame['image_shape']} | GOES-19 ABI C13: {identity}"
        )
        if frame["error"]:
            print(f"    Metadata issue: {frame['error']}")
    print("\nConsecutive frame time differences:")
    for index, (first, second) in enumerate(zip(frames, frames[1:]), start=1):
        minutes = time_difference_minutes(first["timestamp"], second["timestamp"])
        display = f"{minutes:.3f} minutes" if minutes is not None else "unavailable"
        grid_ok, grid_message = same_grid(first, second)
        print(
            f"  Pair {index}: {timestamp_text(first['timestamp'])} -> "
            f"{timestamp_text(second['timestamp'])} ({display}); "
            f"dimensions/grid: {'PASS' if grid_ok else 'FAIL'} ({grid_message})"
        )


def empty_pair_result(pair_number: int, first: dict, second: dict) -> dict:
    return {
        "pair": pair_number,
        "frame1": first["path"].name,
        "frame2": second["path"].name,
        "timestamp1": timestamp_text(first["timestamp"]),
        "timestamp2": timestamp_text(second["timestamp"]),
        "time_difference_minutes": time_difference_minutes(first["timestamp"], second["timestamp"]),
        "evaluation_pixel_count": None,
        "evaluation_percentage": None,
        "original_mse": None,
        "warped_mse": None,
        "original_mae": None,
        "warped_mae": None,
        "original_psnr": None,
        "warped_psnr": None,
        "original_ssim": None,
        "warped_ssim": None,
        "mean_dx": None,
        "mean_dy": None,
        "mean_flow_magnitude": None,
        "max_flow_magnitude": None,
        "status": "failed",
        "error": None,
    }


def save_pair_visuals(
    visual_dir: Path,
    frame1: dict,
    frame2: dict,
    flow: np.ndarray,
    magnitude: np.ndarray,
    angle: np.ndarray,
    flow_mask: np.ndarray,
    warped: np.ndarray,
    evaluation_mask: np.ndarray,
) -> None:
    """Write plots only when --save-visuals is explicitly requested."""
    visual_dir.mkdir(parents=True, exist_ok=True)
    farneback.plot_magnitude(visual_dir / "flow_magnitude.png", magnitude, flow_mask)
    farneback.plot_direction(visual_dir / "flow_direction.png", angle, flow_mask)
    farneback.plot_quiver(
        visual_dir / "flow_quiver.png", magnitude, flow, flow_mask, 25,
        "Flow vectors on magnitude", "Flow magnitude (pixels)",
    )
    farneback.plot_quiver(
        visual_dir / "flow_overlay.png",
        np.where(flow_mask, frame1["image"], np.nan), flow, flow_mask, 25,
        "Flow overlay on frame 1", "Frame 1 normalized brightness temperature",
    )
    validation.save_warped_frame(visual_dir / "warped_frame1.png", warped, evaluation_mask)
    validation.create_comparison_plot(
        visual_dir / "flow_warp_comparison.png", frame1["image"], frame2["image"], warped, evaluation_mask
    )


def process_pair(
    pair_number: int,
    first_meta: dict,
    second_meta: dict,
    output_dir: Path,
    save_visuals: bool,
) -> dict:
    """Process one consecutive pair, returning a complete per-pair record."""
    result = empty_pair_result(pair_number, first_meta, second_meta)
    for label, metadata in (("frame1", first_meta), ("frame2", second_meta)):
        if metadata["error"]:
            raise ValueError(f"{label}: {metadata['error']}")
        if not metadata["channel_ok"]:
            raise ValueError(f"{label}: not a GOES-19 ABI Channel 13 frame ({metadata['channel_reason']})")
    compatible, reason = same_grid(first_meta, second_meta)
    if not compatible:
        raise ValueError(reason)

    # Only these two image pairs are resident; release them after this call.
    frame1 = farneback.load_flow_frame(first_meta["path"])
    frame2 = farneback.load_flow_frame(second_meta["path"])
    farneback.validate_inputs(frame1, frame2)
    common_mask = farneback.create_common_mask(frame1["mask"], frame2["mask"])
    if not common_mask.any():
        raise ValueError("frames have no common-valid pixels")
    parameters = dict(farneback.DEFAULT_PARAMETERS)
    flow, magnitude, angle, flow_mask = farneback.calculate_farneback_flow(
        frame1["image"], frame2["image"], common_mask, parameters
    )

    # Reuse the single-pair validator's exact input checks, warp convention,
    # evaluation-mask erosion, and metric calculations.
    flow_record = {
        "dx": flow[..., 0],
        "dy": flow[..., 1],
        "magnitude": magnitude,
        "common_valid_mask": common_mask.astype(np.uint8),
        "x": frame1["x"],
        "y": frame1["y"],
    }
    validation.validate_inputs(frame1, frame2, flow_record)
    warped, warped_source_valid = validation.warp_frame(
        frame1["image"], flow_record["dx"], flow_record["dy"], frame1["mask"]
    )
    evaluation_mask = validation.create_evaluation_mask(
        frame1, frame2, flow_record, warped, warped_source_valid
    )
    count = int(evaluation_mask.sum())
    if count == 0:
        raise ValueError("evaluation mask is empty")
    original_metrics = validation.calculate_metrics(frame1["image"], frame2["image"], evaluation_mask)
    warped_metrics = validation.calculate_metrics(warped, frame2["image"], evaluation_mask)
    flow_stats = farneback.calculate_flow_statistics(flow, magnitude, flow_mask)

    result.update(
        {
            "evaluation_pixel_count": count,
            "evaluation_percentage": 100.0 * count / evaluation_mask.size,
            "original_mse": validation.json_number(original_metrics["mse"]),
            "warped_mse": validation.json_number(warped_metrics["mse"]),
            "original_mae": validation.json_number(original_metrics["mae"]),
            "warped_mae": validation.json_number(warped_metrics["mae"]),
            "original_psnr": validation.json_number(original_metrics["psnr"]),
            "warped_psnr": validation.json_number(warped_metrics["psnr"]),
            "original_ssim": validation.json_number(original_metrics["ssim"]),
            "warped_ssim": validation.json_number(warped_metrics["ssim"]),
            "mean_dx": flow_stats["dx_mean"],
            "mean_dy": flow_stats["dy_mean"],
            "mean_flow_magnitude": flow_stats["magnitude_mean"],
            "max_flow_magnitude": flow_stats["magnitude_max"],
            "status": "success",
        }
    )
    result["time_difference_seconds"] = (
        result["time_difference_minutes"] * 60.0
        if result["time_difference_minutes"] is not None else None
    )
    if save_visuals:
        pair_dir = output_dir / "visuals" / f"pair_{pair_number:03d}"
        save_pair_visuals(
            pair_dir, frame1, frame2, flow, magnitude, angle, flow_mask,
            np.where(evaluation_mask, warped, 0.5), evaluation_mask,
        )
        result["visual_directory"] = str(pair_dir)
    return result


def save_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")


def csv_row(result: dict) -> dict:
    return {
        "pair": result["pair"],
        "frame1": result["frame1"],
        "frame2": result["frame2"],
        "timestamp1": result["timestamp1"],
        "timestamp2": result["timestamp2"],
        "time_difference_minutes": result.get("time_difference_minutes"),
        "evaluation_pixels": result.get("evaluation_pixel_count"),
        "evaluation_percentage": result.get("evaluation_percentage"),
        "mse_before": result.get("original_mse"),
        "mse_after": result.get("warped_mse"),
        "mae_before": result.get("original_mae"),
        "mae_after": result.get("warped_mae"),
        "psnr_before": result.get("original_psnr"),
        "psnr_after": result.get("warped_psnr"),
        "ssim_before": result.get("original_ssim"),
        "ssim_after": result.get("warped_ssim"),
        "mean_dx": result.get("mean_dx"),
        "mean_dy": result.get("mean_dy"),
        "mean_flow_magnitude": result.get("mean_flow_magnitude"),
        "max_flow_magnitude": result.get("max_flow_magnitude"),
        "status": result["status"],
        "error": result.get("error"),
    }


def save_master_csv(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(csv_row(result))


def create_metric_plots(plots_dir: Path, results: list[dict]) -> None:
    """Save default-style before/after plots for successful consecutive pairs."""
    successful = [result for result in results if result["status"] == "success"]
    if not successful:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    x = [result["pair"] for result in successful]
    for name, before_key, after_key, ylabel, filename in (
        ("MSE", "original_mse", "warped_mse", "MSE (lower is better)", "mse_comparison.png"),
        ("MAE", "original_mae", "warped_mae", "MAE (lower is better)", "mae_comparison.png"),
        ("PSNR", "original_psnr", "warped_psnr", "PSNR in dB (higher is better)", "psnr_comparison.png"),
        ("SSIM", "original_ssim", "warped_ssim", "SSIM (higher is better)", "ssim_comparison.png"),
    ):
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(x, [float(result[before_key]) for result in successful], label="Before warp")
        axis.plot(x, [float(result[after_key]) for result in successful], label="After warp")
        axis.set_title(f"{name} before and after Farneback warp")
        axis.set_xlabel("Consecutive frame pair")
        axis.set_ylabel(ylabel)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=150)
        plt.close(fig)


def average_metric(results: list[dict], key: str) -> float | None:
    values = [float(result[key]) for result in results if result["status"] == "success" and result[key] is not None]
    return float(np.mean(values)) if values else None


def improvement_counts(results: list[dict]) -> dict:
    successful = [result for result in results if result["status"] == "success"]
    rules = {
        "mse": lambda before, after: after < before,
        "mae": lambda before, after: after < before,
        "psnr": lambda before, after: after > before,
        "ssim": lambda before, after: after > before,
    }
    return {
        metric: sum(rule(float(result[f"original_{metric}"]), float(result[f"warped_{metric}"])) for result in successful)
        for metric, rule in rules.items()
    }


def print_summary(frames: list[dict], results: list[dict], csv_path: Path) -> None:
    successful = [result for result in results if result["status"] == "success"]
    failed = len(results) - len(successful)
    averages = {
        metric: (average_metric(results, f"original_{metric}"), average_metric(results, f"warped_{metric}"))
        for metric in METRIC_NAMES
    }
    improved = improvement_counts(results)
    print("\nBATCH FARNEBACK EVALUATION COMPLETE")
    print("=" * 44)
    print(f"Frames found: {len(frames)}")
    print(f"Consecutive pairs: {len(results)}")
    print(f"Successful pairs: {len(successful)}")
    print(f"Failed pairs: {failed}")
    print("\nAverage metrics across successful pairs:")
    print(f"{'Metric':<8} {'Before':>16} {'After':>16}")
    for metric in METRIC_NAMES:
        before, after = averages[metric]
        before_text = f"{before:.8f}" if before is not None else "unavailable"
        after_text = f"{after:.8f}" if after is not None else "unavailable"
        print(f"{metric.upper():<8} {before_text:>16} {after_text:>16}")
    print("\nImprovement counts:")
    for metric in METRIC_NAMES:
        print(f"{metric.upper()}: {improved[metric]} / {len(successful)} pairs")
    print(f"\nResults saved to: {csv_path}")
    print("Metric gains describe image alignment on these pairs; they do not prove physically correct cloud motion.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing *_flow_input.nc files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root for batch results and plots")
    parser.add_argument("--save-visuals", action="store_true", help="Save per-pair flow and warp figures")
    args = parser.parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    results_dir = output_dir / "batch_results"
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    frames = discover_frames(input_dir)
    print_frame_inventory(frames)
    if len(frames) < 2:
        print("At least two flow-input frames are required to create a consecutive pair.")
        csv_path = output_dir / "batch_results.csv"
        save_master_csv(csv_path, [])
        print_summary(frames, [], csv_path)
        return

    results = []
    for pair_number, (first, second) in enumerate(zip(frames, frames[1:]), start=1):
        print(f"\nProcessing Pair {pair_number}: {first['path'].name} -> {second['path'].name}")
        try:
            result = process_pair(pair_number, first, second, results_dir, args.save_visuals)
            print(
                f"  Evaluation: {result['evaluation_pixel_count']:,} pixels "
                f"({result['evaluation_percentage']:.2f}%)"
            )
            print(
                f"  MSE {float(result['original_mse']):.8f} -> {float(result['warped_mse']):.8f}; "
                f"SSIM {float(result['original_ssim']):.6f} -> {float(result['warped_ssim']):.6f}"
            )
        except Exception as error:
            result = empty_pair_result(pair_number, first, second)
            result["error"] = str(error)
            print(f"  FAILED: {error}")
        results.append(result)
        save_json(results_dir / f"pair_{pair_number:03d}.json", result)

    csv_path = output_dir / "batch_results.csv"
    save_master_csv(csv_path, results)
    create_metric_plots(plots_dir, results)
    print_summary(frames, results, csv_path)
    if args.save_visuals:
        print(f"Per-pair visualizations saved under: {results_dir / 'visuals'}")


if __name__ == "__main__":
    main()
