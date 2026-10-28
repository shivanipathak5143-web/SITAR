"""Run Farneback midpoint interpolation for every chronological frame triplet."""

from __future__ import annotations

import argparse
import json
from datetime import timezone
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from .interpolate_farneback import DEFAULT_OUTPUT_DIR, interpolate_frames, resolve_path
    from .validate_interpolation import validate_pair
    from .validate_optical_flow import parse_timestamp
except ImportError:
    from interpolate_farneback import DEFAULT_OUTPUT_DIR, interpolate_frames, resolve_path
    from validate_interpolation import validate_pair
    from validate_optical_flow import parse_timestamp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "flow_input"
METRIC_NAMES = ("mse", "mae", "psnr", "ssim")


def read_frame_timestamp(path: Path):
    """Read only time metadata so batch discovery does not load image arrays."""
    with xr.open_dataset(path) as ds:
        for name in ("t", "time"):
            if name in ds.coords and ds[name].size:
                timestamp = parse_timestamp(np.asarray(ds[name].values).reshape(-1)[0])
                if timestamp is not None:
                    return timestamp
        for name in ("time_coverage_start", "timestamp", "start_time"):
            timestamp = parse_timestamp(ds.attrs.get(name))
            if timestamp is not None:
                return timestamp
    raise ValueError(f"timestamp unavailable in {path.name}; chronological sorting is required")


def sort_frames(paths: list[Path]) -> list[tuple[Path, object]]:
    stamped = [(path, read_frame_timestamp(path)) for path in paths]
    def sort_key(item):
        timestamp = item[1]
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        return timestamp
    return sorted(stamped, key=sort_key)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_batch(input_dir: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    input_dir, output_dir = resolve_path(input_dir), resolve_path(output_dir)
    files = sorted(input_dir.glob("*_flow_input.nc")) if input_dir.exists() else []
    print(f"Input directory: {input_dir}")
    print(f"Frames found: {len(files)}")
    if len(files) < 3:
        raise ValueError(f"at least 3 *_flow_input.nc frames are required; found {len(files)}")
    frames = sort_frames(files)
    succeeded: list[dict] = []
    failed: list[dict] = []
    for index in range(len(frames) - 2):
        before, target, after = frames[index][0], frames[index + 1][0], frames[index + 2][0]
        experiment_dir = output_dir / f"experiment_{index + 1:02d}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        predicted_path = experiment_dir / "predicted_intermediate.nc"
        label = frames[index + 1][1].strftime("%H%M")
        print(f"\nExperiment {index + 1:02d}: {before.name} + {after.name} -> predict {target.name}")
        try:
            # Prediction consumes endpoint frames only. The target is opened by
            # validation only after the predicted NetCDF has been written.
            interpolation = interpolate_frames(before, after, predicted_path)
            reference = {
                "ground_truth_file": target.name,
                "ground_truth_timestamp": frames[index + 1][1].isoformat(),
                "role": "Held-out reference; not used to generate the prediction.",
            }
            write_json(experiment_dir / "ground_truth_reference.json", reference)
            experiment_info = {
                "experiment": index + 1,
                "frame_before": before.name,
                "ground_truth_frame": target.name,
                "frame_after": after.name,
                "predicted_file": predicted_path.name,
                "predicted_timestamp": interpolation["timestamp"].isoformat() if interpolation["timestamp"] else "unavailable",
                "ground_truth_timestamp": frames[index + 1][1].isoformat(),
                "flow_source": interpolation["flow_source"],
                "scientific_caveat": "Farneback optical-flow-based baseline; metric agreement does not establish physically accurate cloud motion.",
            }
            write_json(experiment_dir / "experiment.json", experiment_info)
            metrics = validate_pair(predicted_path, target, experiment_dir, label)
            succeeded.append({
                "experiment": index + 1,
                "before": before.name,
                "target": target.name,
                "after": after.name,
                **metrics["metrics"],
                "directory": str(experiment_dir),
                "evaluation_pixels": metrics["comparison_pixel_count"],
            })
            print("  " + " | ".join(f"{name.upper()}={metrics['metrics'][name]:.8f}" for name in METRIC_NAMES))
            print(f"  Evaluation pixels: {metrics['comparison_pixel_count']:,} ({metrics['comparison_percentage_of_image']:.2f}%)")
            print(f"  Output: {experiment_dir}")
        except Exception as error:
            failed.append({"experiment": index + 1, "before": before.name, "target": target.name, "after": after.name, "error": str(error)})
            write_json(experiment_dir / "failure.json", failed[-1])
            print(f"  FAILED: {error}")
    return succeeded, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing *_flow_input.nc frames")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for interpolation experiments")
    args = parser.parse_args()
    succeeded, failed = run_batch(args.input_dir, args.output_dir)
    print("\nBATCH INTERPOLATION SUMMARY")
    print(f"Experiments planned: {len(succeeded) + len(failed)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    print("\nExperiment | Input Before | Ground Truth | Input After | MSE | MAE | PSNR | SSIM")
    for row in succeeded:
        print(f"{row['experiment']:>10} | {row['before']} | {row['target']} | {row['after']} | " + " | ".join(f"{row[name]:.8f}" for name in METRIC_NAMES))
    if succeeded:
        print("\nAVERAGE METRICS")
        for name in METRIC_NAMES:
            print(f"Average {name.upper()}: {float(np.mean([row[name] for row in succeeded])):.8f}")
    if failed:
        print("\nFailures:")
        for row in failed:
            print(f"  Experiment {row['experiment']:02d}: {row['error']}")
    summary_path = resolve_path(args.output_dir) / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, {
        "experiments_planned": len(succeeded) + len(failed),
        "succeeded": succeeded,
        "failed": failed,
        "average_metrics": {name: float(np.mean([row[name] for row in succeeded])) for name in METRIC_NAMES} if succeeded else {},
        "scientific_caveat": "Farneback baseline metrics do not prove physically correct cloud motion.",
    })
    print(f"Batch summary JSON: {summary_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
