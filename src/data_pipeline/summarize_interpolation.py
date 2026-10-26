"""Summarize existing Farneback interpolation validation results."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "interpolation"
CSV_PATH = RESULTS_DIR / "farneback_baseline_results.csv"
SUMMARY_PATH = RESULTS_DIR / "farneback_baseline_summary.json"
PLOTS_DIR = RESULTS_DIR / "baseline_plots"
METRIC_NAMES = ("mse", "mae", "psnr", "ssim")
SCIENTIFIC_NOTE = (
    "These metrics evaluate agreement between the predicted intermediate frame and the "
    "held-out observed frame. They do not by themselves establish physically accurate cloud-motion estimation."
)
SCOPE_NOTE = "Farneback baseline performance across four held-out intermediate frames from the current GOES-19 C13 sequence."
CSV_COLUMNS = (
    "experiment",
    "input_frame_1",
    "ground_truth_frame",
    "input_frame_2",
    "ground_truth_timestamp",
    "total_pixels",
    "comparison_pixels",
    "comparison_percentage",
    "mse",
    "mae",
    "psnr",
    "ssim",
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def is_validation_record(value: dict) -> bool:
    metrics = value.get("metrics")
    return (
        isinstance(metrics, dict)
        and all(name in metrics for name in METRIC_NAMES)
        and all(
            key in value
            for key in (
                "predicted_file",
                "ground_truth_file",
                "total_pixels",
                "comparison_pixel_count",
                "comparison_percentage_of_image",
            )
        )
    )


def experiment_identifier(directory: Path, metadata: dict, validation: dict | None) -> str:
    identifier = metadata.get("experiment")
    if identifier is None and validation is not None:
        identifier = validation.get("experiment")
    if identifier is not None:
        return str(identifier)
    match = re.search(r"\d+", directory.name)
    return match.group(0) if match else directory.name


def discover_results() -> tuple[list[dict], list[dict], int]:
    records: list[dict] = []
    issues: list[dict] = []
    experiment_dirs = sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()) if RESULTS_DIR.exists() else []
    batch_summary = RESULTS_DIR / "batch_summary.json"
    batch_data = read_json(batch_summary) if batch_summary.exists() else {}

    for directory in experiment_dirs:
        json_files = list(directory.rglob("*.json"))
        metadata_path = next((path for path in json_files if path.name == "experiment.json"), None)
        failure_path = next((path for path in json_files if path.name == "failure.json"), None)
        validation_path = None
        validation = None
        for path in json_files:
            try:
                candidate = read_json(path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if is_validation_record(candidate):
                validation_path, validation = path, candidate
                break

        metadata = read_json(metadata_path) if metadata_path else {}
        identifier = experiment_identifier(directory, metadata, validation)
        if validation is None:
            if failure_path:
                failure = read_json(failure_path)
                issue = {
                    "experiment": identifier,
                    "status": "failed",
                    "reason": failure.get("error", "Validation JSON is missing."),
                }
            else:
                issue = {
                    "experiment": identifier,
                    "status": "missing_validation",
                    "reason": "No successful interpolation validation JSON was found.",
                }
            issues.append(issue)
            continue

        metadata = metadata or {}
        ground_truth_timestamp = (
            validation.get("ground_truth_timestamp")
            or metadata.get("ground_truth_timestamp")
        )
        try:
            timestamp = datetime.fromisoformat(str(ground_truth_timestamp).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            issues.append({
                "experiment": identifier,
                "status": "failed",
                "reason": f"Missing or invalid ground-truth timestamp in {validation_path}.",
            })
            continue

        numeric_metrics = {name: float(validation["metrics"][name]) for name in METRIC_NAMES}
        if not all(math.isfinite(value) for value in numeric_metrics.values()):
            issues.append({"experiment": identifier, "status": "failed", "reason": "Validation metrics contain non-finite values."})
            continue

        records.append({
            "experiment": identifier,
            "input_frame_1": metadata.get("frame_before", ""),
            "ground_truth_frame": validation["ground_truth_file"],
            "input_frame_2": metadata.get("frame_after", ""),
            "ground_truth_timestamp": str(ground_truth_timestamp),
            "_timestamp": timestamp,
            "total_pixels": int(validation["total_pixels"]),
            "comparison_pixels": int(validation["comparison_pixel_count"]),
            "comparison_percentage": float(validation["comparison_percentage_of_image"]),
            **numeric_metrics,
        })

    records.sort(key=lambda row: row["_timestamp"])
    planned = int(batch_data.get("experiments_planned", 0) or 0)
    planned = max(planned, len(records) + len(issues))
    if planned > len(records) + len(issues):
        present_ids = {issue["experiment"] for issue in issues}
        present_ids.update(row["experiment"] for row in records)
        for number in range(1, planned + 1):
            if str(number) not in present_ids:
                issues.append({
                    "experiment": str(number),
                    "status": "missing_experiment",
                    "reason": "No experiment directory or successful validation was found.",
                })
    return records, issues, planned


def save_csv(records: list[dict]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in CSV_COLUMNS})


def metric_summary(records: list[dict]) -> dict:
    summaries = {}
    for name in METRIC_NAMES:
        values = np.asarray([record[name] for record in records], dtype=float)
        summaries[name] = {
            "average": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return summaries


def plot_metrics(records: list[dict], summaries: dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [record["_timestamp"].strftime("%H:%M") for record in records]
    for name in METRIC_NAMES:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(labels, [record[name] for record in records], marker="o", linewidth=2)
        axis.set_title(f"{name.upper()} across held-out frames")
        axis.set_xlabel("Ground-truth time (UTC)")
        axis.set_ylabel(name.upper())
        axis.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{name}_by_experiment.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for axis, name in zip(axes.flat, METRIC_NAMES):
        average = summaries[name]["average"]
        axis.bar([name.upper()], [average], color="#287c8e")
        axis.set_title(f"Average {name.upper()}")
        axis.set_ylabel(name.upper())
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("Farneback baseline metric averages (separate scales)")
    fig.savefig(PLOTS_DIR / "metric_averages.png", dpi=150)
    plt.close(fig)


def print_report(records: list[dict], issues: list[dict], planned: int, summaries: dict, average_coverage: float) -> None:
    print("# FARNEBACK INTERPOLATION BASELINE")
    print(SCOPE_NOTE)
    print(f"Experiments found: {planned}")
    print(f"Successful experiments: {len(records)}")
    print(f"Failed/missing experiments: {len(issues)}")
    if issues:
        for issue in issues:
            print(f"  Experiment {issue['experiment']} ({issue['status']}): {issue['reason']}")

    print("\nExperiment Results:")
    print("Experiment | Ground Truth | MSE | MAE | PSNR | SSIM")
    for record in records:
        print(
            f"{record['experiment']:>10} | {record['_timestamp'].strftime('%H:%M')} | "
            f"{record['mse']:.10f} | {record['mae']:.10f} | {record['psnr']:.8f} | {record['ssim']:.8f}"
        )
    if not records:
        print("No successful validation records were available to summarize.")
        return

    print("\nOverall Baseline (arithmetic mean across experiments):")
    for name in METRIC_NAMES:
        print(f"Average {name.upper():<4}: {summaries[name]['average']:.10f}")
    print("\nStandard Deviation (population):")
    for name in METRIC_NAMES:
        print(f"{name.upper():<4}: {summaries[name]['standard_deviation']:.10f}")
    print("\nRanges:")
    for name in METRIC_NAMES:
        print(f"{name.upper():<4}: min {summaries[name]['minimum']:.10f} / max {summaries[name]['maximum']:.10f}")
    print(f"\nAverage comparison coverage: {average_coverage:.6f} %")


def main() -> None:
    records, issues, planned = discover_results()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(records)
    summaries = metric_summary(records) if records else {}
    average_coverage = float(np.mean([record["comparison_percentage"] for record in records])) if records else 0.0
    summary = {
        "number_of_experiments": planned,
        "successful_experiments": len(records),
        "failed_or_missing_experiments": len(issues),
        "failed_or_missing_details": issues,
        "metrics": summaries,
        "average_mse": summaries.get("mse", {}).get("average"),
        "average_mae": summaries.get("mae", {}).get("average"),
        "average_psnr": summaries.get("psnr", {}).get("average"),
        "average_ssim": summaries.get("ssim", {}).get("average"),
        "standard_deviation_mse": summaries.get("mse", {}).get("standard_deviation"),
        "standard_deviation_mae": summaries.get("mae", {}).get("standard_deviation"),
        "standard_deviation_psnr": summaries.get("psnr", {}).get("standard_deviation"),
        "standard_deviation_ssim": summaries.get("ssim", {}).get("standard_deviation"),
        "average_comparison_percentage": average_coverage,
        "average_method": "simple arithmetic mean across successful experiments; no weighted average",
        "scope": SCOPE_NOTE,
        "scientific_note": SCIENTIFIC_NOTE,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if records:
        plot_metrics(records, summaries)
    else:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print_report(records, issues, planned, summaries, average_coverage)
    print(f"\nCSV: {CSV_PATH}")
    print(f"Summary JSON: {SUMMARY_PATH}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()