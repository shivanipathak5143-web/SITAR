"""Compare a predicted midpoint with its held-out GOES flow-input frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    from .interpolate_farneback import PROJECT_ROOT, resolve_path
    from .validate_optical_flow import calculate_metrics, parse_timestamp
except ImportError:
    from interpolate_farneback import PROJECT_ROOT, resolve_path
    from validate_optical_flow import calculate_metrics, parse_timestamp

PREDICTED_VARIABLE = "predicted_frame"
REFERENCE_VARIABLE = "normalized_brightness_temperature"


def load_comparison_frame(path: Path, variable: str) -> dict:
    """Load one comparison raster and its mask/grid, releasing the file handle."""
    with xr.open_dataset(path) as ds:
        if variable not in ds or "valid_mask" not in ds:
            raise KeyError(f"{path.name} must contain {variable!r} and 'valid_mask'")
        image = ds[variable]
        mask = ds["valid_mask"]
        if image.dims != ("y", "x") or image.ndim != 2:
            raise ValueError(f"{variable} in {path.name} must have dimensions ('y', 'x')")
        if mask.dims != image.dims or mask.shape != image.shape:
            raise ValueError(f"valid_mask in {path.name} does not match {variable}")
        if not all(name in ds.coords for name in ("x", "y")):
            raise ValueError(f"{path.name} is missing x/y coordinates")
        return {
            "image": np.asarray(image.load().values, dtype=np.float32),
            "mask": np.asarray(mask.load().values),
            "x": np.asarray(ds["x"].values),
            "y": np.asarray(ds["y"].values),
            "x_attrs": dict(ds["x"].attrs),
            "y_attrs": dict(ds["y"].attrs),
            "attrs": dict(ds.attrs),
            "timestamp": parse_timestamp(ds["t"].values.reshape(-1)[0]) if "t" in ds.coords and ds["t"].size else None,
        }


def compare_grids(predicted: dict, reference: dict) -> None:
    if predicted["image"].shape != reference["image"].shape:
        raise ValueError("predicted and reference image shapes differ")
    for name in ("x", "y"):
        if predicted[name].shape != reference[name].shape or not np.allclose(predicted[name], reference[name], rtol=0, atol=1e-7):
            raise ValueError(f"predicted and reference {name} grids differ")
        if predicted[f"{name}_attrs"].get("units", "") != reference[f"{name}_attrs"].get("units", ""):
            raise ValueError(f"predicted and reference {name} coordinate units differ")


def save_figure(path: Path, image: np.ndarray, title: str, mask: np.ndarray, *, difference: bool = False) -> None:
    shown = np.ma.masked_where(~mask, image)
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.set_facecolor("black")
    cmap = "magma" if difference else "gray"
    kwargs = {"vmin": 0, "vmax": 1} if not difference else {"vmin": 0, "vmax": 1}
    plotted = axis.imshow(shown, origin="upper", cmap=cmap, **kwargs)
    fig.colorbar(plotted, ax=axis, label="Absolute normalized difference" if difference else "Normalized brightness temperature")
    axis.set_title(title)
    axis.set_xlabel("Column (pixels)")
    axis.set_ylabel("Row (pixels)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_comparison(path: Path, predicted: np.ndarray, reference: np.ndarray, difference: np.ndarray, mask: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, values, title, cmap, limits in zip(
        axes,
        (predicted, reference, difference),
        ("Predicted midpoint", "Held-out ground truth", "Absolute difference"),
        ("gray", "gray", "magma"),
        ((0, 1), (0, 1), (0, 1)),
    ):
        axis.set_facecolor("black")
        shown = np.ma.masked_where(~mask, values)
        plotted = axis.imshow(shown, origin="upper", cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_title(title)
        axis.set_xlabel("Column (pixels)")
        fig.colorbar(plotted, ax=axis, shrink=0.8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def validate_pair(predicted_path: Path, reference_path: Path, output_dir: Path, label: str | None = None) -> dict:
    predicted_path, reference_path, output_dir = map(resolve_path, (predicted_path, reference_path, output_dir))
    predicted = load_comparison_frame(predicted_path, PREDICTED_VARIABLE)
    reference = load_comparison_frame(reference_path, REFERENCE_VARIABLE)
    compare_grids(predicted, reference)
    for name, frame in (("predicted", predicted), ("reference", reference)):
        if not np.isfinite(frame["mask"]).all() or not np.isin(frame["mask"], (0, 1)).all():
            raise ValueError(f"{name} valid_mask is not binary")
    valid = (predicted["mask"] == 1) & (reference["mask"] == 1)
    valid &= np.isfinite(predicted["image"]) & np.isfinite(reference["image"])
    if not valid.any():
        raise ValueError("predicted and reference frames have no common valid pixels")
    for name, frame in (("predicted", predicted), ("reference", reference)):
        values = frame["image"][valid]
        if ((values < -1e-6) | (values > 1.0 + 1e-6)).any():
            raise ValueError(f"{name} normalized values fall outside [0, 1]")
    # Metrics compare only pixels marked as real observations in both frames.
    metrics = calculate_metrics(reference["image"], predicted["image"], valid)
    difference = np.abs(predicted["image"] - reference["image"])
    output_dir.mkdir(parents=True, exist_ok=True)
    time_label = reference["timestamp"].strftime("%H%M") if reference["timestamp"] else "unknown"
    label = label or time_label
    paths = {
        "predicted_image": output_dir / f"predicted_{label}.png",
        "ground_truth_image": output_dir / f"ground_truth_{label}.png",
        "difference_image": output_dir / f"difference_{label}.png",
        "comparison_image": output_dir / f"comparison_{label}.png",
    }
    save_figure(paths["predicted_image"], predicted["image"], "Predicted intermediate frame", predicted["mask"] == 1)
    save_figure(paths["ground_truth_image"], reference["image"], "Held-out ground-truth frame", reference["mask"] == 1)
    save_figure(paths["difference_image"], difference, "Absolute prediction error", valid, difference=True)
    save_comparison(paths["comparison_image"], predicted["image"], reference["image"], difference, valid)
    results = {
        "predicted_file": predicted_path.name,
        "ground_truth_file": reference_path.name,
        "ground_truth_timestamp": reference["timestamp"].isoformat() if reference["timestamp"] else "unavailable",
        "shape": list(predicted["image"].shape),
        "total_pixels": int(valid.size),
        "comparison_pixel_count": int(valid.sum()),
        "comparison_percentage_of_image": float(100.0 * valid.sum() / valid.size),
        "metrics": metrics,
        "plots": {name: path.name for name, path in paths.items()},
        "scientific_caveat": "These image metrics assess agreement with this held-out frame; they do not establish physically accurate cloud motion.",
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", required=True, type=Path, help="Predicted midpoint NetCDF")
    parser.add_argument("--ground-truth", required=True, type=Path, help="Held-out actual midpoint flow-input NetCDF")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for plots and metrics.json")
    parser.add_argument("--label", help="Optional filename label used in output PNG names")
    args = parser.parse_args()
    result = validate_pair(args.predicted, args.ground_truth, args.output_dir, args.label)
    print(f"Ground truth: {result['ground_truth_file']}")
    print(f"Total pixels: {result['total_pixels']:,}")
    print(f"Common valid pixels: {result['comparison_pixel_count']:,} ({result['comparison_percentage_of_image']:.2f}%)")
    print("Metrics (prediction vs held-out frame):")
    for key, value in result["metrics"].items():
        print(f"  {key.upper()}: {value:.8f}")
    print(f"Validation outputs: {resolve_path(args.output_dir)}")


if __name__ == "__main__":
    main()
