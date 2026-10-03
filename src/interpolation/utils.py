"""
Shared utilities for satellite frame interpolation:
- NetCDF / HDF5 reading and writing
- Normalization / denormalization
- Config loading
- Validation metrics
"""

from __future__ import annotations

import os
import json
import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import yaml

try:
    import xarray as xr
except ImportError as e:
    raise ImportError(
        "xarray is required for NetCDF I/O. Install with: pip install xarray netCDF4"
    ) from e

try:
    import h5py
except ImportError:
    h5py = None


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file is empty or invalid: {path}")
    return cfg


_COMMON_TIR_VAR_NAMES = [
    "TIR1", "TIR_1", "CMI", "Rad", "brightness_temperature",
    "tir", "IRBRTP", "IR1", "temp_11", "BT",
]


def _pick_variable_name(ds: "xr.Dataset", var_name: Optional[str]) -> str:
    if var_name is not None:
        if var_name not in ds.data_vars:
            raise KeyError(
                f"Requested variable '{var_name}' not found in file. "
                f"Available variables: {list(ds.data_vars)}"
            )
        return var_name

    for candidate in _COMMON_TIR_VAR_NAMES:
        if candidate in ds.data_vars:
            return candidate

    for name, da in ds.data_vars.items():
        dims = [d for d in da.dims if da.sizes[d] > 1]
        if len(dims) == 2 and np.issubdtype(da.dtype, np.number):
            return name

    raise ValueError(
        "Could not auto-detect the TIR image variable. "
        f"Please set 'image_variable' explicitly in your config. "
        f"Available variables: {list(ds.data_vars)}"
    )


def load_satellite_frame(
    path: str,
    var_name: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Load a single satellite TIR frame from a .nc or .h5 file.

    Returns
    -------
    data : np.ndarray, shape (H, W), float32, NaN for missing/fill values
    coords : dict with optional keys 'lat', 'lon', 'time'
    attrs : dict of file/variable-level attributes worth preserving
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Satellite frame file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in (".nc", ".nc4", ".netcdf"):
        return _load_nc_frame(path, var_name)
    elif ext in (".h5", ".hdf5"):
        return _load_h5_frame(path, var_name)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}' for {path}. Expected .nc or .h5"
        )


def _load_nc_frame(path: str, var_name: Optional[str]):
    with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
        vname = _pick_variable_name(ds, var_name)
        da = ds[vname]
        da = da.squeeze()

        if da.ndim != 2:
            raise ValueError(
                f"Expected a 2D image after squeezing, got shape {da.shape} "
                f"for variable '{vname}' in {path}. Check that the file "
                f"contains a single frame, not a time-stack."
            )

        data = da.values.astype(np.float32)

        fill_value = da.attrs.get("_FillValue", None)
        if fill_value is not None:
            data = np.where(np.isclose(data, float(fill_value)), np.nan, data)

        coords: Dict[str, Any] = {}
        if "lat" in ds.coords:
            coords["lat"] = ds.coords["lat"].values
        elif "latitude" in ds.coords:
            coords["lat"] = ds.coords["latitude"].values
        if "lon" in ds.coords:
            coords["lon"] = ds.coords["lon"].values
        elif "longitude" in ds.coords:
            coords["lon"] = ds.coords["longitude"].values
        if "time" in ds.coords:
            try:
                coords["time"] = ds.coords["time"].values.item()
            except Exception:
                coords["time"] = str(ds.coords["time"].values)

        attrs = dict(da.attrs)
        attrs["_source_variable"] = vname
        attrs["_source_file"] = os.path.basename(path)

    return data, coords, attrs


def _load_h5_frame(path: str, var_name: Optional[str]):
    if h5py is None:
        raise ImportError("h5py is not installed. Install with: pip install h5py")

    with h5py.File(path, "r") as f:
        if var_name is not None:
            if var_name not in f:
                raise KeyError(
                    f"Dataset '{var_name}' not found in {path}. "
                    f"Available keys: {list(f.keys())}"
                )
            dset_name = var_name
        else:
            candidates = [k for k in _COMMON_TIR_VAR_NAMES if k in f]
            if not candidates:
                candidates = [
                    k for k in f.keys()
                    if hasattr(f[k], "shape") and len(f[k].shape) == 2
                ]
            if not candidates:
                raise ValueError(
                    f"Could not auto-detect TIR dataset in {path}. "
                    f"Available keys: {list(f.keys())}"
                )
            dset_name = candidates[0]

        dset = f[dset_name]
        data = dset[()].astype(np.float32)
        if data.ndim != 2:
            data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data, got shape {data.shape} in {path}")

        attrs = {k: v for k, v in dset.attrs.items()}
        fill_value = attrs.get("_FillValue", None)
        if fill_value is not None:
            data = np.where(np.isclose(data, float(fill_value)), np.nan, data)

        attrs["_source_variable"] = dset_name
        attrs["_source_file"] = os.path.basename(path)

    return data, {}, attrs


@dataclass
class NormalizationConfig:
    """
    Configuration for consistent normalization across a sequence.

    method: one of "minmax", "percentile", "stats_file"
    vmin/vmax: used when method == "minmax"
    p_low/p_high: percentiles used when method == "percentile"
    stats_path: JSON file with {"vmin":..., "vmax":...} used when method == "stats_file"
    """
    method: str = "percentile"
    vmin: float = 180.0
    vmax: float = 320.0
    p_low: float = 1.0
    p_high: float = 99.0
    stats_path: Optional[str] = None
    nan_fill_strategy: str = "valid_mean"
    nan_fill_constant: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NormalizationConfig":
        return NormalizationConfig(**{k: v for k, v in d.items() if v is not None})


def _fill_nans(data: np.ndarray, cfg: NormalizationConfig) -> np.ndarray:
    if not np.isnan(data).any():
        return data
    data = data.copy()
    nan_mask = np.isnan(data)
    if nan_mask.all():
        raise ValueError("Frame is entirely NaN/missing — cannot process.")

    if cfg.nan_fill_strategy == "constant":
        data[nan_mask] = cfg.nan_fill_constant
    else:
        valid_mean = float(np.nanmean(data))
        data[nan_mask] = valid_mean
    return data


def normalize_frame(
    data: np.ndarray, cfg: NormalizationConfig
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize a single frame to roughly [0, 1] using a FIXED, sequence-consistent
    range (not per-frame min/max), so temporal intensity relationships between
    frames are preserved rather than each frame being independently stretched.
    """
    data = _fill_nans(data, cfg)

    if cfg.method == "minmax":
        vmin, vmax = cfg.vmin, cfg.vmax
    elif cfg.method == "percentile":
        vmin = float(np.percentile(data, cfg.p_low))
        vmax = float(np.percentile(data, cfg.p_high))
    elif cfg.method == "stats_file":
        if cfg.stats_path is None or not os.path.isfile(cfg.stats_path):
            raise FileNotFoundError(
                f"stats_path '{cfg.stats_path}' not found for method='stats_file'"
            )
        with open(cfg.stats_path, "r") as f:
            stats = json.load(f)
        vmin, vmax = float(stats["vmin"]), float(stats["vmax"])
    else:
        raise ValueError(f"Unknown normalization method: {cfg.method}")

    if vmax <= vmin:
        raise ValueError(f"Invalid normalization range: vmin={vmin}, vmax={vmax}")

    normed = (data - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0.0, 1.0).astype(np.float32)

    norm_meta = {"vmin": vmin, "vmax": vmax, "method": cfg.method}
    return normed, norm_meta


def denormalize_frame(normed: np.ndarray, norm_meta: Dict[str, float]) -> np.ndarray:
    """Invert normalize_frame using the stored vmin/vmax."""
    vmin, vmax = norm_meta["vmin"], norm_meta["vmax"]
    return (normed * (vmax - vmin) + vmin).astype(np.float32)


def write_interpolated_frame_nc(
    output_path: str,
    data: np.ndarray,
    source_coords: Dict[str, Any],
    source_attrs: Dict[str, Any],
    interpolation_factor: float,
    model_name: str,
    model_checkpoint: str,
    source_frame_0: str,
    source_frame_1: str,
    var_name: str = "TIR1_interpolated",
) -> None:
    """
    Write a generated intermediate frame to NetCDF, preserving whatever
    coordinate/attribute information was available from the source frame.
    """
    if os.path.exists(output_path):
        raise FileExistsError(
            f"Refusing to silently overwrite existing file: {output_path}. "
            f"Choose a different --output path."
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dims = ("y", "x")
    coords = {}
    if "lat" in source_coords:
        coords["lat"] = (("y", "x") if source_coords["lat"].ndim == 2 else ("y",), source_coords["lat"])
    if "lon" in source_coords:
        coords["lon"] = (("y", "x") if source_coords["lon"].ndim == 2 else ("x",), source_coords["lon"])

    da = xr.DataArray(data, dims=dims, coords=coords, name=var_name)

    da.attrs["long_name"] = "AI-interpolated thermal infrared brightness temperature"
    da.attrs["source_frame_0"] = os.path.basename(source_frame_0)
    da.attrs["source_frame_1"] = os.path.basename(source_frame_1)
    da.attrs["interpolation_factor"] = float(interpolation_factor)
    da.attrs["model_name"] = model_name
    da.attrs["model_checkpoint"] = os.path.basename(model_checkpoint)
    da.attrs["creation_method"] = "AI/ML frame interpolation (RIFE-based, fine-tuned on satellite TIR)"
    da.attrs["creation_time_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    da.attrs["is_synthetic"] = "true"

    for key in ("units", "standard_name", "grid_mapping", "sensor"):
        if key in source_attrs:
            da.attrs[key] = source_attrs[key]

    ds_out = da.to_dataset()
    ds_out.attrs["title"] = "Synthetic intermediate satellite frame (AI-generated)"
    ds_out.attrs["disclaimer"] = (
        "This frame was NOT directly observed. It was generated by an AI/ML "
        "frame interpolation model from two real observed frames. Not all "
        "original satellite metadata could be generically preserved; only "
        "compatible fields were copied from source_frame_0."
    )

    ds_out.to_netcdf(output_path)


def compute_metrics(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> Dict[str, float]:
    """
    Compute MSE, MAE, PSNR, SSIM between a predicted and target frame (2D numpy
    arrays, same normalized range). `data_range` must match the actual value
    range of pred/target (1.0 if both are in normalized [0,1]).
    """
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio

    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    psnr = float(peak_signal_noise_ratio(target, pred, data_range=data_range))
    ssim = float(structural_similarity(target, pred, data_range=data_range))

    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}