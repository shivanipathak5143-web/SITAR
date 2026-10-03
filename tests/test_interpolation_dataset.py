"""
Tests for SatelliteFrameDataset, using a small set of synthetic .nc frames
written to a temp directory with timestamp-pattern filenames.
"""

import numpy as np
import xarray as xr
import pytest

from src.interpolation.dataset import SatelliteFrameDataset
from src.interpolation.utils import NormalizationConfig


@pytest.fixture
def synthetic_sequence_dir(tmp_path):
    """Creates 6 sequential synthetic frames, 10 minutes apart."""
    timestamps = [f"202601010{h:02d}00" for h in range(0, 60, 10)][:6]
    for i, ts in enumerate(timestamps):
        data = np.full((64, 64), fill_value=250.0 + i, dtype=np.float32)  # distinct per-frame value
        da = xr.DataArray(data, dims=("y", "x"), name="TIR1")
        path = tmp_path / f"frame_{ts}.nc"
        da.to_dataset().to_netcdf(str(path))
    return str(tmp_path)


def test_dataset_builds_triplets_in_order(synthetic_sequence_dir):
    ds = SatelliteFrameDataset(
        data_dir=synthetic_sequence_dir,
        image_size=32,
        crop_size=None,
        stride=1,
        split="train",
        val_fraction=0.34,  # leaves ~4 train frames, 2 val -> at least 1 val triplet not guaranteed; see note below
        augment=False,
        normalization_cfg=NormalizationConfig(method="minmax", vmin=245, vmax=260),
    )
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["frame0"].shape == (1, 32, 32)
    assert sample["target"].shape == (1, 32, 32)
    assert sample["frame1"].shape == (1, 32, 32)
    assert sample["t"] == 0.5


def test_dataset_respects_temporal_order_not_random(synthetic_sequence_dir):
    ds = SatelliteFrameDataset(
        data_dir=synthetic_sequence_dir,
        image_size=32,
        stride=1,
        split="train",
        val_fraction=0.0,
        augment=False,
        normalization_cfg=NormalizationConfig(method="minmax", vmin=245, vmax=260),
    )
    # Because each synthetic frame has a distinct constant value (250+i),
    # target should always be numerically between frame0 and frame1's values.
    sample = ds[0]
    f0_val = sample["frame0"].mean().item()
    ft_val = sample["target"].mean().item()
    f1_val = sample["frame1"].mean().item()
    assert min(f0_val, f1_val) <= ft_val <= max(f0_val, f1_val)


def test_dataset_raises_on_too_few_frames(tmp_path):
    data = np.zeros((16, 16), dtype=np.float32)
    da = xr.DataArray(data, dims=("y", "x"), name="TIR1")
    da.to_dataset().to_netcdf(str(tmp_path / "frame_20260101000000.nc"))
    da.to_dataset().to_netcdf(str(tmp_path / "frame_20260101001000.nc"))

    with pytest.raises(ValueError):
        SatelliteFrameDataset(data_dir=str(tmp_path), split="train")