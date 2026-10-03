"""
Tests for NetCDF I/O and normalization, using small synthetic .nc files
(no real satellite datasets required).
"""

import os
import numpy as np
import xarray as xr
import pytest

from src.interpolation.utils import (
    load_satellite_frame,
    normalize_frame,
    denormalize_frame,
    write_interpolated_frame_nc,
    NormalizationConfig,
)


@pytest.fixture
def synthetic_nc_file(tmp_path):
    def _make(filename="frame_0000.nc", fill_nan_corner=False):
        data = np.random.uniform(200, 300, size=(32, 32)).astype(np.float32)
        if fill_nan_corner:
            data[0:2, 0:2] = np.nan
        da = xr.DataArray(data, dims=("y", "x"), name="TIR1")
        da.attrs["units"] = "K"
        path = str(tmp_path / filename)
        da.to_dataset().to_netcdf(path)
        return path
    return _make


def test_load_satellite_frame_reads_correct_shape(synthetic_nc_file):
    path = synthetic_nc_file()
    data, coords, attrs = load_satellite_frame(path)
    assert data.shape == (32, 32)
    assert attrs["_source_variable"] == "TIR1"


def test_load_satellite_frame_handles_nan(synthetic_nc_file):
    path = synthetic_nc_file(fill_nan_corner=True)
    data, _, _ = load_satellite_frame(path)
    assert np.isnan(data[0, 0])  # NaN preserved at load time


def test_normalize_and_denormalize_roundtrip(synthetic_nc_file):
    path = synthetic_nc_file()
    data, _, _ = load_satellite_frame(path)
    cfg = NormalizationConfig(method="minmax", vmin=200.0, vmax=300.0)
    normed, meta = normalize_frame(data, cfg)

    assert normed.min() >= 0.0 and normed.max() <= 1.0

    recovered = denormalize_frame(normed, meta)
    np.testing.assert_allclose(recovered, np.clip(data, 200, 300), atol=1e-2)


def test_nan_fill_strategy_valid_mean(synthetic_nc_file):
    path = synthetic_nc_file(fill_nan_corner=True)
    data, _, _ = load_satellite_frame(path)
    cfg = NormalizationConfig(method="minmax", vmin=200.0, vmax=300.0, nan_fill_strategy="valid_mean")
    normed, _ = normalize_frame(data, cfg)
    assert not np.isnan(normed).any()


def test_write_interpolated_frame_nc_creates_valid_file(tmp_path):
    out_path = str(tmp_path / "interp.nc")
    data = np.random.uniform(200, 300, size=(16, 16)).astype(np.float32)

    write_interpolated_frame_nc(
        output_path=out_path,
        data=data,
        source_coords={},
        source_attrs={"units": "K"},
        interpolation_factor=0.5,
        model_name="satellite_rife",
        model_checkpoint="dummy.pth",
        source_frame_0="frame0.nc",
        source_frame_1="frame1.nc",
    )

    assert os.path.isfile(out_path)
    with xr.open_dataset(out_path) as ds:
        var = list(ds.data_vars)[0]
        assert ds[var].attrs["is_synthetic"] == "true"
        assert ds[var].attrs["source_frame_0"] == "frame0.nc"
        np.testing.assert_allclose(ds[var].values, data)


def test_write_refuses_to_overwrite_existing_file(tmp_path):
    out_path = str(tmp_path / "interp.nc")
    data = np.zeros((8, 8), dtype=np.float32)
    write_interpolated_frame_nc(
        out_path, data, {}, {}, 0.5, "m", "c.pth", "f0.nc", "f1.nc"
    )
    with pytest.raises(FileExistsError):
        write_interpolated_frame_nc(
            out_path, data, {}, {}, 0.5, "m", "c.pth", "f0.nc", "f1.nc"
        )