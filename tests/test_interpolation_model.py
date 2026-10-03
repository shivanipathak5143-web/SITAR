"""
Basic shape/contract tests for SatelliteRIFE. No real satellite data required.
"""

import torch
from src.interpolation.models.rife_model import SatelliteRIFE
from src.interpolation.models.warp import backward_warp


def test_warp_zero_flow_is_identity():
    img = torch.rand(1, 1, 32, 32)
    flow = torch.zeros(1, 2, 32, 32)
    warped = backward_warp(img, flow)
    assert torch.allclose(img, warped, atol=1e-5)


def test_model_accepts_single_channel_and_outputs_single_channel():
    model = SatelliteRIFE(in_channels=1, out_channels=1, base_channels=16, num_scales=2)
    frame0 = torch.rand(2, 1, 64, 64)
    frame1 = torch.rand(2, 1, 64, 64)

    out = model(frame0, frame1, t=0.5)

    assert out["pred"].shape == (2, 1, 64, 64)
    assert out["flow_t0"].shape == (2, 2, 64, 64)
    assert out["flow_t1"].shape == (2, 2, 64, 64)
    assert out["mask"].shape == (2, 1, 64, 64)
    assert torch.all(out["pred"] >= 0) and torch.all(out["pred"] <= 1)


def test_model_rejects_wrong_channel_count():
    model = SatelliteRIFE(in_channels=1, out_channels=1, base_channels=16, num_scales=2)
    frame0 = torch.rand(1, 3, 64, 64)  # RGB-shaped input, should be rejected
    frame1 = torch.rand(1, 3, 64, 64)

    try:
        model(frame0, frame1)
        assert False, "Expected ValueError for wrong channel count"
    except ValueError:
        pass


def test_model_with_external_flow_initialization():
    model = SatelliteRIFE(
        in_channels=1, out_channels=1, base_channels=16, num_scales=2,
        use_external_flow_init=True,
    )
    frame0 = torch.rand(1, 1, 64, 64)
    frame1 = torch.rand(1, 1, 64, 64)
    flow_01 = torch.zeros(1, 2, 64, 64)
    flow_10 = torch.zeros(1, 2, 64, 64)

    out = model(frame0, frame1, t=0.5, flow_01=flow_01, flow_10=flow_10)
    assert out["pred"].shape == (1, 1, 64, 64)