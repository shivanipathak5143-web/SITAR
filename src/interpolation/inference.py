"""
Run a trained model on two real satellite frames to produce a synthetic
intermediate frame, written back out as NetCDF.

Usage:
    python -m src.interpolation.inference \
        --frame0 data/sample/frame_0000.nc \
        --frame1 data/sample/frame_0020.nc \
        --checkpoint checkpoints/interpolation_best.pth \
        --output data/processed/frame_0010.nc
"""

from __future__ import annotations
import argparse
from typing import Optional

import torch

from .models.rife_model import SatelliteRIFE
from .utils import (
    load_satellite_frame,
    normalize_frame,
    denormalize_frame,
    write_interpolated_frame_nc,
    NormalizationConfig,
)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    if not checkpoint_path:
        raise ValueError("checkpoint path is required for inference.")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    m = cfg["model"]

    model = SatelliteRIFE(
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        base_channels=m["base_channels"],
        num_scales=m["num_scales"],
        use_external_flow_init=m["use_external_flow_init"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def run_inference(
    frame0_path: str,
    frame1_path: str,
    checkpoint_path: str,
    output_path: str,
    t: float = 0.5,
    var_name: Optional[str] = None,
    device_str: str = "auto",
) -> None:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model, cfg = load_model_from_checkpoint(checkpoint_path, device)
    norm_cfg = NormalizationConfig.from_dict(cfg["data"]["normalization"])
    resolved_var_name = var_name or cfg["data"]["image_variable"]

    data0, coords0, attrs0 = load_satellite_frame(frame0_path, var_name=resolved_var_name)
    data1, _, _ = load_satellite_frame(frame1_path, var_name=resolved_var_name)

    if data0.shape != data1.shape:
        raise ValueError(
            f"frame0 and frame1 have mismatched shapes: {data0.shape} vs {data1.shape}. "
            f"Both input frames must cover the same spatial grid."
        )

    normed0, norm_meta = normalize_frame(data0, norm_cfg)
    normed1, _ = normalize_frame(data1, norm_cfg)

    t0 = torch.from_numpy(normed0).unsqueeze(0).unsqueeze(0).float().to(device)
    t1 = torch.from_numpy(normed1).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(t0, t1, t=t)
        pred_normed = out["pred"].squeeze(0).squeeze(0).cpu().numpy()

    pred_physical = denormalize_frame(pred_normed, norm_meta)

    write_interpolated_frame_nc(
        output_path=output_path,
        data=pred_physical,
        source_coords=coords0,
        source_attrs=attrs0,
        interpolation_factor=t,
        model_name=cfg["model"]["name"],
        model_checkpoint=checkpoint_path,
        source_frame_0=frame0_path,
        source_frame_1=frame1_path,
    )
    print(f"Wrote interpolated frame to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference to generate an interpolated satellite frame.")
    parser.add_argument("--frame0", type=str, required=True)
    parser.add_argument("--frame1", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--t", type=float, default=0.5, help="Interpolation time in (0,1), default 0.5")
    parser.add_argument("--var-name", type=str, default=None, help="Override auto-detected variable name")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    run_inference(
        frame0_path=args.frame0,
        frame1_path=args.frame1,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        t=args.t,
        var_name=args.var_name,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()